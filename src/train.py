'''
By Real2CAD group at ETH Zurich
3DV Group 14
Yue Pan, Yuanwen Yue, Bingxin Ke, Yujie He

Editted based on the codes of the JointEmbedding paper (https://github.com/xheon/JointEmbedding)

As for our contributions, please check our report
'''
'''
Operations to follow:

Method 1 (suggested for trial and debugging)
1. module load tmux (optional)
2. sh ./scripts/start_interactive_normal.sh (and then wait)
3. tmux (optional, for doing something in parallel with the training)
4. bash run.sh
5. check the wandb dashboards and decide if you'd like to make an early stop (modifying and then try again without terminating the job)


Method 2 (suggested for a longer training sequence after you make sure there's no stupid error)
1. sh ./scripts/submit_job_[xxx].sh
2. then just wait and check the wandb dashboards
'''

import argparse
import json
import os
import random
import numpy as np
from collections import defaultdict
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, utils
import wandb
# import ptflops

from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses as metric_losses
from pytorch_metric_learning import miners

from models import *
import data
import utils
from test import evaluate_retrieval_metrics

category_name_dict = {"03001627": "Chair", "04379243": "Table", "02747177": "Trash bin", "02818832": "Bed",
                      "02871439": "Bookshelf", "02933112": "Cabinet", "04256520": "Sofa", "other": "Other"}
category_idx_dict_cls = {"03001627": 0, "04379243": 1, "02747177": 2, "02818832": 3, "02871439": 4, "02933112": 5,
                         "04256520": 6, "other": 7}  # classification
category_num = len(category_idx_dict_cls)
symmetry_num = 4


def forward(scan, cad, negative, positive, separation_model, completion_model, multihead_model, embedding_model, device,
            criterion_separation, criterion_completion, criterion_classification,
            criterion_retrieval, criterion_retrieval_name, offline_sample, retrieval_miner,
            weight_task, separation_model_on=True, balanced_weight=True, wb_visualize_on=False, vis_name="train/",
            train_symmetry: bool = True, criterion_symmetry=None,
            train_scale: bool = True, criterion_scale=None, scale_range=(0.5, 2)):
    # Prepare scan sample
    scan_fg_mask, scan_name = scan["mask"], scan["name"]
    if separation_model_on:
        scan_model = scan["content"]

    batch_size = np.shape(scan_fg_mask)[0]
    # 32*32*32 in our case
    voxel_size = 1.0 * np.shape(scan_fg_mask)[2] * np.shape(scan_fg_mask)[3] * np.shape(scan_fg_mask)[4]

    # remap to 0 or 1 (due to the interpolation during rotation augmentation)
    scan_fg_mask = torch.where(scan_fg_mask > 0.5, torch.ones(scan_fg_mask.shape), torch.zeros(scan_fg_mask.shape))
    # make sure that the functions work for both ths tdf and binary occupancy case
    if separation_model_on:
        # fg_mask (foreground mask) must lay in the scan's occupancy region
        scan_fg_mask = torch.where(scan_model == 0, torch.zeros(scan_fg_mask.shape), scan_fg_mask)
        scan_model = scan_model.to(device, non_blocking=True)
        # positive_count_scan = len(torch.nonzero(scan_model))
        positive_count_scan = len(scan_model[scan_model > 0.5])

    scan_fg_mask = scan_fg_mask.to(device, non_blocking=True)  # 1 for foreground, 0 for others (background or free)
    # positive_count_fg = len(torch.nonzero(scan_fg_mask)) # unit: voxel
    positive_count_fg = len(scan_fg_mask[scan_fg_mask > 0.5])  # unit: voxel

    weight_positive_fg = (batch_size * voxel_size - positive_count_fg) / positive_count_fg  # balanced weight

    # Prepare CAD sample
    cad_model, cad_name = cad["content"], cad["name"]
    cad_model = cad_model.to(device, non_blocking=True)

    # positive_count_cad = len(torch.nonzero(cad_model)) # fix: for tdf, cad_model can be between 0 and 1, it's better use 0.5 instead of 0 to count the positive number
    positive_count_cad = len(cad_model[cad_model > 0.5])  # unit: voxel
    weight_positive_cad = (batch_size * voxel_size - positive_count_cad) / positive_count_cad

    # Prepare negative sample
    negative_model, negative_name = negative["content"], negative["name"]
    negative_model = negative_model.to(device, non_blocking=True)

    # Prepare positive sample
    positive_model, positive_name = positive["content"], positive["name"]
    positive_model = positive_model.to(device, non_blocking=True)

    # load task specific weight
    weight_fgs, weight_bgs, weight_com, weight_cls, weight_sym, weight_scale, weight_retr = weight_task

    # Pass data through networks
    # Use balanced BCE with weight for 0 and 1 (consider to do this in each sample of the batch)
    # 1) Separate foreground 
    if separation_model_on:  # predict foreground from the object proposal
        foreground, _ = separation_model(scan_model)  # size:[batch_size, 1, 32, 32, 32]
        # TODO: consider use sigmoid -> BCE & softmax -> NLL (reference: Scan2CAD)
        if balanced_weight:
            criterion_separation_fg = nn.BCEWithLogitsLoss(reduction="none",
                                                           pos_weight=torch.tensor(weight_positive_fg))
        else:
            criterion_separation_fg = criterion_separation  # no weight
        # Attention! sigmoid already involved in BCEWithLogitsLoss, you do not need to apply sigmoid before the function
        # size: [batch_size, 1, 32, 32, 32] -> 1
        loss_foreground = torch.mean(criterion_separation_fg(foreground, scan_fg_mask))

        loss_foreground = weight_fgs * loss_foreground

        foreground = torch.sigmoid(foreground)  # -> [0,1]
    else:
        loss_foreground = 0
        foreground = scan_fg_mask  # directly use known foreground mask

    # 2) Complete foreground w.r.t. CAD model
    # Please take attetion that the CAD model may not be aligned well with the scan object (so you need to consider this since it's voxel representation)
    completed, hidden = completion_model(foreground)  # classification
    if balanced_weight:
        criterion_separation_com = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(weight_positive_cad))
    else:
        criterion_separation_com = criterion_completion  # no weight
    # Attention! sigmoid already involved in BCEWithLogitsLoss, you do not need to apply sigmoid before the function
    loss_completion = torch.mean(criterion_separation_com(completed, cad_model))  # size:[batch_size, 1, 32, 32, 32]
    loss_completion = weight_com * loss_completion

    completed = torch.sigmoid(completed)  # -> [0,1]

    # filter the completed voxel
    # completed = torch.where(completed > 0.5, completed, torch.zeros(completed.shape))

    # 3) Classification loss (extract feature <hidden> from bottleneck) # TODO: consider add more heads (symmetry property, scale, etc.)
    # pred_cat, pred_sym, pred_scale = multihead_model(torch.sigmoid(hidden)) # hidden - size: [batch_size, 256, 1, 1, 1]
    multihead_pred = multihead_model(torch.sigmoid(hidden))  # hidden - size: [batch_size, 256, 1, 1, 1]
    # category classification
    pred_cat = multihead_pred['category']
    pred_cat = pred_cat.view(-1, category_num, 1)
    positive_cat = torch.tensor([utils.get_category_idx(temp_name.split("/")[0]) for temp_name in cad_name])  # classifier takes input from 0
    positive_cat = positive_cat.to(device)
    labels = positive_cat[0:int(batch_size / 2 - 1)].repeat(2)  # [only used in online triplet sampling] we only use half of the scan segment and its corresponding cad model to keep the batch size same as the input
    positive_cat = positive_cat.view(-1, 1)
    # category classification loss
    loss_classification = criterion_classification(pred_cat, positive_cat)  # [0 - 7]
    loss_classification = torch.mean(loss_classification)
    loss_classification = weight_cls * loss_classification
    # category classification accuracy
    _, pred_cat_y = torch.max(pred_cat, 1)
    acc_classification = torch.eq(pred_cat_y, positive_cat).sum().item() / pred_cat_y.size()[0]
    # symmetry classification
    if train_symmetry:
        pred_sym = multihead_pred['symmetry']
        pred_sym = pred_sym.view(-1, symmetry_num, 1)
        positive_sym = torch.tensor([utils.get_symmetric_idx(_name) for _name in scan['symmetry']])
        positive_sym = positive_sym.to(device)
        positive_sym = positive_sym.view(-1, 1)
        # loss
        loss_symmetry = criterion_symmetry(pred_sym, positive_sym)
        loss_symmetry = torch.mean(loss_symmetry)
        loss_symmetry = weight_sym * loss_symmetry
        # accuracy
        _, pred_sym_y = torch.max(pred_sym, 1)
        acc_symmetry = torch.eq(pred_sym_y, positive_sym).sum().item() / pred_cat_y.size()[0]
    else:
        loss_symmetry = 0 * loss_classification  # * loss_classification because it's a batch
        acc_symmetry = 0
    # scale regression
    if train_scale:
        pred_scale = multihead_pred['scale']
        pred_scale = pred_scale.view(-1, 3)
        # print(f"pred_scale: {pred_scale}, \tscan['scale']: {scan['scale']}")
        gt_scale = scan['scale']
        gt_scale = gt_scale.to(device)
        # loss
        loss_scale = criterion_scale(pred_scale, gt_scale)
        loss_scale = torch.mean(loss_scale)
        loss_scale = weight_scale * loss_scale
        # ratio
        ratio_scale = pred_scale / gt_scale
        ratio_scale = torch.mean(ratio_scale, 0)
    else:
        loss_scale = 0 * loss_classification
        ratio_scale = [0, 0, 0]

    # 4) Embed completed output as a triplet
    # anchor: completed output, positive: CAD model, negative: random CAD sample
    if offline_sample:
        # anchor, positive, negative = embedding_model(completed, cad_model, negative_model)  # size:[batch_size, 256, 1, 1, 1]
        anchor, positive, negative = embedding_model(completed, positive_model, negative_model)
        a, p, n = anchor.view(anchor.shape[0], -1), positive.view(anchor.shape[0], -1), negative.view(anchor.shape[0],
                                                                                                      -1)  # size:[batch_size, 256]
        if criterion_retrieval_name == "tripletMargin":
            loss_retrieval = criterion_retrieval(a, p, n).mean()  # size:[batch_size, 256]->1
        else:  # circle loss
            loss_retrieval = criterion_retrieval(*measure_similarity(a, p, n))
    else:  # online
        embeddings = embedding_model(completed, cad_model)  # size:[batch_size, 256, 1, 1, 1]
        embeddings = embeddings.view(embeddings.shape[0], -1)
        if criterion_retrieval_name == "tripletMargin":
            hard_pairs = retrieval_miner(embeddings, labels)  # Batch_hard miner For each element in the batch, this miner will find the hardest positive and hardest negative, and use those to form a single triplet. So for a batch size of N, this miner will output N triplets. (now we use semi-hard pairs)
            loss_retrieval = criterion_retrieval(embeddings, labels, hard_pairs)  # it may be better to have approximately equal object for each class in the mini-batch
        else:  # circle loss
            loss_retrieval = criterion_retrieval(embeddings, labels)

    loss_retrieval = weight_retr * loss_retrieval

    if wb_visualize_on:
        # visualization in wandb (show the first sample in the batch):
        wb_vis_obj = {}
        if separation_model_on:
            scan_voxel = scan_model[0].cpu().detach().numpy().reshape((32, 32, 32))
            foreground_voxel = foreground[0].cpu().detach().numpy().reshape((32, 32, 32))  # prediction
            scan_voxel_wb = data.voxel2point(scan_voxel, name=scan_name[0])
            foreground_voxel_wb = data.voxel2point(foreground_voxel, color_mode='prob')
            wb_vis_obj[vis_name + "scan object"] = wandb.Object3D(scan_voxel_wb)
            wb_vis_obj[vis_name + "foreground prediction"] = wandb.Object3D(foreground_voxel_wb)

        completed_voxel = completed[0].cpu().detach().numpy().reshape((32, 32, 32))  # prediction
        foreground_voxel_gt = scan_fg_mask[0].cpu().detach().numpy().reshape((32, 32, 32))
        completed_voxel_gt = cad_model[0].cpu().detach().numpy().reshape((32, 32, 32))
        positive_voxel = positive_model[0].cpu().detach().numpy().reshape((32, 32, 32))
        negative_voxel = negative_model[0].cpu().detach().numpy().reshape((32, 32, 32))

        positive_cat = utils.wandb_color_lut(cad_name[0].split("/")[0])
        positive_cat_name = utils.get_category_name(cad_name[0].split("/")[0])
        negative_cat = utils.wandb_color_lut(negative_name[0].split("/")[0])
        negative_cat_name = utils.get_category_name(negative_name[0].split("/")[0])

        completed_voxel_wb = data.voxel2point(completed_voxel, color_mode='prob', visualize_prob_threshold=0.75)
        foreground_mask_wb = data.voxel2point(foreground_voxel_gt, positive_cat)
        completed_voxel_gt_wb = data.voxel2point(completed_voxel_gt, positive_cat,
                                                 name=positive_cat_name + ":" + cad_name[0])
        positive_voxel_wb = data.voxel2point(positive_voxel, positive_cat,
                                             name=positive_cat_name + ":" + positive_name[0])
        negative_voxel_wb = data.voxel2point(negative_voxel, negative_cat,
                                             name=negative_cat_name + ":" + negative_name[0])

        wb_vis_obj[vis_name + "foreground mask"] = wandb.Object3D(foreground_mask_wb)
        wb_vis_obj[vis_name + "completion prediction"] = wandb.Object3D(completed_voxel_wb)
        wb_vis_obj[vis_name + "completion gt (annotated cad)"] = wandb.Object3D(completed_voxel_gt_wb)
        wb_vis_obj[vis_name + "positive cad"] = wandb.Object3D(positive_voxel_wb)
        wb_vis_obj[vis_name + "negative cad"] = wandb.Object3D(negative_voxel_wb)

        wandb.log(wb_vis_obj)

    return loss_foreground, loss_completion, loss_classification, loss_symmetry, loss_scale, loss_retrieval, acc_classification, acc_symmetry, ratio_scale


def main(opt: argparse.Namespace) -> None:
    utils.set_gpu(opt.gpu)
    device = torch.device("cuda")

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # begining timestamp
    run_name = opt.name + "_" + ts  # modified to a name that is easier to index
    run_path = os.path.join(opt.output_root, run_name)
    if not os.path.exists(run_path):
        os.mkdir(run_path)
    assert os.access(run_path, os.W_OK)
    print(f"Start training {run_path}")
    print(vars(opt))

    # Set wandb
    if opt.wandb_vis_on:
        utils.setup_wandb()
        # save_dir=opt.output_root # figure out
        wandb.init(project="ScanCADJoint", entity="real2cad", config=vars(opt), dir=run_path)  # team 'real2cad'
        # wandb.init(project="ScanCADJoint", config=vars(opt), dir=run_path) # your own worksapce
        wandb.run.name = run_name

    # Save config
    os.makedirs(run_path, exist_ok=True)
    config_log = os.path.join(run_path, "config.json")
    with open(config_log, "w") as f:
        json.dump(vars(opt), f, indent=4)

    if opt.representation == "tdf":
        trans = data.truncation_normalization_transform
    else:  # binary_occupancy
        trans = data.to_occupancy_grid

    scan_base_path = None
    if opt.scan_dataset_name == "scannet":
        scan_base_path = opt.scannet_path
    elif opt.scan_dataset_name == "2d3ds":
        scan_base_path = opt.s2d3ds_path

    # Prepare Data
    # TODO: combine them into one common function
    train_dataset: Dataset = data.TrainingDataset(scan2cad_file=opt.scan2cad_file, scan_base_path=scan_base_path,
                                                  cad_base_path=opt.shapenet_voxel_path,
                                                  quat_file=opt.scan2cad_quat_file,
                                                  full_annotation_file=opt.full_annotation_file,
                                                  model_pool_file=opt.modelpool_file,
                                                  scan_dataset_name=opt.scan_dataset_name,
                                                  splits=["train"], input_only_mask=not opt.separation_model_on,
                                                  transformation=trans, align_to_scan=opt.align_to_scan,
                                                  rotation_aug=opt.rotation_augmentation,
                                                  flip_aug=opt.flip_augmentation,
                                                  jitter_aug=opt.jitter_augmentation,
                                                  rotation_aug_count=opt.rotation_trial_count,
                                                  add_negatives=True,
                                                  negative_sample_strategy=opt.negative_sample_strategy)

    train_dataloader: DataLoader = DataLoader(train_dataset, shuffle=True, batch_size=opt.batch_size,
                                              num_workers=opt.num_workers, pin_memory=True)

    val_dataset: Dataset = data.TrainingDataset(scan2cad_file=opt.scan2cad_file, scan_base_path=scan_base_path,
                                                cad_base_path=opt.shapenet_voxel_path, quat_file=opt.scan2cad_quat_file,
                                                full_annotation_file=opt.full_annotation_file,
                                                model_pool_file=opt.modelpool_file,
                                                scan_dataset_name=opt.scan_dataset_name,
                                                splits=["validation"], input_only_mask=not opt.separation_model_on,
                                                transformation=trans, align_to_scan=opt.align_to_scan,
                                                rotation_aug=opt.rotation_augmentation, flip_aug=opt.flip_augmentation,
                                                jitter_aug=opt.jitter_augmentation,
                                                rotation_aug_count=opt.rotation_trial_count,
                                                add_negatives=True,
                                                negative_sample_strategy=opt.negative_sample_strategy)

    val_dataloader: DataLoader = DataLoader(val_dataset, shuffle=True, batch_size=opt.batch_size,
                                            num_workers=opt.num_workers, pin_memory=True)

    # Model
    if opt.skip_connection_sep:
        separation_model: nn.Module = HourGlassMultiOutSkip(ResNetEncoderSkip(1), ResNetDecoderSkip(1))
    else:
        separation_model: nn.Module = HourGlassMultiOut(ResNetEncoder(1), ResNetDecoder(1))

    if opt.skip_connection_com:
        completion_model: nn.Module = HourGlassMultiOutSkip(ResNetEncoderSkip(1), ResNetDecoderSkip(1))
    else:
        completion_model: nn.Module = HourGlassMultiOut(ResNetEncoder(1), ResNetDecoder(1))  # multiout for classification

    # multihead_model: nn.Module = CatalogClassifier([256, 1, 1, 1], 8)  #classification (8 class)
    multihead_model: nn.Module = MultiHeadModule([256, 1, 1, 1], 8, train_symmetry=opt.train_symmetry, train_scale=opt.train_scale)

    if opt.offline_sample:
        embedding_model: nn.Module = TripletNet(ResNetEncoder(1))  # for offline assiging
    else:
        embedding_model: nn.Module = TripletNetBatchMix(ResNetEncoder(1))  # for online mining (half anchor scan + half positive cad)

    # Main loop
    iteration_number = 0
    resume_epoch = 0
    best_iter = 0
    best_epoch = 0
    best_score = 0
    resume_on = False

    # Load checkpoints
    if opt.resume is not None:
        resume_run_path = opt.resume.split("/")[0]
        checkpoint_name = opt.resume.split("/")[1]
        resume_run_path = os.path.join(opt.output_root, resume_run_path)
        if not os.path.exists(os.path.join(resume_run_path, f"{checkpoint_name}.pt")):
            checkpoint_name = "best"
        print("Resume from checkpoint " + checkpoint_name)
        loaded_model = torch.load(os.path.join(resume_run_path, f"{checkpoint_name}.pt"))
        separation_model.load_state_dict(loaded_model["separation"])
        completion_model.load_state_dict(loaded_model["completion"])
        multihead_model.load_state_dict(loaded_model["multihead"])
        embedding_model.load_state_dict(loaded_model["metriclearning"])
        iteration_number = loaded_model["iteration"]
        resume_epoch = loaded_model["epoch"]
        resume_on = True

    separation_model = separation_model.to(device)
    completion_model = completion_model.to(device)
    multihead_model = multihead_model.to(device)
    embedding_model = embedding_model.to(device)

    model_parameters = list(separation_model.parameters()) + \
                       list(completion_model.parameters()) + \
                       list(multihead_model.parameters()) + \
                       list(embedding_model.parameters())  # classification

    # Optimizer
    optimizer = optim.Adam(
        model_parameters, lr=opt.learning_rate, weight_decay=opt.weight_decay)

    if resume_on:
        optimizer.load_state_dict(loaded_model["optimizer"])

    if opt.wandb_vis_on:
        wandb.watch(separation_model, log="all")
        wandb.watch(completion_model, log="all")
        wandb.watch(multihead_model, log="all")
        wandb.watch(embedding_model, log="all")

    # Loss
    criterion_separation = nn.BCEWithLogitsLoss(reduction="none")
    criterion_completion = nn.BCEWithLogitsLoss(reduction="none")
    criterion_classification = nn.CrossEntropyLoss(reduction="none")
    criterion_symmetry = nn.CrossEntropyLoss(reduction='none')
    criterion_scale = nn.MSELoss(reduction='none')

    if opt.offline_sample:
        if opt.loss_retrieval == "tripletMargin":
            criterion_retrieval = nn.TripletMarginLoss(reduction="none", margin=opt.triplet_margin)
        else:
            criterion_retrieval = CircleLoss(m=opt.circle_margin, gamma=opt.circle_gamma)
    else:  # online_sample (mining) in a mini-batch
        if opt.loss_retrieval == "tripletMargin":
            criterion_retrieval = metric_losses.TripletMarginLoss(embedding_regularizer=LpRegularizer(),
                                                                  margin=opt.triplet_margin)
        else:
            criterion_retrieval = metric_losses.CircleLoss(embedding_regularizer=LpRegularizer(),
                                                           m=opt.circle_margin, gamma=opt.circle_gamma)

    # online triplet miner
    if opt.miner_type == "semihard":
        retrieval_miner = miners.TripletMarginMiner(margin=opt.triplet_margin, type_of_triplets="semihard")
    elif opt.miner_type == "easyhard":
        retrieval_miner = miners.BatchEasyHardMiner(pos_strategy=miners.BatchEasyHardMiner.EASY,
                                                    neg_strategy=miners.BatchEasyHardMiner.SEMIHARD,
                                                    allowed_pos_range=None,
                                                    allowed_neg_range=None)
    else:
        retrieval_miner = miners.MultiSimilarityMiner()

    weight_task = opt.weight_fgs, opt.weight_bgs, opt.weight_com, opt.weight_cls, opt.weight_sym, opt.weight_scale, opt.weight_retr

    # main loop (epoch)
    for epoch in range(opt.num_epochs):
        if epoch < resume_epoch:
            continue

        # for each epoch, we re-generate the negative samples randomly (dealing with overfitting of embedding loss)
        if opt.negative_sample_strategy == "other_cat":
            train_dataloader.dataset.regenerate_negatives()
        elif opt.negative_sample_strategy == "same_cat":
            train_dataloader.dataset.regenerate_negatives_same_category()
        else:  # mix
            train_dataloader.dataset.regenerate_negatives_mix()

        if opt.rotation_augmentation == "fixed" and (epoch - 1) % opt.canonical_learning_step == 0:  # for canonical learning
            rotations = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]  # rotation around y axis for a random angle within the list (0:30:330), 12 possibilities #TODO: use the parameter rotation_trial_count here
            degree = random.choice(rotations)
            train_dataloader.dataset.reset_rotation(degree)
            val_dataloader.dataset.reset_rotation(degree)

        if epoch < opt.retrieval_epoch:  # disabling embedding model
            utils.freeze_model(embedding_model)
        else:
            utils.unfreeze_model(embedding_model)

        if epoch > opt.finetune_epoch:  # disabling mtl models, only fine tune the embedding model
            utils.freeze_model(separation_model)
            utils.freeze_model(completion_model)
            utils.freeze_model(multihead_model)
        else:
            utils.unfreeze_model(separation_model)
            utils.unfreeze_model(completion_model)
            utils.unfreeze_model(multihead_model)

        # loop (mini-batch, iteration)
        for _, (scan, cad, negative, positive) in enumerate(train_dataloader):
            # for each mini-batch in epoch
            utils.stepwise_learning_rate_decay(optimizer, opt.learning_rate, iteration_number,
                                               [10000, 20000, 30000])  # learning_rate * 0.1 after such step

            separation_model.train()
            completion_model.train()
            multihead_model.train()  # classification
            embedding_model.train()

            visualize_3d = False
            if iteration_number % opt.log_frequency == opt.log_frequency - 1 and opt.wandb_vis_on:
                visualize_3d = True

            losses = forward(scan, cad, negative, positive,
                             separation_model, completion_model, multihead_model, embedding_model, device,
                             criterion_separation, criterion_completion, criterion_classification,
                             criterion_retrieval, opt.loss_retrieval, opt.offline_sample, retrieval_miner,
                             weight_task, opt.separation_model_on, opt.balanced_weight, False, "3d_train/",
                             train_symmetry=opt.train_symmetry, criterion_symmetry=criterion_symmetry,
                             train_scale=opt.train_scale, criterion_scale=criterion_scale,
                             scale_range=opt.train_scale_range)  # do not show 3d result on wandb for training (save storage space)

            loss_foreground, loss_completion, loss_classification, loss_symmetry, loss_scale, loss_retrieval, acc_classification, acc_symmetry, ratio_scale = losses
            loss_total = loss_foreground + loss_completion + loss_classification + loss_retrieval

            # Train step
            optimizer.zero_grad()  # Adam
            loss_total.backward()  # backpropagation
            optimizer.step()

            # Log to console
            if iteration_number % opt.log_frequency == opt.log_frequency - 1:
                print(f"[E{epoch:04d}, I{iteration_number:05d}]\tTotal: {loss_total: 05.3f}",
                      f"\tFG: {loss_foreground: 05.3f}",
                      f"\tCompletion: {loss_completion: 05.3f}",
                      f"\tClassification: {loss_classification: 05.3f}",
                      f"\tSymmetry: {loss_symmetry: 05.3f}" if opt.train_symmetry else "",
                      f"\tScale: {loss_scale: 05.3f}" if opt.train_scale else "",
                      f"\tACC_Classification: {acc_classification: 03.2f}",
                      f"\tTriplet: {loss_retrieval: 05.3f}")

                # Log to wandb
                if opt.wandb_vis_on:
                    wandb_log_content = {
                        'epoch': epoch, 'iteration': iteration_number, 'loss_train/total': loss_total,
                        'loss_train/fg_seg': loss_foreground,
                        'loss_train/comp': loss_completion,
                        'loss_train/classification': loss_classification,
                        'metric_train/acc_classification': acc_classification,
                        'loss_train/triplet': loss_retrieval,}
                    if opt.train_scale:
                        wandb_log_content['loss_train/scale_'] = loss_scale
                        # print(f"loss_scale: {loss_scale}, type={type(loss_scale)}")
                        wandb_log_content['metric_train/_ratio_scale_x'] = ratio_scale[0].item()
                        # print(f"ratio_scale[0].item(): {ratio_scale[0].item()}, type= {type(ratio_scale[0].item())}")
                        wandb_log_content['metric_train/_ratio_scale_y'] = ratio_scale[1].item()
                        wandb_log_content['metric_train/_ratio_scale_z'] = ratio_scale[2].item()
                    if opt.train_symmetry:
                        wandb_log_content['loss_train/symmetry'] = loss_symmetry
                        # print(f"loss_symmetry: {loss_symmetry}, type={type(loss_symmetry)}")
                        wandb_log_content['metric_train/acc_symmetry'] = acc_symmetry
                    wandb.log(wandb_log_content)

            # Validate
            if iteration_number % opt.validate_frequency == opt.validate_frequency - 1:

                # newly added (also generate random negative samples for the validation set)
                if opt.negative_sample_strategy == "other_cat":
                    val_dataloader.dataset.regenerate_negatives()
                elif opt.negative_sample_strategy == "same_cat":
                    val_dataloader.dataset.regenerate_negatives_same_category()
                else:  # mix
                    val_dataloader.dataset.regenerate_negatives_mix()

                with torch.no_grad():
                    separation_model.eval()
                    completion_model.eval()
                    multihead_model.eval()
                    embedding_model.eval()

                    val_losses = defaultdict(list)

                    # Go through entire validation set 
                    for _, (scan_v, cad_v, negative_v, positive_v) in tqdm(enumerate(val_dataloader),
                                                                           total=len(val_dataloader), leave=False):
                        losses = forward(scan_v, cad_v, negative_v, positive_v,
                                         separation_model, completion_model, multihead_model, embedding_model, device,
                                         criterion_separation, criterion_completion, criterion_classification,
                                         criterion_retrieval, opt.loss_retrieval, opt.offline_sample, retrieval_miner,
                                         weight_task, opt.separation_model_on, opt.balanced_weight, opt.wandb_vis_on,
                                         "3d_val/",
                                         train_symmetry=opt.train_symmetry, criterion_symmetry=criterion_symmetry,
                                         train_scale=opt.train_scale, criterion_scale=criterion_scale,
                                         scale_range=opt.train_scale_range
                                         )

                        loss_foreground, loss_completion, loss_classification, loss_symmetry, loss_scale, loss_retrieval, acc_classification, acc_symmetry, ratio_scale = losses
                        loss_total = loss_foreground + loss_completion + loss_classification + loss_retrieval
                        if opt.train_symmetry:
                            loss_total += loss_symmetry
                        if opt.train_scale:
                            loss_total += loss_scale
                        val_losses["FG"].append(loss_foreground.item())
                        val_losses["Completion"].append(loss_completion.item())
                        val_losses["Classification"].append(loss_classification.item())  # classification
                        val_losses["Acc_Classification"].append(acc_classification)
                        if opt.train_symmetry:
                            val_losses["Symmetry"].append(loss_symmetry.item())
                            val_losses["Acc_Symmetry"].append(acc_symmetry)
                        if opt.train_scale:
                            val_losses["Scale"].append(loss_scale)
                            val_losses["Ratio_Scale_x"].append(ratio_scale[0])
                            val_losses["Ratio_Scale_y"].append(ratio_scale[1])
                            val_losses["Ratio_Scale_z"].append(ratio_scale[2])

                        val_losses["Triplet"].append(loss_retrieval.item())
                        val_losses["Total"].append(loss_total.item())

                    # Aggregate losses
                    val_losses_summary = {k: torch.mean(
                        torch.tensor(v)) for k, v in val_losses.items()}
                    print(f"-V [E{epoch:04d}, I{iteration_number:05d}]\tTotal: {val_losses_summary['Total']:05.3f}",
                          f"\evaluate_retrieval_metricstFG: {val_losses_summary['FG']:05.3f} \t",
                          f"\tCompletion: {val_losses_summary['Completion']:05.3f}",
                          f"\tClassification: {val_losses_summary['Classification']:05.3f}",  # classification loss
                          f"\tAcc_Classification: {val_losses_summary['Acc_Classification']:05.3f}",
                          # classification accuracy
                          f"\tSymmetry: {val_losses_summary['Symmetry']}" if opt.train_symmetry else "",
                          f"\tAcc_Symmetry: {val_losses_summary['Acc_Symmetry']}" if opt.train_symmetry else "",
                          f"\tTriplet: {val_losses_summary['Triplet']:05.3f}")

                    if opt.wandb_vis_on:
                        wandb_log_content = {
                            'epoch': epoch, 'iteration': iteration_number,
                            'loss_val/total': val_losses_summary['Total'],
                            'loss_val/fg_seg': val_losses_summary['FG'],
                            'loss_val/comp': val_losses_summary['Completion'],
                            'loss_val/classification': val_losses_summary['Classification'],
                            'metric_val/acc_classification': val_losses_summary['Acc_Classification'],
                            'loss_val/triplet': val_losses_summary['Triplet'],
                            'misc/learning_rate': optimizer.param_groups[0]['lr']}
                        if opt.train_symmetry:
                            wandb_log_content['loss_val/symmetry'] = val_losses_summary['Symmetry']
                            wandb_log_content['metric_val/acc_symmetry'] = val_losses_summary['Acc_Symmetry']
                        if opt.train_scale:
                            wandb_log_content['loss_val/scale_'] = val_losses_summary['Scale']
                            wandb_log_content['metric_val/_ratio_scale_x'] = val_losses_summary['Ratio_Scale_x']
                            wandb_log_content['metric_val/_ratio_scale_y'] = val_losses_summary['Ratio_Scale_y']
                            wandb_log_content['metric_val/_ratio_scale_z'] = val_losses_summary['Ratio_Scale_z']

                        wandb.log(wandb_log_content)

                    # Evaluation: compute retrieval metrics
                    if iteration_number % opt.evaluation_frequency == opt.evaluation_frequency - 1:
                        cat_retrieval_accuracy, model_retrieval_accuracy, ranking_accuracy = evaluate_retrieval_metrics(
                            separation_model, completion_model, multihead_model, embedding_model, device,
                            opt.similarity_file, scan_base_path, opt.shapenet_voxel_path, opt.scan2cad_quat_file,
                            opt.scan_dataset_name, opt.separation_model_on, opt.batch_size, trans,
                            opt.rotation_trial_count, opt.filter_val_pool, opt.val_max_sample_count,
                            wb_visualize_on=opt.wandb_vis_on, vis_sample_count=opt.val_vis_sample_count)

                        if opt.wandb_vis_on:
                            wandb.log({'epoch': epoch, 'iteration': iteration_number,
                                       'metric_val/category_accuracy': cat_retrieval_accuracy,
                                       'metric_val/model_accuracy': model_retrieval_accuracy,
                                       'metric_val/ranking_accuracy': ranking_accuracy})

                        if model_retrieval_accuracy > best_score:
                            best_score = model_retrieval_accuracy
                            best_iter = iteration_number
                            best_epoch = epoch
                            print("Best model updated", f"[E{epoch:04d}]")
                            # Save the best model up-to-now
                            checkpoint_name = "best"
                            utils.batch_save_checkpoint(separation_model, completion_model, multihead_model,
                                                        embedding_model,
                                                        optimizer, run_path, checkpoint_name, iteration_number, epoch)
                            print(f"Saved model at {run_path}/{checkpoint_name}")

                            # Regularly Save checkpoint
            if iteration_number % opt.checkpoint_frequency == opt.checkpoint_frequency - 1:
                checkpoint_name = f"{iteration_number + 1:05d}"
                utils.batch_save_checkpoint(separation_model, completion_model, multihead_model, embedding_model,
                                            optimizer, run_path, checkpoint_name, iteration_number, epoch)
                print(f"Saved model at {run_path}/{checkpoint_name}")

            iteration_number += 1


if __name__ == '__main__':
    args = utils.command_line_parser()
    main(args)

'''
N.B. about the configurations

0. currently, our dataset has about 7000 training samples (and 1500+ validation / 1500+ testing samples)
such data number is not enough, you need to apply some data agumentation

1. batch_size=128 --> GPU memory  6500MiB / 11000MiB, so it's better to just set it at 128
under such case, each epoch has about 55 iterations

'''
'''
TODO List and notifications:
1. add wandb support for better monitoring of the training process (completed) You may check the dashboard of yuepan here: https://wandb.ai/yuepan/ScanCADJoint
2. test hyperparameters (completed)
3. test data agumentation (completed)
4. try to use df/sdf instead of binary voxel for more detailed information of the shape (completed)
5. validation with rotation trials (completed)
6. metric learning, use circle loss instead of triplet loss, use a better model pool json (completed, Yue) # an example: https://github.com/overlappredator/OverlapPredator/blob/main/lib/loss.py
7. consider to do classification and model retrieval together (completed, Ke)
8. try different weight strategy among different loss (completed, He)
9. add skip connection, or your reconstruction would not be fine-grained (completed, Pan)
10. try TSNE for visualizing the joint embedding feature space, and add it to wandb (completed, Pan and Yue) # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
11. apply the offline embedding (completed, Pan)
12. add the coarse-to-fine registration module (completed, Pan)
13. add the regitration result visualization module  (completed, Pan)
'''
