'''
By Real2CAD group at ETH Zurich
3DV Group 14
Yue Pan, Yuanwen Yue, Bingxin Ke, Yujie He

Editted based on the codes of the JointEmbedding paper (https://github.com/xheon/JointEmbedding)

As for our contributions, please check our report
'''

import argparse
import json
from typing import List, Tuple, Dict
import os
from datetime import datetime
import random
import math
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sklearn.manifold import TSNE

import wandb
import open3d as o3d

import data
import metrics
import utils
from models import *

from typing import List, Tuple, Any


def main(opt: argparse.Namespace):
    # Configure environment
    utils.set_gpu(opt.gpu)
    device = torch.device("cuda")
    
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # begining timestamp
    run_name = opt.name + "_" + ts  # modified to a name that is easier to index
    run_path = os.path.join(opt.output_root, run_name)
    if not os.path.exists(run_path):
        os.mkdir(run_path)
    assert os.access(run_path, os.W_OK)
    print(f"Start testing {run_path}")
    print(vars(opt))

    # Set wandb
    visualize_on = False
    if opt.wandb_vis_on:
        utils.setup_wandb()
        wandb.init(project="ScanCADJoint", entity="real2cad", config=vars(opt), dir=run_path) # team 'real2cad'
        #wandb.init(project="ScanCADJoint", config=vars(opt), dir=run_path) # your own worksapce
        wandb.run.name = run_name
        visualize_on = True

    # Model
    if opt.skip_connection_sep:
        separation_model: nn.Module = HourGlassMultiOutSkip(ResNetEncoderSkip(1), ResNetDecoderSkip(1))
    else:
        separation_model: nn.Module = HourGlassMultiOut(ResNetEncoder(1), ResNetDecoder(1))            
                                                                                                                
    if opt.skip_connection_com:
        completion_model: nn.Module = HourGlassMultiOutSkip(ResNetEncoderSkip(1), ResNetDecoderSkip(1))                                                
    else:
        completion_model: nn.Module = HourGlassMultiOut(ResNetEncoder(1),ResNetDecoder(1))  #multiout for classification

    classification_model: nn.Module = CatalogClassifier([256, 1, 1, 1], 8)  #classification (8 class)

    if opt.offline_sample:
        embedding_model: nn.Module = TripletNet(ResNetEncoder(1)) #for offline assiging
    else:
        embedding_model: nn.Module = TripletNetBatchMix(ResNetEncoder(1)) #for online mining (half anchor scan + half positive cad)
    
    if opt.representation == "tdf":
        trans = data.truncation_normalization_transform
    else:  # binary_occupancy
        trans = data.to_occupancy_grid

    # Load checkpoints
    resume_run_path = opt.resume.split("/")[0]
    checkpoint_name = opt.resume.split("/")[1] 
    resume_run_path = os.path.join(opt.output_root, resume_run_path)   
    if not os.path.exists(os.path.join(resume_run_path, f"{checkpoint_name}.pt")):
        checkpoint_name = "best"
    print("Resume from checkpoint " + checkpoint_name)
    loaded_model = torch.load(os.path.join(resume_run_path, f"{checkpoint_name}.pt"))
    
    if opt.separation_model_on:
        separation_model.load_state_dict(loaded_model["separation"])
        separation_model = separation_model.to(device)
        separation_model.eval()
    else:
        separation_model = None

    completion_model.load_state_dict(loaded_model["completion"])
    classification_model.load_state_dict(loaded_model["classification"])
    embedding_model.load_state_dict(loaded_model["metriclearning"])

    completion_model = completion_model.to(device)
    classification_model = classification_model.to(device)
    embedding_model = embedding_model.to(device)

    # Make sure models are in evaluation mode   
    completion_model.eval()
    classification_model.eval()
    embedding_model.eval()

    print("Begin evaluation")

    if opt.scenelist_file is None:
        # test_scan_list = ["scene0000_00", "scene0001_00", "scene0002_00", "scene0003_00", "scene0004_00", "scene0005_00",
        #  "scene0006_00", "scene0007_00", "scene0008_00", "scene0009_00", "scene0010_00", "scene0011_00", "scene0012_00", "scene0013_00", "scene0014_00", "scene0015_00"]
        test_scan_list =["scene0030_00", "scene0031_00", "scene0032_00", "scene0033_00"]
    
    else:
        with open(opt.scenelist_file) as f:
            test_scan_list = json.load(f)

    scan_base_path = None
    if opt.scan_dataset_name == "scannet":
        scan_base_path = opt.scannet_path
    elif opt.scan_dataset_name == "2d3ds":
        scan_base_path = opt.s2d3ds_path
    
    # Compute similarity metrics
    # TODO: Update scan2cad_quat_file for 2d3ds dataset
    # retrieval_metrics = evaluate_retrieval_metrics(separation_model, completion_model, classification_model, embedding_model, device,
    #                                                 opt.similarity_file, scan_base_path, opt.shapenet_voxel_path, opt.scan2cad_quat_file,
    #                                                 opt.scan_dataset_name, opt.separation_model_on, opt.batch_size, trans,
    #                                                 opt.rotation_trial_count, opt.filter_val_pool, opt.val_max_sample_count, 
    #                                                 wb_visualize_on = visualize_on, vis_sample_count = opt.val_vis_sample_count) 

    # Compute cad embeddings
    if opt.embed_mode:
        embed_cad_pool(embedding_model, device, opt.modelpool_file, opt.shapenet_voxel_path, opt.cad_embedding_path, 
                       opt.batch_size, trans, opt.rotation_trial_count)
        
        # embed_scan_objs(separation_model, completion_model, classification_model, embedding_model, device,
        #                 opt.scan2cad_file, scan_base_path, opt.scan_embedding_path, opt.scan_dataset_name, 
        #                 opt.separation_model_on, opt.batch_size, trans)
              

    # accomplish Real2CAD task # TODO: add evaluation based on CD
    retrieve_in_scans(separation_model, completion_model, classification_model, embedding_model, device, test_scan_list,
                      opt.cad_embedding_path, opt.cad_apperance_file, scan_base_path, opt.shapenet_voxel_path, opt.shapenet_pc_path, opt.real2cad_result_path,
                      opt.scan_dataset_name, opt.separation_model_on, opt.batch_size, trans, 
                      opt.rotation_trial_count, opt.filter_val_pool, opt.in_the_wild_mode, opt.init_scale_method,  
                      opt.icp_reg_mode, opt.icp_dist_thre, opt.icp_with_scale_on, opt.only_rot_z, opt.only_coarse_reg, visualize_on)
                                                     
    # print(retrieval_metrics)


    # TSNE plot
    # embedding_tsne(opt.cad_embedding_path, opt.scan_embedding_path, opt.tsne_img_path, True, visualize_on)

    # Compute domain confusion # TODO update (use together with TSNE)
    # train_confusion_results = evaluate_confusion(separation_model, completion_model, embedding_model,
    #                                              device, opt.confusion_train_path, opt.scannet_path, opt.shapenet_path,
    #                                              opt.confusion_num_neighbors, "train")
    # print(train_confusion_results) #confusion_mean, conditional_confusions_mean
    #
    # val_confusion_results = evaluate_confusion(separation_model, completion_model, embedding_model,
    #                                            device, opt.confusion_val_path, opt.scannet_path, opt.shapenet_path,
    #                                            opt.confusion_num_neighbors, "validation")
    # print(val_confusion_results) #confusion_mean, conditional_confusions_mean

    pass   

def evaluate_confusion(separation: nn.Module, completion: nn.Module, triplet: nn.Module, device, dataset_path: str,
                       scannet_path: str, shapenet_path: str, num_neighbors: int, data_split: str, batch_size: int = 1,
                       trans=data.to_occupancy_grid, verbose: bool = False) -> Tuple[np.array, list]:
    # Configure datasets
    dataset: Dataset = data.TrainingDataset(dataset_path, scannet_path, shapenet_path, "all", [data_split], scan_rep="sdf",
                                     transformation=trans)
    dataloader: DataLoader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=0)

    embeddings: List[torch.Tensor] = []  # contains all embedding vectors
    names: List[str] = []  # contains the names of the samples
    category: List[int] = [] # contains the category label of the samples
    domains: List[int] = []  # contains number labels for domains (scan=0/cad=1)

    # Iterate over data
    for scan, cad in tqdm(dataloader, total=len(dataloader)):
        # Move data to GPU
        scan_data = scan["content"].to(device)
        cad_data = cad["content"].to(device)

        with torch.no_grad():
            # Pass scan through networks
            scan_foreground, _ = separation(scan_data)
            scan_completed, _ = completion(torch.sigmoid(scan_foreground))
            scan_latent = triplet.embed(torch.sigmoid(scan_completed)).view(batch_size, -1)
            embeddings.append(scan_latent)
            names.append(f"/scan/{scan['name']}")
            domains.append(0) # scan

            # Embed cad
            cad_latent = triplet.embed(cad_data).view(batch_size, -1)
            embeddings.append(cad_latent)
            names.append(f"/cad/{cad['name']}")
            domains.append(1)  # cad

    embedding_space = torch.cat(embeddings, dim=0)  # problem
    embedding_space = embedding_space.cpu().numpy()

    domain_labels: np.array = np.asarray(domains)
    cat_labels: np.array = np.asarray(category)

    # Compute distances between all samples
    distance_matrix = metrics.compute_distance_matrix(embedding_space)
    confusion, conditional_confusions = metrics.compute_knn_confusions(distance_matrix, domain_labels, num_neighbors)
    confusion_mean = np.average(confusion)
    conditional_confusions_mean = [np.average(conf) for conf in conditional_confusions]

    return confusion_mean, conditional_confusions_mean


def evaluate_retrieval_metrics(separation_model: nn.Module, completion_model: nn.Module, classification_model: nn.Module,
                                embedding_model: nn.Module, device, similarity_dataset_path: str,
                                scan_base_path: str, cad_base_path: str, gt_quat_file: str = None, 
                                scan_dataset_name: str = "scannet", separation_model_on: bool = False, batch_size: int = 1,
                                trans=data.to_occupancy_grid, rotation_count: int = 1, filter_pool: bool = False, test_sample_limit: int = 999999,
                                wb_visualize_on = True, vis_name: str = "eval/", vis_sample_count: int = 5, verbose: bool = True):
                                
    # interested_categories = ["02747177", "02808440", "02818832", "02871439", "02933112", "03001627", "03211117",
    #                        "03337140", "04256520", "04379243"]

    unique_scan_objects, unique_cad_objects, sample_idx = get_unique_samples(similarity_dataset_path, rotation_count, test_sample_limit)

    
    if separation_model_on:
        scan_input_format = ".sdf"
        scan_input_folder_extension = "_object_voxel"
        scan_pc_folder_extension = "_object_pc"
        input_only_mask = False
    else:
        scan_input_format = ".mask"
        scan_input_folder_extension = "_mask_voxel"
        scan_pc_folder_extension = "_mask_pc"
        input_only_mask = True


    scan_dataset: Dataset = data.InferenceDataset(scan_base_path, unique_scan_objects, scan_input_format, "scan",
                                                 transformation=trans, scan_dataset = scan_dataset_name, input_only_mask=input_only_mask)  

    scan_dataloader = torch.utils.data.DataLoader(dataset=scan_dataset, shuffle=True, batch_size=batch_size)

    rotation_ranking_on = False
    if rotation_count > 1:
        rotation_ranking_on = True
    
    # eval mode
    # separation_model.eval()
    # completion_model.eval()
    # classification_model.eval()  
    # embedding_model.eval()

    record_test_cloud = True
    # vis_sep_com_count = vis_sample_count

    # load all the scan object and cads that are waiting for testing
    # # Evaluate all unique scan segments' embeddings
    embeddings: Dict[str, np.array] = {}
    mid_pred_cats: Dict[str, str] = {}
    for names, elements in tqdm(scan_dataloader, total=len(scan_dataloader)):
        # Move data to GPU
        elements = elements.to(device)
        with torch.no_grad():
            if separation_model_on:
                scan_foreground, _ = separation_model(elements)
                scan_foreground = torch.sigmoid(scan_foreground)
                scan_completed, hidden = completion_model(scan_foreground) 
            else:
                scan_completed, hidden = completion_model(elements) 
            mid_pred_cat = classification_model.predict_name(torch.sigmoid(hidden))  # class str
            scan_completed = torch.sigmoid(scan_completed)
            # scan_completed = torch.where(scan_completed > 0.5, scan_completed, torch.zeros(scan_completed.shape))
            scan_latent = embedding_model.embed(scan_completed)

        for idx, name in enumerate(names):
            embeddings[name] = scan_latent[idx].cpu().numpy().squeeze()
            mid_pred_cats[name] = mid_pred_cat[idx]
 
        #embeddings[names[0]] = scan_latent.cpu().numpy().squeeze() # why [0] ?  now works only for batch_size = 1
        #mid_pred_cats[name[0]] = mid_pred_cat
        if wb_visualize_on:   # TODO, may have bug
            scan_voxel = elements[0].cpu().detach().numpy().reshape((32, 32, 32))
            scan_cloud = data.voxel2point(scan_voxel)
            wb_vis_dict = {vis_name + "input scan object": wandb.Object3D(scan_cloud)}
            if separation_model_on:
                foreground_voxel = scan_foreground[0].cpu().detach().numpy().reshape((32, 32, 32))
                foreground_cloud = data.voxel2point(foreground_voxel, color_mode='prob')
                wb_vis_dict[vis_name + "point_cloud_foreground"] = wandb.Object3D(foreground_cloud)
            completed_voxel = scan_completed[0].cpu().detach().numpy().reshape((32, 32, 32))
            completed_cloud = data.voxel2point(completed_voxel, color_mode='prob', visualize_prob_threshold = 0.75)
            wb_vis_dict[vis_name + "point_cloud_completed"] = wandb.Object3D(completed_cloud)
            wandb.log(wb_vis_dict)

    # Evaluate all unique cad embeddings
    # update unique_cad_objects
    cad_dataset: Dataset = data.InferenceDataset(cad_base_path, unique_cad_objects, ".df", "cad",
                                                transformation=trans)
    cad_dataloader = torch.utils.data.DataLoader(dataset=cad_dataset, shuffle=False, batch_size=batch_size)

    for names, elements in tqdm(cad_dataloader, total=len(cad_dataloader)):
        # Move data to GPU
        elements = elements.to(device)
        with torch.no_grad():
            #cad_latent = embedding_model.embed(elements).view(-1)
            cad_latent = embedding_model.embed(elements)
        for idx, name in enumerate(names):
            embeddings[name] = cad_latent[idx].cpu().numpy().squeeze()
        #embeddings[name[0]] = cad_latent.cpu().numpy().squeeze()

    # Load GT alignment quat file

    with open(gt_quat_file) as qf: # rotation quaternion of the cad models relative to the scan object
        quat_content = json.load(qf)
        json_quat = quat_content["scan2cad_objects"]
                
    # Evaluate metrics
    with open(similarity_dataset_path) as f:
        samples = json.load(f).get("samples")
        test_sample_limit = min(len(samples), test_sample_limit)
        samples = list(samples[i] for i in sample_idx)

        retrieved_correct = 0
        retrieved_total = 0
        retrieved_cat_correct = 0
        retrieved_cat_total = 0
        ranked_correct = 0
        ranked_total = 0

        # Top 7 categories and the others
        selected_categories = ["03001627", "04379243", "02747177", "02818832", "02871439", "02933112", "04256520", "other"]
        category_name_dict = {"03001627": "Chair", "04379243": "Table","02747177": "Trash bin","02818832": "Bed", "02871439":"Bookshelf", "02933112":"Cabinet","04256520":"Sofa","other":"Other"}
        category_idx_dict = {"03001627": 0, "04379243": 1, "02747177": 2, "02818832": 3, "02871439": 4, "02933112": 5, "04256520": 6, "other": 7}

        per_category_retrieved_correct = {category: 0 for category in selected_categories}
        per_category_retrieved_total = {category: 0 for category in selected_categories}
        per_category_ranked_correct = {category: 0 for category in selected_categories}
        per_category_ranked_total = {category: 0 for category in selected_categories}

        idx = 0
        visualize = False
        vis_sample = []
        if vis_sample_count > 0:
            visualize = True
            vis_sample = random.sample(samples, vis_sample_count)
        
        # Iterate over all annotations
        for sample in tqdm(samples, total=len(samples)):
            reference_name = sample["reference"]["name"].replace("/scan/", "")

            reference_quat = json_quat.get(reference_name, [1.0, 0.0, 0.0, 0.0])

            reference_embedding = embeddings[reference_name][np.newaxis, :]
            #reference_embedding = embeddings[reference_name]

            # only search nearest neighbor in the pool  (the "ranked" list should be a subset of the "pool")
            pool_names = [p["name"].replace("/cad/", "") for p in sample["pool"]]

            # Filter pool with classification result
            if filter_pool:
                mid_pred_cat = mid_pred_cats[reference_name] # class str
                if mid_pred_cat != 'other': 
                    temp_pool_names = list(filter(lambda x: x.split('/')[0] == mid_pred_cat, pool_names))
                else: # deal with other categories
                    temp_pool_names = list(filter(lambda x: x.split('/')[0] not in selected_categories, pool_names))
                if len(temp_pool_names) != 0:
                    pool_names = temp_pool_names
                    #print("filter pool on")

            pool_names = np.asarray(pool_names)

            pool_names_all = []
            if rotation_ranking_on:
                pool_embeddings = []
                deg_step = np.around(360.0 / rotation_count)
                for p in pool_names:
                    for i in range(rotation_count):
                        cur_rot = int(i * deg_step)
                        cur_rot_p = p + "_" + str(cur_rot)
                        pool_names_all.append(cur_rot_p)
                        pool_embeddings.append(embeddings[cur_rot_p])
                pool_names_all = np.asarray(pool_names_all)
            else:
                pool_embeddings = [embeddings[p] for p in pool_names]
                pool_names_all = pool_names

            pool_embeddings = np.asarray(pool_embeddings)

            # Compute distances in embedding space
            distances = scipy.spatial.distance.cdist(reference_embedding, pool_embeddings, metric="euclidean")
            sorted_indices = np.argsort(distances, axis=1)
            sorted_distances = np.take_along_axis(distances, sorted_indices, axis=1) # [1, filtered_pool_size * rotation_trial_count]
            sorted_distances = sorted_distances[0] # [filtered_pool_size * rotation_trial_count]

            predicted_ranking = np.take(pool_names_all, sorted_indices)[0].tolist()

            ground_truth_names = [r["name"].replace("/cad/", "") for r in sample["ranked"]]
            ground_truth_cat = ground_truth_names[0].split("/")[0]

            # ground_truth_cat = reference_name.split("_")[4] #only works for scannet (scan2cad)

            predicted_cat = predicted_ranking[0].split("/")[0]
            
            # retrieval accuracy (top 1 [nearest neighbor] model is in the ranking list [1-3]) 
            sample_retrieved_correct = 1 if metrics.is_correctly_retrieved(predicted_ranking, ground_truth_names) else 0
            retrieved_correct += sample_retrieved_correct
            retrieved_total += 1

            # the top 1's category is correct [specific category str belongs to 'other' would also be compared]
            sample_cat_correct = 1 if metrics.is_category_correctly_retrieved(predicted_cat, ground_truth_cat) else 0
            #sample_cat_correct = 1 if metrics.is_category_correctly_retrieved(mid_pred_cat, ground_truth_cat) else 0
            retrieved_cat_correct += sample_cat_correct
            retrieved_cat_total += 1

            # per-category retrieval accuracy
            reference_category = metrics.get_category_from_list(ground_truth_cat, selected_categories)
            per_category_retrieved_correct[reference_category] += sample_retrieved_correct
            per_category_retrieved_total[reference_category] += 1

            # ranking quality
            sample_ranked_correct = metrics.count_correctly_ranked_predictions(predicted_ranking, ground_truth_names)
            ranked_correct += sample_ranked_correct
            ranked_total += len(ground_truth_names)
            per_category_ranked_correct[reference_category] += sample_ranked_correct
            per_category_ranked_total[reference_category] += len(ground_truth_names)

            if wb_visualize_on and visualize and sample in vis_sample:
                idx = idx + 1
                # raw scan segment
                parts = reference_name.split("_")
                if scan_dataset_name == 'scannet': 
                    scan_name = parts[0] + "_" + parts[1]
                    object_name = scan_name + "__" + parts[3]
                    scan_cat_name = utils.get_category_name(parts[4])
                    scan_cat = utils.wandb_color_lut(parts[4])
                    scan_path = os.path.join(scan_base_path, scan_name, scan_name + scan_input_folder_extension, object_name + scan_input_format)
                elif scan_dataset_name == '2d3ds': 
                    area_name = parts[0]+"_"+parts[1]
                    room_name = parts[2]+"_"+parts[3]
                    scan_cat_name = parts[4]
                    scan_cat = utils.wandb_color_lut(utils.get_category_code_from_2d3ds(scan_cat_name))
                    scan_path = os.path.join(scan_base_path, area_name, room_name, room_name + scan_input_folder_extension, reference_name + scan_input_format)
            
                if separation_model_on:
                    scan_voxel_raw = data.load_raw_df(scan_path)
                    scan_voxel = data.to_occupancy_grid(scan_voxel_raw).tdf.reshape((32, 32, 32))
                else:
                    scan_voxel_raw = data.load_mask(scan_path)
                    scan_voxel = scan_voxel_raw.tdf.reshape((32, 32, 32))
            
                scan_grid2world = scan_voxel_raw.matrix
                scan_voxel_res = scan_voxel_raw.size
                #print("Scan voxel shape:", scan_voxel.shape)
                
                scan_wb_obj = data.voxel2point(scan_voxel, scan_cat, name=scan_cat_name + ":" + object_name)
                wb_vis_retrieval_dict = {vis_name+"input scan object " + str(idx):  wandb.Object3D(scan_wb_obj)}

                # ground truth cad to scan rotation
                Rsc = data.get_rot_cad2scan(reference_quat)

                # ground truth cad
                if scan_dataset_name == 'scannet': 
                    gt_cad = os.path.join(parts[4], parts[5])  
                elif scan_dataset_name == '2d3ds':
                    gt_cad = ground_truth_names[0]
                gt_cad_path = os.path.join(cad_base_path, gt_cad+ ".df")
                gt_cad_voxel = data.to_occupancy_grid(data.load_raw_df(gt_cad_path)).tdf
                gt_cad_voxel_rot = data.rotation_augmentation_interpolation_v3(gt_cad_voxel, "cad", aug_rotation_z = 0, pre_rot_mat = Rsc).reshape((32, 32, 32))
                gt_cad_wb_obj = data.voxel2point(gt_cad_voxel_rot, scan_cat, name=scan_cat_name + ":" + gt_cad)
                wb_vis_retrieval_dict[vis_name+"ground truth cad " + str(idx)] = wandb.Object3D(gt_cad_wb_obj)

                # Top K choices
                top_k = 4
                for top_i in range(top_k):
                    predicted_cad = predicted_ranking[top_i]
                    predicted_cad_path = os.path.join(cad_base_path, predicted_cad.split("_")[0] + ".df")
                    predicted_cad_voxel = data.to_occupancy_grid(data.load_raw_df(predicted_cad_path)).tdf
                    predicted_rotation = int(predicted_cad.split("_")[1]) # deg
                    predicted_cad_voxel_rot = data.rotation_augmentation_interpolation_v3(predicted_cad_voxel, "dummy", aug_rotation_z = predicted_rotation).reshape((32, 32, 32))
                    predicted_cat = utils.wandb_color_lut(predicted_cad.split("/")[0])
                    predicted_cat_name = utils.get_category_name(predicted_cad.split("/")[0])
                    predicted_cad_wb_obj = data.voxel2point(predicted_cad_voxel_rot, predicted_cat, name=predicted_cat_name + ":" + predicted_cad)
                    wb_title = vis_name + "Top " + str(top_i+1) + " retrieved cad with rotation " + str(idx)
                    wb_vis_retrieval_dict[wb_title] = wandb.Object3D(predicted_cad_wb_obj)

                wandb.log(wb_vis_retrieval_dict)
                
                
        print("retrieval accuracy")
        cat_retrieval_accuracy = retrieved_cat_correct / retrieved_cat_total
        print(f"correct: {retrieved_cat_correct}, total: {retrieved_cat_total}, category level (rough) accuracy: {cat_retrieval_accuracy:4.3f}")

        cad_retrieval_accuracy = retrieved_correct/retrieved_total
        print(f"correct: {retrieved_correct}, total: {retrieved_total}, cad model level (fine-grained) accuracy: {cad_retrieval_accuracy:4.3f}")

        if verbose:
            for (category, correct), total in zip(per_category_retrieved_correct.items(),
                                                  per_category_retrieved_total.values()):
                category_name = utils.get_category_name(category)
                if total == 0:
                    print(
                        f"{category}:[{category_name}] {correct:>5d}/{total:>5d} --> Nan")
                else:
                    print(
                        f"{category}:[{category_name}] {correct:>5d}/{total:>5d} --> {correct / total:4.3f}")
        
        ranking_accuracy = ranked_correct / ranked_total
        # print("ranking quality")
        
        # print(f"correct: {ranked_correct}, total: {ranked_total}, ranking accuracy: {ranking_accuracy:4.3f}")

        # if verbose:
        #     for (category, correct), total in zip(per_category_ranked_correct.items(),
        #                                           per_category_ranked_total.values()):
        #         category_name = utils.get_category_name(category)
        #         if 0 == total:
        #             print(
        #                 f"{category}:[{category_name}] {correct:>5d}/{total:>5d} --> Nan")
        #         else:
        #             print(
        #                 f"{category}:[{category_name}] {correct:>5d}/{total:>5d} --> {correct / total:4.3f}")
    
    return cat_retrieval_accuracy, cad_retrieval_accuracy, ranking_accuracy


def embed_scan_objs(separation_model: nn.Module, completion_model: nn.Module, classification_model: nn.Module,
                    embedding_model: nn.Module, device, scan_obj_list_path: str, scan_base_path: str, output_path: str, 
                    scan_dataset_name: str = "scannet", separation_model_on: bool = False,
                    batch_size: int = 1, trans=data.to_occupancy_grid, output: bool = True):
    
    #load scan list
    with open(scan_obj_list_path) as f:
        scenes = json.load(f)["scan2cad_objects"]
        unique_scan_objects = list(set(scenes.keys()))
    
    scan_seg_count = len(unique_scan_objects)

    if separation_model_on:
        scan_input_format = ".sdf"
        scan_input_folder_extension = "_object_voxel"
        scan_pc_folder_extension = "_object_pc"
        input_only_mask = False
    else:
        scan_input_format = ".mask"
        scan_input_folder_extension = "_mask_voxel"
        scan_pc_folder_extension = "_mask_pc"
        input_only_mask = True

    scan_dataset: Dataset = data.InferenceDataset(scan_base_path, unique_scan_objects, scan_input_format, "scan",
                                                  transformation=trans, scan_dataset = scan_dataset_name, input_only_mask=input_only_mask)  
    
    scan_dataloader = torch.utils.data.DataLoader(dataset=scan_dataset, shuffle=False, batch_size=batch_size)

    # # Evaluate all unique scan segments' embeddings
    embeddings_all: Dict[str, Dict] = {}

    for names, elements in tqdm(scan_dataloader, total=len(scan_dataloader)):
        # Move data to GPU
        elements = elements.to(device)
        with torch.no_grad():
            if separation_model_on:
                scan_foreground, _ = separation_model(elements)
                scan_foreground = torch.sigmoid(scan_foreground)
                scan_completed, _ = completion_model(scan_foreground) 
            else:
                scan_completed, _ = completion_model(elements) 
            scan_completed = torch.sigmoid(scan_completed)
            scan_latent = embedding_model.embed(scan_completed)

        for idx, name in enumerate(names):
            cur_scan_embedding = scan_latent[idx].cpu().numpy().squeeze()
            if scan_dataset_name == "scannet":
                cat = name.split("_")[4]
            elif scan_dataset_name == "2d3ds":
                cat = utils.get_category_code_from_2d3ds(name.split("_")[4])
            if cat not in embeddings_all.keys():
                embeddings_all[cat] = {name: cur_scan_embedding}
            else:
                embeddings_all[cat][name] = cur_scan_embedding
    
    print("Embed [", scan_seg_count, "] scan segments")

    if output:
        torch.save(embeddings_all, output_path)
        print("Output scan segement embeddings to [", output_path, "]")

# TODO better to have a different data input for scan2cad_file
def embed_cad_pool(embedding_model: nn.Module, device, modelpool_path: str, shapenet_path: str, output_path: str, 
                   batch_size: int = 1, trans=data.to_occupancy_grid, rotation_count: int = 1, output: bool = True):
    
    #load model pool
    with open(modelpool_path) as f:
        model_pool = json.load(f)
        # delete duplicate elements from each list
        for cat, cat_list in model_pool.items():
            cat_list_filtered = list(set(cat_list))
            model_pool[cat] = cat_list_filtered

    rotation_ranking_on = False
    if rotation_count > 1:
        rotation_ranking_on = True

    # get unique cad names (with rotation)
    unique_cads = []
    categories = list(model_pool.keys())
    for category in categories:
        cat_pool = model_pool[category]
        for cad_element in cat_pool:
            cad = cad_element.replace("/cad/", "")
            if rotation_ranking_on:
                deg_step = np.around(360.0 / rotation_count)
                for i in range(rotation_count):
                    cur_rot = int(i * deg_step)
                    cur_cad = cad + "_" + str(cur_rot)
                    unique_cads.append(cur_cad)
            else:
                unique_cads.append(cad)
    
    # unique_cads = list(unique_cads)

    cad_dataset: Dataset = data.InferenceDataset(shapenet_path, unique_cads, ".df", "cad", transformation=trans)                                       
    cad_dataloader = torch.utils.data.DataLoader(dataset=cad_dataset, shuffle=False, batch_size=batch_size)
    cad_count = len(unique_cads)
    embeddings_all: Dict[str, Dict] = {}
    for category in categories:
        embeddings_all[category] = {}

    print("Embed [", cad_count, "] CAD models with [",rotation_count, "] rotations from [", len(categories), "] categories")

    for names, elements in tqdm(cad_dataloader, total=len(cad_dataloader)):
        # Move data to GPU
        elements = elements.to(device)
        with torch.no_grad():
            cad_latent = embedding_model.embed(elements)
        for idx, name in enumerate(names):
            cur_cat = name.split("/")[0]
            embeddings_all[cur_cat][name] = cad_latent[idx].cpu().numpy().squeeze()
    
    if output:
        torch.save(embeddings_all, output_path)
        print("Output CAD embeddings to [", output_path, "]")


def embedding_tsne(cad_embeddings_path: str, scan_embeddings_path: str, out_path: str,
                   joint_embedding: bool = True, visualize_on: bool = True, rot_count: int = 12):
    
    rotation_step = 360 / rot_count

    cad_embeddings_dict = torch.load(cad_embeddings_path)

    scan_embeddings_dict = torch.load(scan_embeddings_path)

    sample_rate_cad = 20
    sample_rate_scan = 10

    cad_embeddings = []
    cad_cat = []
    cad_rot = []
    cad_flag = []
    count = 0
    cats = list(cad_embeddings_dict.keys())
    for cat, cat_embeddings_dict in cad_embeddings_dict.items():
        for cad_id, cad_embedding in cat_embeddings_dict.items():
            if np.random.randint(sample_rate_cad)==0: # random selection
                rot = int(int(cad_id.split('_')[1])/rotation_step)
                #print(rot)
                cad_embeddings.append(cad_embedding)
                cad_cat.append(cat)
                cad_rot.append(rot)
                cad_flag.append(0) # is cad
            count += 1
    
    cad_embeddings = np.asarray(cad_embeddings)

    if joint_embedding:
        scan_embeddings = []
        scan_cat = []
        scan_flag = []
        count = 0
        cats = list(scan_embeddings_dict.keys())
        for cat, cat_embeddings_dict in scan_embeddings_dict.items():
            for scan_id, scan_embedding in cat_embeddings_dict.items():
                if np.random.randint(sample_rate_scan)==0: # random selection
                    scan_embeddings.append(scan_embedding)
                    scan_cat.append(cat)
                    scan_flag.append(1) # is scan
                count += 1
        
        scan_embeddings = np.asarray(scan_embeddings)

        joint_embeddings = np.vstack((cad_embeddings, scan_embeddings))
        joint_cat = cad_cat + scan_cat
        joint_flag = cad_flag + scan_flag

        print("Visualize the joint embedding space of scan and CAD")
        tsne = TSNE(n_components=2, init='pca', random_state=501)
        embedding_tsne = tsne.fit_transform(joint_embeddings)
    
    else:
        print("Visualize the embedding space of CAD")
        tsne = TSNE(n_components=2, init='pca', random_state=501)
        embedding_tsne = tsne.fit_transform(cad_embeddings)


    # Visualization (2 dimensional)
    x_min, x_max = embedding_tsne.min(0), embedding_tsne.max(0)
    embedding_tsne = (embedding_tsne - x_min) / (x_max - x_min)  # normalization
    
    marker_list = ['o', '>', 'x', '.', ',', '+', 'v', '^', '<', 's', 'd', '8']

    legends = []
    cat_names = []
    cat_idxs = [] 
    # deal with 'others' category
    for cat in cats:
        cat_name = utils.get_category_name(cat)
        cat_idx = utils.get_category_idx(cat)+1
        if cat_name not in cat_names:
            cat_names.append(cat_name)
        if cat_idx not in cat_idxs:
            cat_idxs.append(cat_idx)

    for i in range(len(cat_names)):
        legend = mpatches.Patch(color=plt.cm.Set1(cat_idxs[i]), label=cat_names[i])
        legends.append(legend)

    legends = list(set(legends))
    
    plt.figure(figsize=(10, 10))
    for i in range(embedding_tsne.shape[0]):
        if joint_embedding:
            if joint_flag[i] == 0: # cad
                plt.scatter(embedding_tsne[i, 0], embedding_tsne[i, 1], s=40, color=plt.cm.Set1(utils.get_category_idx(joint_cat[i])+1),
                        marker=marker_list[0], alpha = 0.5)
            else: # scan
                plt.scatter(embedding_tsne[i, 0], embedding_tsne[i, 1], s=40, color=plt.cm.Set1(utils.get_category_idx(joint_cat[i])+1), 
                        marker=marker_list[1])
        else: # only cad embeddings
            plt.scatter(embedding_tsne[i, 0], embedding_tsne[i, 1], s=40, color=plt.cm.Set1(utils.get_category_idx(cad_cat[i])+1), marker=marker_list[cad_rot[i]], alpha = 0.8) # cad only


    plt.xticks([])
    plt.yticks([])
    # plt.legend(handles=lengends, loc='upper left', fontsize=12)
    plt.legend(handles=legends, loc='best', fontsize=16)

    #plt.savefig(os.path.join(out_path, "cad_embedding_tsne.jpg"), dpi=1000)
    if joint_embedding:
        save_path =  out_path+"_joint_embedding_tsne.jpg"
    else:
        save_path =  out_path+"_cad_embedding_tsne.jpg"
        
    plt.savefig(save_path, dpi=1000)
    print("TSNE image saved to [", save_path, " ]")

    if visualize_on:
        wandb.log({"embedding_tsne": plt})

'''
    Retrieve cads in a list of scans and then apply CAD to scan alignment
'''
def retrieve_in_scans(separation_model: nn.Module, completion_model: nn.Module, classification_model: nn.Module,
                      embedding_model: nn.Module, device, test_scan_list: List[str], cad_embeddings_path: str, cad_appearance_file: str,
                      scan_base_path: str, cad_voxel_base_path: str, cad_pc_base_path: str, result_out_path: str, 
                      scan_dataset_name: str = "scannet", separation_model_on: bool = False, 
                      batch_size: int = 1, trans=data.to_occupancy_grid, rotation_count: int = 1, 
                      filter_pool: bool = True, in_the_wild: bool = True, init_scale_method: str = "naive", 
                      icp_mode: str = "p2p", corr_dist_thre_scale = 4.0, estimate_scale_icp: bool = False,
                      rot_only_around_z: bool = False, use_coarse_reg_only: bool = False,
                      visualize: bool = False, wb_vis_name: str = "2d3ds_test/"):
                      
    
    cad_embeddings = torch.load(cad_embeddings_path)
    
    all_categories = list(cad_embeddings.keys())
    selected_categories = ["03001627", "04379243", "02747177", "02818832", "02871439", "02933112", "04256520", "other"]
    #{"03001627": "Chair", "04379243": "Table", "02747177": "Trash bin", "02818832": "Bed",  "02871439": "Bookshelf", "02933112": "Cabinet", "04256520": "Sofa"}
                     
    other_categories = list(set(all_categories).difference(set(selected_categories)))

    if separation_model_on:
        scan_input_format = ".sdf"
        scan_input_folder_extension = "_object_voxel"
        scan_pc_folder_extension = "_object_pc"
        input_only_mask = False
    else:
        scan_input_format = ".mask"
        scan_input_folder_extension = "_mask_voxel"
        scan_pc_folder_extension = "_mask_pc"
        input_only_mask = True
    
    if not in_the_wild:
        with open(cad_appearance_file) as f:
            cad_appearance_dict = json.load(f)

    unique_scan_objects, scene_scan_objects = get_scan_objects_in_scenes(test_scan_list, scan_base_path, 
                            extension = scan_input_format, folder_extension=scan_input_folder_extension)


    scan_dataset: Dataset = data.InferenceDataset(scan_base_path, unique_scan_objects, scan_input_format, "scan",
                                                 transformation=trans, scan_dataset = scan_dataset_name, input_only_mask=input_only_mask)  
    scan_dataloader = torch.utils.data.DataLoader(dataset=scan_dataset, shuffle=False, batch_size=batch_size)

    rotation_ranking_on = False
    if rotation_count > 1:
        rotation_ranking_on = True
    
    deg_step = np.around(360.0 / rotation_count)

    # load all the scan object and cads that are waiting for testing
    # # Evaluate all unique scan segments' embeddings
    scan_embeddings: Dict[str, np.array] = {}
    completed_voxels: Dict[str, np.array] = {} # saved the completed scan object (tensor) [potential issue: limited memory]
    mid_pred_cats: Dict[str, str] = {}
    for names, elements in tqdm(scan_dataloader, total=len(scan_dataloader)):
        # Move data to GPU
        elements = elements.to(device)
        with torch.no_grad():
            if separation_model_on:
                scan_foreground, _ = separation_model(elements)
                scan_foreground = torch.sigmoid(scan_foreground)
                scan_completed, hidden = completion_model(scan_foreground) 
            else:
                scan_completed, hidden = completion_model(elements)   
            mid_pred_cat = classification_model.predict_name(torch.sigmoid(hidden))  # class str
            scan_completed = torch.sigmoid(scan_completed)
            scan_latent = embedding_model.embed(scan_completed)

        for idx, name in enumerate(names):
            scan_embeddings[name] = scan_latent[idx].cpu().numpy().squeeze()
            mid_pred_cats[name] = mid_pred_cat[idx]
            if init_scale_method == "bbx":
                # record the completed object
                completed_voxels[name] = scan_completed[idx].cpu().numpy()
                
    results = []

    # TODO: try to make it running in parallel to speed up
    for scene_name, scan_objects in scene_scan_objects.items():
        scene_results = {"id_scan": scene_name, "aligned_models": []}
        print("Process scene [", scene_name, "]")
        print("---------------------------------------------")
        for scan_object in tqdm(scan_objects, total=len(scan_objects)):
            
            print("Process scan segement [", scan_object, "]")

            scan_object_embedding = scan_embeddings[scan_object][np.newaxis, :]
            
            if not in_the_wild:
                cad_embeddings_in_scan = {}
                cad_list_in_scan = list(cad_appearance_dict[scene_name].keys()) 
                for cad in cad_list_in_scan:
                    parts = cad.split("_")
                    cat = parts[0]
                    cad_id = parts[1]
                    if cat not in cad_embeddings_in_scan.keys():
                        cad_embeddings_in_scan[cat] = {}
                    
                    if rotation_ranking_on:
                        for i in range(rotation_count):
                            cur_rot = int(i * deg_step)
                            cad_str = cat+"/"+cad_id+"_" + str(cur_rot)
                            cad_embeddings_in_scan[cat][cad_str] = cad_embeddings[cat][cad_str]
                    else:  
                        cad_str = cat+"/"+cad_id
                        cad_embeddings_in_scan[cat][cad_str] = cad_embeddings[cat][cad_str]
            else:
                cad_embeddings_in_scan = cad_embeddings

            all_categories = list(cad_embeddings_in_scan.keys())        
            other_categories = list(set(all_categories).difference(set(selected_categories)))

            filtered_cad_embeddings = {}
            mid_pred_cat = mid_pred_cats[scan_object] # class str
            print("Predicted category:", utils.get_category_name(mid_pred_cat))

            if mid_pred_cat != 'other':
                if filter_pool and mid_pred_cat in all_categories:
                    filtered_cad_embeddings = cad_embeddings_in_scan[mid_pred_cat]
                else: # search in the whole model pool (when we do not enable pool filtering or when the category prediction is not in the pool's category keys)
                    for cat in all_categories:
                        filtered_cad_embeddings = {**filtered_cad_embeddings, **cad_embeddings_in_scan[cat]}
            else: # if is classified as 'other', search in the categories of 'other' (when other categories do exsit in the model pool's keys)
                if filter_pool and len(other_categories)>0:
                    for cat in other_categories:
                        filtered_cad_embeddings = {**filtered_cad_embeddings, **cad_embeddings_in_scan[cat]}
                else:
                    for cat in all_categories:
                        filtered_cad_embeddings = {**filtered_cad_embeddings, **cad_embeddings_in_scan[cat]}

            
            pool_names = list(filtered_cad_embeddings.keys())
            pool_embeddings = [filtered_cad_embeddings[p] for p in pool_names]
            pool_embeddings = np.asarray(pool_embeddings)

            # Compute distances in embedding space
            distances = scipy.spatial.distance.cdist(scan_object_embedding, pool_embeddings, metric="euclidean") # figure out which distance is the better
            sorted_indices = np.argsort(distances, axis=1)
            sorted_distances = np.take_along_axis(distances, sorted_indices, axis=1) # [1, filtered_pool_size * rotation_trial_count]
            sorted_distances = sorted_distances[0] # [filtered_pool_size * rotation_trial_count]

            predicted_ranking = np.take(pool_names, sorted_indices)[0].tolist()

            # apply registration

            # load scan segment (target cloud)
            parts = scan_object.split("_")
            if scan_dataset_name == 'scannet': 
                scan_name = parts[0] + "_" + parts[1]
                object_name = scan_name + "__" + parts[3]
                scan_path = os.path.join(scan_base_path, scan_name, scan_name + scan_input_folder_extension, scan_object + scan_input_format)
                scan_pc_path = os.path.join(scan_base_path, scan_name, scan_name + scan_pc_folder_extension, scan_object + ".pcd")
            elif scan_dataset_name == '2d3ds': 
                area_name = parts[0]+"_"+parts[1]
                room_name = parts[2]+"_"+parts[3]
                scan_path = os.path.join(scan_base_path, area_name, room_name, room_name + scan_input_folder_extension, scan_object + scan_input_format)
                scan_pc_path = os.path.join(scan_base_path, area_name, room_name, room_name + scan_pc_folder_extension, scan_object + ".pcd")
            
            if separation_model_on:
                scan_voxel_raw = data.load_raw_df(scan_path)
                scan_voxel = data.to_occupancy_grid(scan_voxel_raw).tdf.reshape((32, 32, 32))
            else:
                scan_voxel_raw = data.load_mask(scan_path)
                scan_voxel = scan_voxel_raw.tdf.reshape((32, 32, 32))
            
            scan_grid2world = scan_voxel_raw.matrix
            scan_voxel_res = scan_voxel_raw.size
            #print("Scan voxel shape:", scan_voxel.shape)

            #res = scan_grid2world[0,0]
            
            if init_scale_method == "bbx":
                # get the completed scan object (voxel representation)
                scan_completed_voxel = completed_voxels[scan_object] # np.array
                # rotate to CAD's canonical system
                if rotation_ranking_on:
                    top_1_predicted_rotation = int(predicted_ranking[0].split("_")[1]) # deg
                else:
                    top_1_predicted_rotation = 0
                scan_completed_voxel_rot = data.rotation_augmentation_interpolation_v3(scan_completed_voxel, "dummy", aug_rotation_z = -top_1_predicted_rotation).reshape((32, 32, 32))
                # calculate bbx length of the rotated completed scan object in voxel space (unit: voxel) 
                scan_bbx_length = data.cal_voxel_bbx_length(scan_completed_voxel_rot) # output: np.array([bbx_lx, bbx_ly, bbx_lz])
            
            scan_voxel_wb = data.voxel2point(scan_voxel, name=scan_object)
            if visualize:
                wandb_vis_dict = {wb_vis_name+"input scan object": wandb.Object3D(scan_voxel_wb)}
            
            # load point cloud
            scan_seg_pcd = o3d.io.read_point_cloud(scan_pc_path)
            scan_seg_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=scan_voxel_res, max_nn=30))

            # initialization
            candidate_cads = []
            candidate_cad_pcds = []
            candidate_rotations = []
            candidate_cads_voxel_wb = []
            candidate_T_init = []
            candidate_T_reg = []
            candidate_fitness_reg = []
            
            # Apply registration
            corr_dist_thre = corr_dist_thre_scale * scan_voxel_res
            top_k = 4
            for top_i in range(top_k):
                # load cad (source cloud)
                predicted_cad = predicted_ranking[top_i]
                title_str = "Top "+ str(top_i+1) + " retrieved model "
                print(title_str, predicted_cad)
                if rotation_ranking_on:
                    predicted_rotation = int(predicted_cad.split("_")[1]) # deg
                else:
                    predicted_rotation = 0
                predicted_cat = utils.wandb_color_lut(predicted_cad.split("/")[0])
                predicted_cat_name = utils.get_category_name(predicted_cad.split("/")[0])
                cad_voxel_path = os.path.join(cad_voxel_base_path, predicted_cad.split("_")[0] + ".df")
                cad_voxel = data.to_occupancy_grid(data.load_raw_df(cad_voxel_path)).tdf
                cad_voxel_rot = data.rotation_augmentation_interpolation_v3(cad_voxel, "dummy", aug_rotation_z = predicted_rotation).reshape((32, 32, 32))    
                cad_voxel_wb = data.voxel2point(cad_voxel_rot, predicted_cat, name=predicted_cat_name + ":" + predicted_cad)
                if visualize:
                    wandb_vis_dict[wb_vis_name + title_str] = wandb.Object3D(cad_voxel_wb)
                cad_pcd_path = os.path.join(cad_pc_base_path, predicted_cad.split("_")[0] + ".pcd")
                cad_pcd = o3d.io.read_point_cloud(cad_pcd_path) 
                if icp_mode == "p2l": # requires normal
                    cad_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
                
                if init_scale_method == "bbx":
                    # calculate bbx length of the (none rotated) retrieved cad in voxel space (unit: voxel) 
                    cad_bbx_length = data.cal_voxel_bbx_length(cad_voxel) # output: np.array([bbx_lx, bbx_ly, bbx_lz]) 
                    cad_scale_multiplier = scan_bbx_length / cad_bbx_length # potential issue (int/int), result should be float
                    direct_scale_on = False
                elif init_scale_method == "learning":
                    direct_scale_on = True
                    cad_scale_multiplier = np.ones(3) # comment later, replace with the predicted value
                else: # init_scale_method == "naive":
                    cad_scale_multiplier = np.ones(3)
                    direct_scale_on = False

                # Transformation initial guess    
                T_init = data.get_tran_init_guess(scan_grid2world, predicted_rotation, direct_scale = direct_scale_on, 
                                                  cad_scale_multiplier=cad_scale_multiplier) 
                
                # Apply registration 
                if icp_mode == "p2l": # point-to-plane distance metric
                    T_reg, eval_reg = data.reg_icp_p2l_o3d(cad_pcd, scan_seg_pcd, corr_dist_thre, T_init, estimate_scale_icp, rot_only_around_z)
                else:  # point-to-point distance metric
                    T_reg, eval_reg = data.reg_icp_p2p_o3d(cad_pcd, scan_seg_pcd, corr_dist_thre, T_init, estimate_scale_icp, rot_only_around_z)

                fitness_reg = eval_reg.fitness
                candidate_cads.append(predicted_cad)
                candidate_cad_pcds.append(cad_pcd)
                candidate_rotations.append(predicted_rotation)
                candidate_cads_voxel_wb.append(cad_voxel_wb)
                candidate_T_init.append(T_init)
                candidate_T_reg.append(T_reg)
                candidate_fitness_reg.append(fitness_reg)
            
            candidate_fitness_reg = np.array(candidate_fitness_reg)
            best_idx = np.argsort(candidate_fitness_reg)[-1]
            T_reg_best = candidate_T_reg[best_idx]
            T_reg_init_best = candidate_T_init[best_idx]
            cad_best_pcd = candidate_cad_pcds[best_idx]
            cad_best = candidate_cads[best_idx].split("_")[0]
            cad_best_cat = cad_best.split("/")[0]
            cad_best_id = cad_best.split("/")[1]
            cat_best_name = utils.get_category_name(cad_best_cat)

            pair_before_reg = data.o3dreg2point(cad_best_pcd, scan_seg_pcd, T_reg_init_best, down_rate = 5)
            pair_after_reg = data.o3dreg2point(cad_best_pcd, scan_seg_pcd, T_reg_best, down_rate = 5)

            print("Select retrived model [", cad_best, "] ( Top", str(best_idx+1), ") as the best model")
            print("The best transformation:")
            print(T_reg_best)
            
            if visualize:
                wandb.log(wandb_vis_dict) # retrieval

                wandb.log({"reg/"+"before registration ": wandb.Object3D(pair_before_reg),
                          "reg/"+"after registration ": wandb.Object3D(pair_after_reg)}) # registration

            if use_coarse_reg_only:
                T_reg_best = T_reg_init_best.tolist() # for json format 
            else:
                T_reg_best = T_reg_best.tolist() # for json format 

            aligned_model = {"scan_obj_id": scan_object, "catid_cad": cad_best_cat, "cat_name": cat_best_name, "id_cad": cad_best_id, "trs_mat": T_reg_best}
                            
            scene_results["aligned_models"].append(aligned_model)

            # TODO: refine the results use a scene graph based learning
            
        results.append(scene_results)

    # write to json
    with open(result_out_path, "w") as f:
        json.dump(results, f, indent=4)
        print("Write the results to [", result_out_path, "]")
             
    
def get_scan_objects_in_scenes(test_scan_list: List[str], scan_dataset_path: str, extension: str = ".sdf", folder_extension: str = "_object_voxel") -> Tuple[List[str], Dict[str, List[str]]]:  

    ##example: (ScanNet) scene0085_00
    ##example: (2D3DS) Area_3/office_1  [extension = ".mask", folder_extension="_mask_voxel"]

    unique_scan_objects = []
    scene_scan_objects = {}

    for test_scene in test_scan_list:
        scene_scan_objects[test_scene]=[]
        room_scene = test_scene.split("/")[-1]
        test_scene_path = os.path.join(scan_dataset_path, test_scene, room_scene+folder_extension)
        for scan_object_file in os.listdir(test_scene_path):
        # example (ScanNet): scene0085_00__1
        # example (2D3DS): Area_3_lounge_1_chair_18_3
        
            if scan_object_file.endswith(extension):
                unique_scan_objects.append(scan_object_file.split(".")[0])
                scene_scan_objects[test_scene].append(scan_object_file.split(".")[0])
    
    return unique_scan_objects, scene_scan_objects
                
# def get_unique_samples(dataset_path: str, interested_cat: List[str]) -> Tuple[List[str], List[str]]:
def get_unique_samples(dataset_path: str, rotation_count: int = 1, test_sample_limit: int = 999999) -> Tuple[List[str], List[str], List[int]]:

    unique_scans = set()
    unique_cads = set()

    rotation_ranking_on = False
    if rotation_count > 1:
        rotation_ranking_on = True

    with open(dataset_path) as f:
        samples = json.load(f).get("samples")
        test_sample_limit = min(len(samples), test_sample_limit)
        random_sample_idx = np.random.choice(len(samples), test_sample_limit, replace=False)
        random_sample_idx = random_sample_idx.tolist()
        samples = list(samples[i] for i in random_sample_idx)

    for sample in tqdm(samples, desc="Gather unique samples"):
        scan = sample["reference"]["name"].replace("/scan/", "")
        unique_scans.add(scan)

        for cad_element in sample["pool"]:
            cad = cad_element["name"].replace("/cad/", "")
            # if cad.split("/")[0] in interested_cat:
            if rotation_ranking_on:
                deg_step = np.around(360.0 / rotation_count)
                for i in range(rotation_count):
                    cur_rot = int(i * deg_step)
                    cur_cad = cad + "_" + str(cur_rot)
                    unique_cads.add(cur_cad)
            else:
                unique_cads.add(cad)

    print(f"# Unique scan samples: {len(unique_scans)}, # unique cad element with unique rotation {len(unique_cads)}")
    
    # unique_scan format example:
    # scene0085_00__0_02818832_e91c2df09de0d4b1ed4d676215f46734
    # unique_cad format example:
    # 03001627/811c349efc40c6feaf288f952624966_150

    return list(unique_scans), list(unique_cads), random_sample_idx


if __name__ == '__main__':
    args = utils.command_line_parser()
    main(args)


'''
# BACKUP

def retrieve_in_scene(separation_model: nn.Module, completion_model: nn.Module, classification_model: nn.Module,
                      embedding_model: nn.Module, device, test_scan_list: List[str], scannet_path: str,
                      cad_embeddings_path: str, cad_pcd_path: str, result_out_path: str, 
                      batch_size: int = 1, trans=data.to_occupancy_grid,
                      rotation_count: int = 1, filter_pool: bool = True, icp_mode: str = "p2p", corr_dist_thre_scale = 4.0,
                      estimate_scale_icp: bool = False, rot_only_around_z: bool = False, 
                      use_coarse_reg_only: bool = False, visualize: bool = False):
                      
    
    cad_embeddings = torch.load(cad_embeddings_path)
    
    all_categories = list(cad_embeddings.keys())
    selected_categories = ["03001627", "04379243", "02747177", "02818832", "02871439", "02933112", "04256520", "other"]
    other_categories = list(set(all_categories).difference(set(selected_categories)))

    unique_scan_objects, scene_scan_objects = get_scan_objects_in_scenes(test_scan_list, scannet_path)

    scan_dataset: Dataset = data.InferenceDataset(scannet_path, unique_scan_objects, ".sdf", "scan",
                                                 transformation=trans)  # change to the same as training (the scan segment needed to exsit)
    scan_dataloader = torch.utils.data.DataLoader(dataset=scan_dataset, shuffle=False, batch_size=batch_size)

    rotation_ranking_on = False
    if rotation_count > 1:
        rotation_ranking_on = True

    # load all the scan object and cads that are waiting for testing
    # # Evaluate all unique scan segments' embeddings
    scan_embeddings: Dict[str, np.array] = {}
    mid_pred_cats: Dict[str, str] = {}
    for names, elements in tqdm(scan_dataloader, total=len(scan_dataloader)):
        # Move data to GPU
        elements = elements.to(device)
        with torch.no_grad():
            scan_foreground, _ = separation_model(elements)
            scan_foreground = torch.sigmoid(scan_foreground)
            scan_completed, hidden = completion_model(scan_foreground) 
            mid_pred_cat = classification_model.predict_name(torch.sigmoid(hidden))  # class str
            scan_completed = torch.sigmoid(scan_completed)
            scan_latent = embedding_model.embed(scan_completed)

        for idx, name in enumerate(names):
            scan_embeddings[name] = scan_latent[idx].cpu().numpy().squeeze()
            mid_pred_cats[name] = mid_pred_cat[idx]

    results = []
    
    # TODO: try to make it running in parallel to speed up
    for scene_name, scan_objects in scene_scan_objects.items():
        scene_results = {"id_scan": scene_name, "aligned_models": []}
        print("Process scene [", scene_name, "]")
        print("---------------------------------------------")
        for scan_object in tqdm(scan_objects, total=len(scan_objects)):
            
            print("Process scan segement [", scan_object, "]")

            scan_object_embedding = scan_embeddings[scan_object][np.newaxis, :]
            
            filtered_cad_embeddings = {}
            mid_pred_cat = mid_pred_cats[scan_object] # class str
            print("Predicted category:", utils.get_category_name(mid_pred_cat))
            if filter_pool:
                if mid_pred_cat != 'other': 
                    filtered_cad_embeddings = cad_embeddings[mid_pred_cat]
                else:
                    for cat in other_categories:
                        filtered_cad_embeddings = {**filtered_cad_embeddings, **cad_embeddings[cat]}
            else:
                for cat in all_categories:
                    filtered_cad_embeddings = {**filtered_cad_embeddings, **cad_embeddings[cat]}
            
            pool_names = list(filtered_cad_embeddings.keys())
            pool_embeddings = [filtered_cad_embeddings[p] for p in pool_names]
            pool_embeddings = np.asarray(pool_embeddings)

            # Compute distances in embedding space
            distances = scipy.spatial.distance.cdist(scan_object_embedding, pool_embeddings, metric="euclidean")
            sorted_indices = np.argsort(distances, axis=1)
            sorted_distances = np.take_along_axis(distances, sorted_indices, axis=1) # [1, filtered_pool_size * rotation_trial_count]
            sorted_distances = sorted_distances[0] # [filtered_pool_size * rotation_trial_count]

            predicted_ranking = np.take(pool_names, sorted_indices)[0].tolist()

            # apply registration

            # load scan segment (target cloud)
            parts = scan_object.split("_")
            scan_name = parts[0] + "_" + parts[1]
            object_name = scan_name + "__" + parts[3]
            scan_path = os.path.join(scannet_path, scan_name, scan_name + "_objects", object_name + ".sdf")
            scan_voxel_raw = data.load_raw_df(scan_path)
            scan_grid2world = scan_voxel_raw.matrix
            scan_voxel_res = scan_voxel_raw.size
            scan_pcd_path = os.path.join(scannet_path, scan_name, scan_name + "_objects_pcd", object_name + ".pcd")
            scan_seg_pcd = o3d.io.read_point_cloud(scan_pcd_path)
            scan_seg_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=scan_voxel_res, max_nn=30))
            
            # load cad (source cloud)
            predicted_cad_1 = predicted_ranking[0]
            print("Top 1 retrieved model:", predicted_cad_1)
            predicted_rotation_1 = int(predicted_cad_1.split("_")[1]) # deg
            cad_pcd_path_1 = os.path.join(cad_pcd_path, predicted_cad_1.split("_")[0] + ".pcd")
            cad_pcd_1 = o3d.io.read_point_cloud(cad_pcd_path_1) 
            if icp_mode == "p2l":
                cad_pcd_1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

            predicted_cad_2 = predicted_ranking[1]
            print("Top 2 retrieved model:", predicted_cad_2)
            predicted_rotation_2 = int(predicted_cad_2.split("_")[1]) # deg
            cad_pcd_path_2 = os.path.join(cad_pcd_path, predicted_cad_2.split("_")[0] + ".pcd")
            cad_pcd_2 = o3d.io.read_point_cloud(cad_pcd_path_2) 
            if icp_mode == "p2l":
                cad_pcd_2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

            predicted_cad_3 = predicted_ranking[2]
            print("Top 3 retrieved model:", predicted_cad_3)
            predicted_rotation_3 = int(predicted_cad_3.split("_")[1]) # deg
            cad_pcd_path_3 = os.path.join(cad_pcd_path, predicted_cad_3.split("_")[0] + ".pcd")
            cad_pcd_3 = o3d.io.read_point_cloud(cad_pcd_path_3)
            if icp_mode == "p2l":
                cad_pcd_3.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

            predicted_cad_4 = predicted_ranking[3]
            print("Top 4 retrieved model:", predicted_cad_4)
            predicted_rotation_4 = int(predicted_cad_4.split("_")[1]) # deg
            cad_pcd_path_4 = os.path.join(cad_pcd_path, predicted_cad_4.split("_")[0] + ".pcd")
            cad_pcd_4 = o3d.io.read_point_cloud(cad_pcd_path_4)
            if icp_mode == "p2l":
                cad_pcd_4.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

            candidate_cads = [predicted_cad_1, predicted_cad_2, predicted_cad_3, predicted_cad_4]
            candidate_cad_pcds = [cad_pcd_1, cad_pcd_2, cad_pcd_3, cad_pcd_4]
            candidate_rotations = [predicted_rotation_1, predicted_rotation_2, predicted_rotation_3, predicted_rotation_4]
            
            print("Apply registration")
            # Transformation initial guess   
            # TODO: find out better way to estimate the scale init guess (use bounding box of the foreground and cad voxels) !!! try it later 
            cad_scale_mult = 1.05          
            T_init_1 = data.get_tran_init_guess(scan_grid2world, predicted_rotation_1, cad_scale_multiplier=cad_scale_mult)
            T_init_2 = data.get_tran_init_guess(scan_grid2world, predicted_rotation_2, cad_scale_multiplier=cad_scale_mult)
            T_init_3 = data.get_tran_init_guess(scan_grid2world, predicted_rotation_3, cad_scale_multiplier=cad_scale_mult)
            T_init_4 = data.get_tran_init_guess(scan_grid2world, predicted_rotation_4, cad_scale_multiplier=cad_scale_mult)

            # Apply registration (point-to-plane)
            corr_dist_thre = corr_dist_thre_scale * scan_voxel_res
            if icp_mode == "p2l": # point-to-plane distance metric
                T_reg_1, eval_reg_1 = data.reg_icp_p2l_o3d(cad_pcd_1, scan_seg_pcd, corr_dist_thre, T_init_1, estimate_scale_icp, rot_only_around_z)
                T_reg_2, eval_reg_2 = data.reg_icp_p2l_o3d(cad_pcd_2, scan_seg_pcd, corr_dist_thre, T_init_2, estimate_scale_icp, rot_only_around_z)
                T_reg_3, eval_reg_3 = data.reg_icp_p2l_o3d(cad_pcd_3, scan_seg_pcd, corr_dist_thre, T_init_3, estimate_scale_icp, rot_only_around_z)
                T_reg_4, eval_reg_4 = data.reg_icp_p2l_o3d(cad_pcd_4, scan_seg_pcd, corr_dist_thre, T_init_4, estimate_scale_icp, rot_only_around_z)
            else:  # point-to-point distance metric
                T_reg_1, eval_reg_1 = data.reg_icp_p2p_o3d(cad_pcd_1, scan_seg_pcd, corr_dist_thre, T_init_1, estimate_scale_icp, rot_only_around_z)
                T_reg_2, eval_reg_2 = data.reg_icp_p2p_o3d(cad_pcd_2, scan_seg_pcd, corr_dist_thre, T_init_2, estimate_scale_icp, rot_only_around_z)
                T_reg_3, eval_reg_3 = data.reg_icp_p2p_o3d(cad_pcd_3, scan_seg_pcd, corr_dist_thre, T_init_3, estimate_scale_icp, rot_only_around_z)
                T_reg_4, eval_reg_4 = data.reg_icp_p2p_o3d(cad_pcd_4, scan_seg_pcd, corr_dist_thre, T_init_4, estimate_scale_icp, rot_only_around_z)

            fitness = np.array([eval_reg_1.fitness, eval_reg_2.fitness, eval_reg_3.fitness, eval_reg_4.fitness])
            best_idx = np.argsort(fitness)[-1]
            T_reg = [T_reg_1, T_reg_2, T_reg_3, T_reg_4]
            T_reg_init = [T_init_1, T_init_2, T_init_3, T_init_4]
            T_reg_best = T_reg[best_idx]
            T_reg_init_best = T_reg_init[best_idx]
            cad_best_pcd = candidate_cad_pcds[best_idx]
            cad_best = candidate_cads[best_idx].split("_")[0]
            cad_best_cat = cad_best.split("/")[0]
            cad_best_id = cad_best.split("/")[1]
            cat_best_name = utils.get_category_name(cad_best_cat)
                      
            pair_before_reg = data.o3dreg2point(cad_best_pcd, scan_seg_pcd, T_reg_init_best, down_rate = 5)
            pair_after_reg = data.o3dreg2point(cad_best_pcd, scan_seg_pcd, T_reg_best, down_rate = 5)
            
            if visualize:
                wandb.log({"reg/"+"before registration ": wandb.Object3D(pair_before_reg),
                          "reg/"+"after registration ": wandb.Object3D(pair_after_reg)})

            print("Select retrived model [", cad_best, "] ( Top", str(best_idx+1), ") as the best model")
            print("The best transformation:")
            print(T_reg_best)

            if use_coarse_reg_only:
                T_reg_best = T_reg_init_best.tolist() # for json format 
            else:
                T_reg_best = T_reg_best.tolist() # for json format 

            aligned_model = {"catid_cad": cad_best_cat, "cat_name": cat_best_name, "id_cad": cad_best_id, "trs_mat": T_reg_best}
                            
            scene_results["aligned_models"].append(aligned_model)

            # TODO: refine the results use a scene graph based learning
            
        results.append(scene_results)

    # write to json
    with open(result_out_path, "w") as f:
        json.dump(results, f, indent=4)
        print("Write the results to [", result_out_path, "]")
'''