import argparse
import os
import json


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def command_line_parser():
    parser = argparse.ArgumentParser(
        description="Joint Embedding (ICCV 2019), edited by Real2CAD group at ETH Zurich")

    parser.add_argument("--name", type=str, default="",
                        help="name of this test")
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint, please specify the model timestamp and checkpoint, example: model_name_xxx_2021-01-01_01-11-11/01000')

    parser.add_argument("--wandb_vis_on", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128,
                        help="how many samples per batch")
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    # Data related
    parser.add_argument("--scan_dataset_name", type=str, default="scannet", help="select from scannet and 2d3ds")

    # dataset path
    parser.add_argument("--scannet_path", type=str,
                        help="path to the base folder of scan object sdf voxel [32x32x32] and (or) foreground mask [32x32x32] of the ScanNet Dataset")
    parser.add_argument("--s2d3ds_path", type=str,
                        help="path to the base folder of scan object sdf voxel [32x32x32] and (or) foreground mask [32x32x32] of the Standford 2D-3D-S Dataset")
    parser.add_argument("--shapenet_voxel_path", type=str,
                        help="path to the base folder of shapenet model df voxel [32x32x32]")
    parser.add_argument("--shapenet_pc_path", type=str,
                        help="path to the base folder of shapenet model's pcd format point cloud")
    # dataset related json path
    parser.add_argument("--scan2cad_file", type=str,
                        help="path to the file recording the 1-to-1 correspondence between scan object and cad model and their dataset split")
    parser.add_argument("--scan2cad_quat_file", type=str,
                        help="path to the file recording the 1-to-1 rotation quaternion from cad model to the scan object")
    parser.add_argument("--modelpool_file", type=str, default=None,
                        help="path to the file recording the potential model pool for match")
    parser.add_argument("--scenelist_file", type=str, default=None,
                        help="path to the file recording the scans waiting for accomplish the Real2CAD task")
    parser.add_argument("--similarity_file", type=str,
                        help="Path to the json file of the scan-cad similarity dataset (for evaluation purpose), use the one filtered from the validation set")
    parser.add_argument("--cad_apperance_file", type=str, default=None,
                        help="Path to the json file that contains the appearance and count of the cad model in each scan, only works when it's a none in-the-wild prediction")
    parser.add_argument("--full_annotation_file", type=str, default=None,
                        help=" path to the file recording full annotation (including scale, and symmetry)")

    # output path
    parser.add_argument("--cad_embedding_path", type=str, default=None,
                        help="Path to the json file of the embedding of the cads in the model pool")
    parser.add_argument("--scan_embedding_path", type=str, default=None,
                        help="Path to the json file of the embedding of the scan segments in the scan segment list")
    parser.add_argument("--real2cad_result_path", type=str,
                        help="Path to the output json of the Real2CAD task's results")
    parser.add_argument("--tsne_img_path", type=str, help="Path to the output folder of the TSNE image")
    parser.add_argument("--output_root", type=str, help="Base path of the training model outputs")

    # run mode
    parser.add_argument("--embed_mode", type=str2bool, nargs='?', const=True, default=False,
                        help="only for test, true when the embeddings of the cad pool is not already generated, false when you want to directly load the already generated embeddings")
    parser.add_argument("--in_the_wild_mode", type=str2bool, nargs='?', const=True, default=True,
                        help="only for test, true when the cad appearance in each scan is not konwn beforehand, false when we have a ground truth set of CAD models given (Scan2CAD benchmark's case)")

    # Model configuration
    parser.add_argument("--representation", type=str, default="binary",
                        help="representation of the voxel, selected from tdf and binary")

    parser.add_argument("--backbone_model_name", type=str, default="resnet3d", help="TODO")

    parser.add_argument("--separation_model_on", type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument("--skip_connection_sep", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--skip_connection_com", type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument("--balanced_weight", type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument("--negative_sample_strategy", type=str, default="other_cat",
                        help="select from other_cat, same_cat and mix")

    parser.add_argument("--same_category_negative_epoch", type=int, default=0,
                        help="before this epoch, we sample negative samples in other category to have a rough metric learning, after this epoch, we sample negative samples in the same category as the anchor")

    parser.add_argument("--loss_retrieval", type=str, default="tripletMargin",
                        help="select from tripletMargin or circle loss")

    parser.add_argument("--offline_sample", type=str2bool, nargs='?', const=True, default=True,
                        help="Assign positive and negative samples offline or finding (mining) pairs online in a mini-batch")

    parser.add_argument("--miner_type", type=str, default="base",
                        help="online triplet mining strategy, select from semihard, easyhard and others, only works for online sampling (offline_sample=false)")

    parser.add_argument("--train_scale", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--train_scale_range", type=float, nargs="+")

    parser.add_argument("--train_symmetry", type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument("--triplet_margin", type=float, default=0.2)
    parser.add_argument("--circle_margin", type=float, default=0.4)
    parser.add_argument("--circle_gamma", type=float, default=80.0)

    parser.add_argument("--weight_fgs", type=float, default=1.0)
    parser.add_argument("--weight_bgs", type=float, default=1.0)  # deprecated
    parser.add_argument("--weight_com", type=float, default=1.0)
    parser.add_argument("--weight_cls", type=float, default=1.0)  # classification
    parser.add_argument("--weight_sym", type=float, default=1.0)  # symmetry
    parser.add_argument("--weight_scale", type=float, default=1.0)  # scale
    parser.add_argument("--weight_retr", type=float, default=1.0)

    parser.add_argument("--retrieval_epoch", type=int, default=0,
                        help="embedding model would be freezed before this epoch")
    parser.add_argument("--finetune_epoch", type=int, default=1000,
                        help="all the models before embedding would be freezed after this epoch, for fine tuning")

    parser.add_argument("--rotation_augmentation", type=str, default="random",
                        help="fixed, random, none")  # only apply the rotation augmentation
    parser.add_argument("--canonical_learning_step", type=int, default=5,
                        help="define how many epoches would we wait to switch the rotation augmentation angle, only works when rotation_augmentation==fixed, used for canonical transformation learning")

    parser.add_argument("--flip_augmentation", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--jitter_augmentation", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument("--align_to_scan", type=str2bool, nargs='?', const=True, default=False,
                        help="align cad to scan or align scan to cad (canonical system)")
    parser.add_argument("--rotation_trial_count", type=int, default=12,
                        help="number of evenly seperated rotation trial (around z axis) of the CAD model while retrieval")

    parser.add_argument("--filter_val_pool", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument("--init_scale_method", type=str, default="naive", help="the method used to estimate the scale initial guess, select from naive, bbx and learning" )

    # registration related

    parser.add_argument("--icp_reg_mode", type=str, default="p2p",
                        help="distance metric of ICP registration, select from p2p (point to point) and p2l (point to plane)")
    parser.add_argument("--icp_dist_thre", type=float, default=4.0,
                        help="nearest neighbor correspondence distance threshold, multiply voxel resolution of the scan segement")
    parser.add_argument("--icp_with_scale_on", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--only_rot_z", type=str2bool, nargs='?', const=True, default=False,
                        help="Only estimate the rotation around z axis during registration")
    parser.add_argument("--only_coarse_reg", type=str2bool, nargs='?', const=True, default=False,
                        help="Directly use the coarse registration results, do not rely on ICP fine regsitration")

    parser.add_argument("--val_max_sample_count", type=int, default=100,
                        help="number of the used scan object samples for calculating the retrieval metrics while each validation step")
    parser.add_argument("--val_vis_sample_count", type=int, default=5,
                        help="number of the scan object samples for visualization the retrieval performance")

    parser.add_argument("--log_frequency", type=int, default=10,
                        help="unit: iteration, current training loss would be printed")
    parser.add_argument("--validate_frequency", type=int, default=200,
                        help="unit: iteration, current model would be tested on the validation set")
    parser.add_argument("--evaluation_frequency", type=int, default=400,
                        help="unit: iteration, current model would be evaluated using the similarity dataset (should be the multiplier of validate_frequency)")
    parser.add_argument("--checkpoint_frequency", type=int, default=1000,
                        help="unit: iteration, current model would be saved in the output folder")

    args = parser.parse_args()

    return args
