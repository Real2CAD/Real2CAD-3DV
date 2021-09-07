#!/bin/bash

base_path=/cluster/project/infk/courses/3d_vision_21/group_14/1_data

# specify the test name here
#test_name=wobg_trihard_wosc_binary_v3_p100
#test_name=wobg_trihard_class_binary_v3_p100
#test_name=wobg_circle_class_binary_v3_p100
#test_name=wobg_trioff_class_binary_canon_v3_p100
#test_name=trioffhardmix_lp_skip2s_tdf_v3_p100
#test_name=trioffhardmix_lp_skip2s_binary_v3_p100
#test_name=trioffhardmix_lp_skip2s_binary_inter0_v3_p100
#test_name=test_align
#test_name=test_retrieval
#test_name=test_retrieval_without_filter
#test_name=trioffhard_lp_skip3_class_binary_v3_p100
#test_name=trihard_class_binary_v3_p20
#test_name=wobg_trihard_class_binary_v3_p20
#test_name=bal_raug12_align_binary_v3_circle
#test_name=no_skip_with_sep_scannet
test_name=test_train_scale

# Resume from checkpoint
# resume_name=wobg_trioffhard_class_binary_v3_p100
# resume_timestamp=2021-06-07_02-01-02
# resume_checkpoint=best

resume_name=trioffhardmix_lp_skip2s_binary_v3_p100h
resume_timestamp=2021-06-09_09-44-24
resume_checkpoint=11000

# resume_name=trioffhardmix_lp_skip2s_binary_v3_p100h
# resume_timestamp=2021-06-09_18-14-59
# resume_checkpoint=05900

# resume_name=new_with_sep_scannet
# resume_timestamp=2021-07-13_21-09-29
# resume_checkpoint=06200

resume_name=fg_skip_sigmoid_before_with_sep_scannet
resume_timestamp=2021-07-14_12-01-20
resume_checkpoint=01600

# if you want start training from scratch, you need to comment the line begin with --resume
# or you need to add it below
#--resume ${USER}_${resume_name}_${resume_timestamp}/${resume_checkpoint}
# --resume yuepan_${resume_name}_${resume_timestamp}/${resume_checkpoint}


# original model list: model_pool_small_list_v3.json
# test larger model list: model_pool_large.json

# original public similarity file (easy): scan-cad_similarity_public_augmented-100_v3.json
# updated similarity file (harder): scan-cad_similarity_augmented-100_v3.json

#--resume yuepan_${resume_name}_${resume_timestamp}/${resume_checkpoint}

command="CUDA_VISIBLE_DEVICES=0
		python3 ./train.py 
		--name ${USER}_${test_name}
        --scan_dataset_name scannet
		    --scannet_path  ${base_path}/ScanNet_GT/
        --s2d3ds_path ${base_path}/2d3ds_GT/
        --shapenet_voxel_path ${base_path}/ShapeNetCore.v2-voxelized-df/
        --scan2cad_file ${base_path}/ScanCADJoint/scan2cad_objects_v3.json
        --scan2cad_quat_file ${base_path}/ScanCADJoint/scan2cad_quat_v3.json
        --modelpool_file ${base_path}/ScanCADJoint/model_pool_large.json
        --similarity_file ${base_path}/ScanCADJoint/scan-cad_similarity_public_augmented-100_v3.json
        --full_annotation_file ${base_path}/ScanCADJoint/full_annotations.json
        --output_root ${base_path}/ScanCADJoint_out/
        --batch_size 128
		    --num_epochs 1000
        --weight_retr 1.0
        --retrieval_epoch 0
        --offline_sample true
        --negative_sample_strategy mix
        --loss_retrieval tripletMargin
        --miner_type easyhard
        --triplet_margin 0.2
        --circle_margin 0.4
        --circle_gamma 80.0
        --learning_rate 0.001
        --weight_decay 0.0005
        --representation binary
        --rotation_augmentation random
        --rotation_trial_count 12
        --align_to_scan true
        --separation_model_on true
        --skip_connection_sep true
        --skip_connection_com false
        --filter_val_pool true
        --train_scale true
        --train_scale_range 0.5 2
        --train_symmetry true
        --log_frequency 10
        --validate_frequency 100
        --val_max_sample_count 10
        --val_vis_sample_count 5
        --evaluation_frequency 500
        --checkpoint_frequency 100
        --wandb_vis_on true"

#log: increase L2 regulariztion a bit (weight decay increases from 1e-4 to 5e-4)

sh ${base_path}/../3_tool/show_logo.sh 

echo "install some packages"
pip install wandb
pip install numpy==1.20.3 # needed for numpy-quaternion
pip install numpy-quaternion
pip install --no-deps pytorch-metric-learning
python -m pip install open3d==0.9

#use `pip uninstall xxx -y` to uninstall redundant libs to free the space



echo "***************"
echo "JointEmbedding of Scan and CAD [edited by Real2CAD group @ ETHZ&EPFL]"
echo "***************"
echo ""


# run
eval $command


# important runs
#bingke_trioffhardmix_lp_skip2s_binary_v3_p100h_2021-06-08_14-55-53
#yuepan_trioffhardmix_lp_skip2s_binary_v3_p100h_2021-06-09_09-44-24
