#!/bin/bash
base_path=/cluster/project/infk/courses/3d_vision_21/group_14/1_data

# specify the test name here
# user yuepan
resume_name=trioffhardmix_lp_skip2s_binary_v3_p100h
resume_timestamp=2021-06-09_09-44-24
resume_checkpoint=11000
#skip_connection_com false

# user yuepan
# resume_name=wobg_trioffhard_class_binary_v3_p100
# resume_timestamp=2021-06-07_00-40-01
# resume_checkpoint=8000
#skip_connection_com true

# user yuayue
#resume_name=without_separation
#resume_timestamp=2021-07-09_12-25-48
#resume_checkpoint=24000

model_name=${resume_name}
# update the checkpoint model system

#results_name=results_scannet_appearance_filter_on_p2p_fine
results_name=results_scannet_appearance_filter_on_p2p_scale_fine
#results_name=results_scannet_key_filter_on_coarse
#results_name=results_scannet_key_filter_off

# If not specify, then the demo list would be used
scan_pool_name=dummy
#--scenelist_file ${base_path}/ScanCADJoint/scene_list_scannet_00.json
##--scenelist_file ${base_path}/ScanCADJoint/scene_list_scannet_interested.json
# --scenelist_file ${base_path}/ScanCADJoint/scene_list_2d3ds_demo.json

model_pool_name=appearance_scannet
# --modelpool_file ${base_path}/ScanCADJoint/model_pool_small_list_v3.json
# --modelpool_file ${base_path}/ScanCADJoint/model_pool_2d3ds.json
# --modelpool_file ${base_path}/ScanCADJoint/model_pool_appearance_scannet.json

#--similarity_path ${base_path}/ScanCADJoint/scan-cad_similarity_public_augmented-100_v3.json  # ScanNet
#--similarity_path ${base_path}/ScanCADJoint/similarity_Area_3_office_3_5_7.json # 2D3DS

command="CUDA_VISIBLE_DEVICES=0
		python3 ./test.py
        --name scannet_test
        --resume yuepan_${resume_name}_${resume_timestamp}/${resume_checkpoint}
        --embed_mode false
        --in_the_wild_mode false
        --scan_dataset_name scannet
		--scannet_path  ${base_path}/ScanNet_GT/
        --shapenet_voxel_path ${base_path}/ShapeNetCore.v2-voxelized-df/
        --shapenet_pc_path ${base_path}/ShapeNetCore.v2-pcd/
        --scan2cad_file ${base_path}/ScanCADJoint/scan2cad_objects_v3.json
        --similarity_file ${base_path}/ScanCADJoint/scan-cad_similarity_public_augmented-100_v3.json
        --modelpool_file ${base_path}/ScanCADJoint/model_pool_appearance_scannet.json
        --scenelist_file ${base_path}/ScanCADJoint/scene_list_scannet_interested.json
        --scan2cad_quat_file ${base_path}/ScanCADJoint/scan2cad_quat_v3.json
        --cad_apperance_file ${base_path}/ScanCADJoint/scannet_cad_appearances.json
        --output_root ${base_path}/ScanCADJoint_out/
        --cad_embedding_path ${base_path}/ScanCADJoint_out/cad_embeddings/cad_embedding_${model_name}_${model_pool_name}.pt
        --scan_embedding_path ${base_path}/ScanCADJoint_out/scan_embeddings/scan_embedding_${model_name}_${scan_pool_name}.pt
        --real2cad_result_path ${base_path}/ScanCADJoint_out/real2cad_results/${results_name}.json
        --tsne_img_path ${base_path}/ScanCADJoint_out/tsne_results/${model_name}
        --batch_size 128
        --representation binary
        --separation_model_on true
        --skip_connection_sep true
        --skip_connection_com false
        --filter_val_pool true
        --rotation_trial_count 12
        --init_scale_method naive
        --only_coarse_reg false
        --only_rot_z false
        --icp_reg_mode p2p
        --icp_dist_thre 5.0
        --icp_with_scale_on true
        --val_vis_sample_count 5
        --wandb_vis_on true"


#sh ${base_path}/../3_tool/show_logo.sh
echo "***************"
echo "JointEmbedding of Scan and CAD [edited by Real2CAD group @ ETHZ\EPFL]"
echo "***************"
echo ""


# run
eval $command