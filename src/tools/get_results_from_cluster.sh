#!/bin/sh

# run this script on your own computer instead of the cluster server

cluster_path=login.leonhard.ethz.ch:/cluster/project/infk/courses/3d_vision_21/group_14/1_data/ScanCADJoint_out/real2cad_results
local_path=/media/edward/Seagate/1_data/Real2CAD_results

json_file=results_scannet_00_filter_on_p2l.json

scp ${cluster_path}/${json_file} ${local_path}/${json_file}