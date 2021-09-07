# Script used to visualize the results of Real2CAD task for qualitative evaluation
# By Yue Pan from ETH Zurich
# 2021. 7
#
# currently support 2 datasets (scannet [Scan2CAD], 2d3ds)
# reference: https://github.com/skanti/Scan2CAD
#
# You can configure the target dataset and dataset path in the shell file VisResults.sh and run it to call this function

import sys
assert sys.version_info >= (3, 5)

import numpy as np
from numpy.linalg import inv
import json
import os
import quaternion
import open3d as o3d
import SE3
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def make_M_from_tqs(t, q, s):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    Rot = quaternion.as_rotation_matrix(q)
    R[0:3, 0:3] = Rot
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M, Rot

def calc_Mbbox(model):
    trs_obj = model["trs"]
    # object detection gt
    bbox_obj = np.asarray(model["bbox"], dtype=np.float64) # important
    center_obj = np.asarray(model["center"], dtype=np.float64) # important

    trans_obj = np.asarray(trs_obj["translation"], dtype=np.float64)
    rot_obj = np.asarray(trs_obj["rotation"], dtype=np.float64)
    q_obj = np.quaternion(rot_obj[0], rot_obj[1], rot_obj[2], rot_obj[3])
    scale_obj = np.asarray(trs_obj["scale"], dtype=np.float64)

    tcenter1 = np.eye(4)
    tcenter1[0:3, 3] = center_obj
    trans1 = np.eye(4)
    trans1[0:3, 3] = trans_obj
    rot1 = np.eye(4)
    rot1[0:3, 0:3] = quaternion.as_rotation_matrix(q_obj)
    scale1 = np.eye(4)
    scale1[0:3, 0:3] = np.diag(scale_obj)
    bbox1 = np.eye(4)
    bbox1[0:3, 0:3] = np.diag(bbox_obj)

    M = trans1.dot(rot1).dot(scale1).dot(tcenter1).dot(bbox1)
    return M
    # in world space
    # bx 0  0  cx
    # 0  by 0  cy
    # 0  0  bz cz
    # 0  0  0   1

def vis_read2cad_results(vis, scene_result, scan_base_folder, model_base_path, color_lut,
                         scan_dataset: str = "scannet", is_gt_file: bool = False,
                         bbx_vis_on: bool = True, scan_vis_on: bool = True, scan_vis_gray: bool = True):

    id_scan = scene_result["id_scan"]

    print("Visualize scene [", id_scan, "]")
    print("-------------------------------------------")

    basedir = os.path.join(scan_base_folder, id_scan)
    id_room = id_scan.split("/")[-1]

    if scan_dataset == "scannet":
        scan_file = os.path.join(basedir, id_room) + "_scan.ply" # ScanNet
    elif scan_dataset == "2d3ds":
        scan_file = os.path.join(basedir, id_room) + "_mesh.ply" # 2D3DS

    #### scan (mesh)
    cur_scan_mesh = o3d.io.read_triangle_mesh(scan_file) # global coordinate system
    cur_scan_mesh.compute_vertex_normals()
    if scan_vis_gray:
        cur_scan_mesh.paint_uniform_color([0.8, 0.8, 0.8])
    # cur_scan_pcd = cur_scan_mesh.sample_points_uniformly(number_of_points=20000)

    scene_vis_contents = []
    if scan_vis_on:
        scene_vis_contents.append(cur_scan_mesh)
        # vis.add_geometry(cur_scan_pcd)
        vis.add_geometry(cur_scan_mesh)

    if is_gt_file:
        trs_ws = scene_result["trs"]
        Tws, Rws = make_M_from_tqs(trs_ws["translation"], trs_ws["rotation"], trs_ws["scale"]) # tran from scan to world
        #print("Tws of the scan [", id_scan, " ]:\n", Tws)

    # for each aligned model
    for object_idx, model in enumerate(scene_result["aligned_models"]):
        id_cad = model["id_cad"]
        catid_cad = model["catid_cad"]

        if catid_cad not in color_lut:
            print("Certain category is not defined, skip the sample")
            continue

        # load color
        color_cad = color_lut[catid_cad]["color"]
        color_cad = (np.array(color_cad) / 255).tolist()

        # load transformation
        tran_cad2scan = np.eye(4)

        if is_gt_file:  # gt file
            # transformation and bbx geometry info
            trs = model["trs"]
            # Twc = SE3.compose_mat4(trs["translation"], trs["rotation"], trs["scale"], -np.array(model["center"])) # tran from cad to world
            Twc, Rwc = make_M_from_tqs(trs["translation"], trs["rotation"], trs["scale"])
            #print("Twc of object [", catid_cad, "/" , id_cad, " ]:\n", Twc)

            T_mirror = np.eye(4)
            T_mirror[0, 0] = -1
            T_mirror[1, 1] = -1

            if scan_dataset == "2d3ds":
                tran_cad2scan = inv(Tws) @ Twc @ inv(T_mirror) # TODO: still have some problem
            elif scan_dataset == "scannet":
                tran_cad2scan = inv(Tws) @ Twc

        else: # result file
            tran_cad2scan = np.asarray(model["trs_mat"])

        # load bbx
        if is_gt_file:
            Twbbox = calc_Mbbox(model)  # from [-1, 1] bbx to world space
            Tsbbox = inv(Tws).dot(Twbbox)  # from [-1, 1] bbx to scan space
            bbox_o = o3d.geometry.OrientedBoundingBox(
                center=(0, 0, 0),
                R=np.eye(3),
                extent=(2, 2, 2),
            )
            bbox_o = bbox_o.rotate(Tsbbox[0:3, 0:3])
            bbox_o = bbox_o.translate(Tsbbox[0:3, 3])
            if bbx_vis_on:
                scene_vis_contents.append(bbox_o)
                vis.add_geometry(bbox_o)

        # load cad model
        cad_folder = os.path.join(model_base_path, catid_cad, id_cad, "models")
        cad_folder_temp = os.path.join(cad_folder, "temp/")
        cad_path = os.path.join(cad_folder, "model_normalized.obj")

        # we don't want to load the texture file (since there's some incompatibility issue within it)
        os.system('mkdir ' + cad_folder_temp)
        os.system('cp ' + cad_path + ' ' + cad_folder_temp)

        cad_path_new = os.path.join(cad_folder_temp, "model_normalized.obj")
        cad_mesh = o3d.io.read_triangle_mesh(cad_path_new)
        cad_mesh.transform(tran_cad2scan)
        cad_mesh.paint_uniform_color(color_cad)
        cad_mesh.compute_vertex_normals()

        scene_vis_contents.append(cad_mesh)
        vis.add_geometry(cad_mesh)

        os.system('rm -r ' + cad_folder_temp)

    # Visualization
    #o3d.visualization.draw_geometries(scene_vis_contents)

    return scene_vis_contents

def get_scene_result(results_list, scene_name):
    scene_results = None
    for results in results_list:
        if results['id_scan'] == scene_name:
            scene_results = results
            break
    return scene_results


# params
parser = argparse.ArgumentParser()
parser.add_argument('--scannet_base_folder', default="/media/edward/SeagateNew/1_data/ScanNet_GT/", help="base folder of the scannet scan dataset")
parser.add_argument('--s2d3ds_base_folder', default="/media/edward/SeagateNew/1_data/2d3ds/2d3ds_GT/", help="base folder of the 2d3ds scan dataset")
parser.add_argument('--cad_base_path', default="/media/edward/SeagateNew/1_data/ShapeNetCore.v2/", help="base folder of the cad models (shapenet dataset)")
parser.add_argument('--color_lut_json', default="/media/edward/SeagateNew/1_data/ScanCADJoint/category_color_lut.json", help="cad color look up table")
parser.add_argument('--scannet_scene_list', default="/media/edward/SeagateNew/1_data/ScanCADJoint/scene_list_scannet_part.json")
parser.add_argument('--s2d3ds_scene_list', default="/media/edward/SeagateNew/1_data/ScanCADJoint/scene_list_2d3ds_part.json")
parser.add_argument('--scannet_gt_json', default="/media/edward/SeagateNew/1_data/ScanCADJoint/scannet_full_annotations.json")
parser.add_argument('--s2d3ds_gt_json', default="/media/edward/SeagateNew/1_data/ScanCADJoint/2d3ds_full_annotations.json")
parser.add_argument('--result_json_1', default="/media/edward/SeagateNew/1_data/ScanCADJoint_out/real2cad_results/results_scannet_00_filter_on_p2l.json", help="predicted real2cad alignments")
parser.add_argument('--result_json_2', default="/media/edward/SeagateNew/1_data/ScanCADJoint_out/real2cad_results/results_scannet_00_filter_on_p2l.json", help="predicted real2cad alignments 2 for comparison")
parser.add_argument('--scan_dataset_name', type=str, default="scannet", help="select from 2d3ds and scannet")
parser.add_argument('--gt_vis_on', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--reg_vis_1_on', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--reg_vis_2_on', type=str2bool, nargs='?', const=True, default=True)
opt = parser.parse_args()

if __name__ == '__main__':

    if opt.scan_dataset_name == "scannet":
        scan_base_folder = opt.scannet_base_folder
        gt_json = opt.scannet_gt_json
        scene_list = opt.scannet_scene_list

    elif opt.scan_dataset_name == "2d3ds":
        scan_base_folder = opt.s2d3ds_base_folder
        gt_json = opt.s2d3ds_gt_json
        scene_list = opt.s2d3ds_scene_list

    with open(gt_json, 'r') as infile:
        gt = json.load(infile)
    with open(opt.result_json_1, 'r') as infile:
        results_1 = json.load(infile)
    with open(opt.result_json_2, 'r') as infile:
        results_2 = json.load(infile)
    with open(opt.color_lut_json, 'r') as infile:
        color_lut = json.load(infile)
    with open(scene_list, 'r') as infile:
        scenes_to_vis = json.load(infile)
    #scenes_to_vis = ["scene0000_00", "scene0001_00", "scene0002_00", "scene0003_00", "scene0004_00", "scene0005_00", "scene0006_00", "scene0007_00", "scene0008_00", "scene0009_00"]
    #scenes_to_vis = ["scene0030_00"]
    #scenes_to_vis = ["scene0011_00"]


    for scene_vis in scenes_to_vis:
        if opt.gt_vis_on:
            gt_scene = get_scene_result(gt, scene_vis)
            gt_vis = o3d.visualization.Visualizer()
            gt_vis.create_window(scene_vis + ": ground truth alignment")
            gt_vis_contents = vis_read2cad_results(gt_vis, gt_scene, scan_base_folder, opt.cad_base_path, color_lut,
                                                   opt.scan_dataset_name, is_gt_file=True,
                                                   bbx_vis_on=False, scan_vis_on=True, scan_vis_gray=True)
            gt_vis.run()
            gt_vis.destroy_window()

        if opt.reg_vis_1_on:
            result1_scene = get_scene_result(results_1, scene_vis)
            results_vis1 = o3d.visualization.Visualizer()
            results_vis1.create_window(scene_vis + ": Real2CAD predicted alignment 1 (coarse)")
            results_vis_content = vis_read2cad_results(results_vis1, result1_scene, scan_base_folder, opt.cad_base_path, color_lut,
                                                       opt.scan_dataset_name, is_gt_file=False,
                                                       bbx_vis_on=False, scan_vis_on=True, scan_vis_gray=True)
            results_vis1.run()
            results_vis1.destroy_window()

        if opt.reg_vis_2_on:
            result2_scene = get_scene_result(results_2, scene_vis)
            results_vis2 = o3d.visualization.Visualizer()
            results_vis2.create_window(scene_vis + ": Real2CAD predicted alignment 2 (fine)")
            results_vis_content2 = vis_read2cad_results(results_vis2, result2_scene, scan_base_folder, opt.cad_base_path, color_lut,
                                                        opt.scan_dataset_name, is_gt_file=False,
                                                        bbx_vis_on=False, scan_vis_on=True, scan_vis_gray=True)
            results_vis2.run()
            results_vis2.destroy_window()







