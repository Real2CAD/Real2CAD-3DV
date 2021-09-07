# Script used to evaluate the results of Real2CAD task for quantitative evaluation
# By Yue Pan from ETH Zurich
# 2021. 7
#
# currently support 2 datasets (scannet [Scan2CAD], 2d3ds)
# reference: https://github.com/skanti/Scan2CAD
#
# You can configure the target dataset and dataset path in the shell file EvaluateResults.sh and run it to call this function

import numpy as np
np.warnings.filterwarnings('ignore')
from numpy.linalg import inv
import json
import os
import quaternion
import SE3
import argparse

# TODO: support 2d3ds dataset
# You may write one on your own, just make sure the metric is the same, also add the Chamfer distance as a metric

# params
parser = argparse.ArgumentParser()
parser.add_argument('--scan_dataset_name', type=str, default="scannet", help="select from 2d3ds and scannet")
parser.add_argument('--scannet_result_json', default="/media/edward/SeagateNew/1_data/ScanCADJoint/real2cad_results/scannet_result.json")
parser.add_argument('--s2d3ds_result_json', default="/media/edward/SeagateNew/1_data/ScanCADJoint/real2cad_results/2d3ds_result.json")
parser.add_argument('--scannet_gt_json', default="/media/edward/SeagateNew/1_data/ScanCADJoint/scannet_full_annotations.json")
parser.add_argument('--s2d3ds_gt_json', default="/media/edward/SeagateNew/1_data/ScanCADJoint/2d3ds_full_annotations.json")
parser.add_argument('--color_lut_json', default="/media/edward/SeagateNew/1_data/ScanCADJoint/category_color_lut.json", help="cad color, cat_id, cat_name look up table")
parser.add_argument('--report_json', default="/media/edward/SeagateNew/1_data/ScanCADJoint_out/real2cad_results/accuracy_report.json")

opt = parser.parse_args()


# helper function to calculate difference between two quaternions
def calc_rotation_diff(q, q00):
    rotation_dot = np.dot(quaternion.as_float_array(q00), quaternion.as_float_array(q))
    rotation_dot_abs = np.abs(rotation_dot)
    try:
        error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    except:
        return 0.0
    error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    error_rotation = np.rad2deg(error_rotation_rad)
    return error_rotation

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

def get_category_code_from_2d3ds(cat_name):
    #["chair", "table", "bookcase", "sofa"]
    category_code_dict = {"chair": "03001627", "table": "04379243",
                      "bookcase": "02871439", "sofa": "04256520"}

    return category_code_dict.get(cat_name, "Other")

def get_scene_result(results_list, scene_name):
    scene_results = None
    for results in results_list:
        if results['id_scan'] == scene_name:
            scene_results = results
            break
    return scene_results

def judge_alignment(T_gt, T_pred, sym, thre_dict):
    t_gt, q_gt, s_gt = SE3.decompose_mat4(T_gt)
    t_pred, q_pred, s_pred = SE3.decompose_mat4(T_pred)

    # error metrics for trans, scale and symmetry-aware rotation
    error_translation = np.linalg.norm(t_pred - t_gt, ord=2) # unit: m
    error_scale = 100.0 * np.abs(np.mean(s_pred / s_gt) - 1) # unit: %

    # consider symmetry property
    if sym == "__SYM_ROTATE_UP_2":
        m = 2
        tmp = [
            calc_rotation_diff(q_pred, q_gt * quaternion.from_rotation_vector([0, (i * 2.0 / m) * np.pi, 0]))
            for i in range(m)]
        error_rotation = np.min(tmp)
    elif sym == "__SYM_ROTATE_UP_4":
        m = 4
        tmp = [
            calc_rotation_diff(q_pred, q_gt * quaternion.from_rotation_vector([0, (i * 2.0 / m) * np.pi, 0]))
            for i in range(m)]
        error_rotation = np.min(tmp)
    elif sym == "__SYM_ROTATE_UP_INF":
        m = 36
        tmp = [
            calc_rotation_diff(q_pred, q_gt * quaternion.from_rotation_vector([0, (i * 2.0 / m) * np.pi, 0]))
            for i in range(m)]
        error_rotation = np.min(tmp)
    else: # none symmetry
        error_rotation = calc_rotation_diff(q_pred, q_gt) # unit: deg

    is_alignment_correct = error_translation <= thre_dict["translation"] and error_rotation <= thre_dict["rotation"] and error_scale <= thre_dict["scale"]

    return is_alignment_correct

def init_benchmark(benchmark_dict):
    benchmark_dict["n_total"] = 0
    benchmark_dict["n_cat_correct"] = 0
    benchmark_dict["n_cad_correct"] = 0
    benchmark_dict["n_align_correct"] = 0
    return benchmark_dict

def cal_accuracy(benchmark_dict):
    if benchmark_dict["n_total"] != 0:
        benchmark_dict["cat_accuracy"] = benchmark_dict["n_cat_correct"] / benchmark_dict["n_total"] * 100.0
        benchmark_dict["cad_accuracy"] = benchmark_dict["n_cad_correct"] / benchmark_dict["n_total"] * 100.0
        benchmark_dict["align_accuracy"] = benchmark_dict["n_align_correct"] / benchmark_dict["n_total"] * 100.0
    else:
        benchmark_dict["cat_accuracy"] = "NaN"
        benchmark_dict["cad_accuracy"] = "NaN"
        benchmark_dict["align_accuracy"] = "NaN"
    return benchmark_dict


if __name__ == "__main__":

    if opt.scan_dataset_name == "scannet":
        result_json = opt.scannet_result_json
        gt_json = opt.scannet_gt_json
    elif opt.scan_dataset_name == "2d3ds":
        result_json = opt.s2d3ds_result_json
        gt_json = opt.s2d3ds_gt_json

    with open(gt_json, 'r') as infile:
        gt = json.load(infile)
    with open(result_json, 'r') as infile:
        results = json.load(infile)
    with open(opt.color_lut_json, 'r') as infile:
        cat_lut = json.load(infile)

    # define thresholds (reference: Scan2CAD)
    thre_dict = {}
    thre_dict["translation"] = 0.2  # unit: m
    thre_dict["rotation"] = 20  # unit: deg
    thre_dict["scale"] = 20  # unit: %

    # counters
    benchmark_total = {}
    benchmark_total = init_benchmark(benchmark_total)
    benchmark_per_scan = {}
    benchmark_per_cat = {}

    # for each scan
    for results_in_scan in results:
        scan_id = results_in_scan["id_scan"]
        gt_in_scan = get_scene_result(gt, scan_id)
        gt_scan_trs = gt_in_scan["trs"]
        Tws_gt = SE3.compose_mat4(gt_scan_trs["translation"], gt_scan_trs["rotation"], gt_scan_trs["scale"])  # tran from scan to world
        gt_models = gt_in_scan["aligned_models"]

        benchmark_per_scan[scan_id] = {}
        benchmark_per_scan[scan_id] = init_benchmark(benchmark_per_scan[scan_id])

        # for each recorded scan object
        for object_idx, pred_model in enumerate(results_in_scan["aligned_models"]):
            scan_obj_id = pred_model["scan_obj_id"]
            obj_idx_in_scan = int(scan_obj_id.split("_")[3]) # TODO: find the way that can work for 2d3ds dataset
            catid_cad_pred = pred_model["catid_cad"]
            id_cad_pred = pred_model["id_cad"]
            Tsc_pred = np.asarray(pred_model["trs_mat"])

            # find the corresponding gt
            gt_model = gt_models[obj_idx_in_scan]
            gt_sym = gt_model["sym"]
            catid_cad_gt = gt_model["catid_cad"]
            cat_name = cat_lut[catid_cad_gt]["category"]
            id_cad_gt = gt_model["id_cad"]
            gt_model_trs = gt_model["trs"]
            #Twc_gt = SE3.compose_mat4(gt_model_trs["translation"], gt_model_trs["rotation"], gt_model_trs["scale"], -np.array(gt_model["center"]))  # tran from scan to world (what dose this center mean)
            Twc_gt = SE3.compose_mat4(gt_model_trs["translation"], gt_model_trs["rotation"], gt_model_trs["scale"])
            Tsc_gt = inv(Tws_gt) @ Twc_gt

            is_same_class = catid_cad_pred == catid_cad_gt
            is_same_cad = id_cad_pred == id_cad_gt

            if cat_name not in benchmark_per_cat.keys():
                benchmark_per_cat[cat_name] = {}
                benchmark_per_cat[cat_name] = init_benchmark(benchmark_per_cat[cat_name])

            benchmark_total["n_total"] += 1
            benchmark_per_scan[scan_id]["n_total"] += 1
            benchmark_per_cat[cat_name]["n_total"] += 1

            if is_same_class:  # <-- proceed only if predicted-model and gt-model are in same class
                benchmark_total["n_cat_correct"] += 1
                benchmark_per_scan[scan_id]["n_cat_correct"] += 1
                benchmark_per_cat[cat_name]["n_cat_correct"] += 1

                if is_same_cad:
                    benchmark_total["n_cad_correct"] += 1
                    benchmark_per_scan[scan_id]["n_cad_correct"] += 1
                    benchmark_per_cat[cat_name]["n_cad_correct"] += 1

                if judge_alignment(Tsc_gt, Tsc_pred, gt_sym, thre_dict):
                    benchmark_total["n_align_correct"] += 1
                    benchmark_per_scan[scan_id]["n_align_correct"] += 1
                    benchmark_per_cat[cat_name]["n_align_correct"] += 1

    # calculate accuracy (unit: %)
    benchmark_total = cal_accuracy(benchmark_total)
    for cat in benchmark_per_cat.keys():
        benchmark_per_cat[cat] = cal_accuracy(benchmark_per_cat[cat])
    for scan in benchmark_per_scan.keys():
        benchmark_per_scan[scan] = cal_accuracy(benchmark_per_scan[scan])

    # print and output the report
    print("*********** instance level mean accuracy ***********")
    print("alignment accuracy(%): {:>03.3f} \t CAD retrieval accuracy(%): {:>03.3f} \t CAD category accuracy(%): {:>03.3f} \n".format(
            benchmark_total["align_accuracy"], benchmark_total["cad_accuracy"], benchmark_total["cat_accuracy"]))

    print("*********** per class accuracy ***********")
    for cat_name in benchmark_per_cat.keys():
        print("category-name: {:>20s} \t alignment accuracy(%): {:>03.3f} \t CAD retrieval accuracy(%): {:>03.3f} \t CAD category accuracy(%): {:>03.3f} \n".format(
            cat_name, benchmark_per_cat[cat_name]["align_accuracy"], benchmark_per_cat[cat_name]["cad_accuracy"], benchmark_per_cat[cat_name]["cat_accuracy"]))

    benchmark_json_dict = {"total": benchmark_total, "per_class": benchmark_per_cat, "per_scan": benchmark_per_scan}
    with open(opt.report_json, 'w') as outfile:  # overwrite existing
        json.dump(benchmark_json_dict, outfile)














