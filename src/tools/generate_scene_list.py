import sys
assert sys.version_info >= (3,5)
import os

import numpy as np
np.warnings.filterwarnings('ignore')
import JSONHelper


if __name__ == '__main__':

    json_data_full = []
    json_data_sub = []
    json_data_demo = []
    base_folder = "/media/edward/Seagate/1_data/ScanCADJoint_GT_v3/"

    # list_subfolders_with_paths = [f.path for f in os.scandir(base_folder) if f.is_dir()]
    # print(list_subfolders_with_paths)

    for path, dir_list, file_list in os.walk(base_folder):
        for dir_name in dir_list:
            json_data_full.append(dir_name)
            if dir_name.split("_")[1] == "00":
                json_data_sub.append(dir_name)
        break # only th next layer

    json_file_data_full = "scene_list_scannet_full.json"
    json_file_data_sub = "scene_list_scannet_00.json"

    json_base_folder = "/media/edward/Seagate/1_data/ScanCADJoint"
    json_file_data_full = os.path.join(json_base_folder, json_file_data_full)
    json_file_data_sub = os.path.join(json_base_folder, json_file_data_sub)


    JSONHelper.write(json_file_data_full, json_data_full)

    JSONHelper.write(json_file_data_sub, json_data_sub)

    print("Scene list saved in:", json_file_data_full)



