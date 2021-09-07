import sys
assert sys.version_info >= (3,5)
import os
import json
import numpy as np
np.warnings.filterwarnings('ignore')


if __name__ == '__main__':

    model_pool_json_data = {}

    data_base_folder = "/media/edward/SeagateNew/1_data/ShapeNetCore.v2-voxelized-df"
    input_json_folder = "/media/edward/SeagateNew/1_data/ScanCADJoint"
    output_json_folder = "/media/edward/SeagateNew/1_data/ScanCADJoint"

    # #category_list = ["03001627", "04379243", "02747177", "02818832", "02871439", "02933112", "04256520", "02808440", "03636649", "02773838", "03938244", "03337140"]
    # category_list_misc = ["03211117", "03928116", "04554684", "03761084", "03642806", "03207941", "04330267", "04004475", "02828884", "03467517", "03593526", "02876657", "02880940", "03085013", "03325088", "02954340", "03691459", "03046257", "02801938", "03991062", "04401088"]
    #
    # #"Chair", "Table", "Trash bin", "Bed", "Bookshelf", "Cabinet", "Sofa", ["Bathtub", "Lamp", "Bag", "Pillow", "File"] (Others)
    #
    # # Method 1: Generate a large model pool from all the available cad models in the interested categories
    # for category in category_list_misc:
    #     sub_folder = os.path.join(data_base_folder, category)
    #     g = os.walk(sub_folder)
    #
    #     cat_list = []
    #     item_count = 0
    #     for path,dir_list,file_list in g:
    #         for file_name in file_list:
    #             file_name_base = file_name.split(".")[0]
    #             file_format = file_name.split(".")[1]
    #             if file_format == "df":
    #                 file_name_with_cat = os.path.join(category, file_name_base)
    #                 cat_list.append(file_name_with_cat)
    #                 #print(file_name_with_cat)
    #                 item_count += 1
    #
    #     model_pool_json_data[category] = cat_list
    #     print("There are [", str(item_count), "] items in category [", category, "]")
    #
    # model_pool_json_file_name = "model_pool_misc.json"
    # model_pool_json_file_path = os.path.join(output_json_folder, model_pool_json_file_name)
    # with open(model_pool_json_file_path, 'w') as outfile:  # overwrite existing
    #     json.dump(model_pool_json_data, outfile)
    #
    # print("Model pool saved in:", model_pool_json_file_path)

    # Method 2: Generate a smaller model pool by sampling in the larger model pool





    # Method 3: Generate a large model pool using all the cad models appears in the scan2cad one-to-one mapping json file



    # Method 4: Generate a model pool using all the cad models appears in the cad_apperance json file
    cad_appearance_json_path = os.path.join(input_json_folder, "cad_appearances.json")

    with open(cad_appearance_json_path, 'r') as infile:
        cad_appearance_dict = json.load(infile)
    scan_list = list(cad_appearance_dict.keys())
    cad_list = []
    for scan_id, cad_dict_of_scan in cad_appearance_dict.items():
        cad_list_of_scan = list(cad_dict_of_scan.keys())
        cad_list = cad_list + cad_list_of_scan

    cad_list = list(set(cad_list))

    for cad_id in cad_list:
        parts = cad_id.split("_")
        cat = parts[0]
        cad = parts[1]
        new_cad_id = cat+"/"+cad
        if cat not in model_pool_json_data.keys():
            model_pool_json_data[cat] = [new_cad_id]
        else:
            model_pool_json_data[cat].append(new_cad_id)

    model_pool_json_file_name = "model_pool_appearance.json"
    model_pool_json_file_path = os.path.join(output_json_folder, model_pool_json_file_name)
    with open(model_pool_json_file_path, 'w') as outfile:  # overwrite existing
        json.dump(model_pool_json_data, outfile)

    print("Model pool saved in:", model_pool_json_file_path)