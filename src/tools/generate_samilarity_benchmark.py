# Useful function to generate the retrieval benchmark

import sys
assert sys.version_info >= (3,5)
import os
print(os.getcwd())

import numpy as np
np.warnings.filterwarnings('ignore')
import copy
import random
import JSONHelper


if __name__ == '__main__':

    json_in = "/media/edward/SeagateNew/1_data/2d3ds/Annotation/similarity_Area_3_office_3_5_7_new.json"
    content = JSONHelper.read(json_in)
    content = content["samples"]

    json_data = {"samples": []}

    json_model_pool = "/media/edward/SeagateNew/1_data/ScanCADJoint/model_pool_large.json"
    model_pool = JSONHelper.read(json_model_pool)
    model_pool_cat_list = model_pool.keys()
    print(model_pool_cat_list)

    # the "ranked" list should be the subset of the "pool"

    model_pool_expand_size = 100

    for sample in content:
        new_sample = copy.deepcopy(sample) # you need to deep copy the list

        # sample_scan = sample["reference"]["name"]
        # scan_cat = sample_scan.split("_")[4]

        sample_pool = sample["ranked"]
        #print(len(sample_pool))

        for i in range(model_pool_expand_size):
            cur_cat = random.sample(model_pool_cat_list, 1)[0]
            cur_model_list = model_pool[cur_cat]
            cur_cad = random.sample(cur_model_list, 1)[0]
            cur_cad_str = "/cad/" + cur_cad
            cur_cad_dict = {"name": cur_cad_str}
            sample_pool.append(cur_cad_dict)

        #print(len(sample_pool))
        new_sample["pool"] = sample_pool

        #print(len(new_sample["ranked"]))

        json_data["samples"].append(new_sample)

    json_out = "/media/edward/SeagateNew/1_data/2d3ds/Annotation/similarity_Area_3_office_3_5_7.json"

    JSONHelper.write(json_out, json_data)
    print("Similarity json saved in:", json_out)
