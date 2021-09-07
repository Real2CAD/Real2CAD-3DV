import JSONHelper
import os

if __name__ == '__main__':

    # training set
    # train_set_json = "/cluster/project/infk/courses/3d_vision_21/group_14/1_data/Scan2CAD-training-data/trainset_pn.json"
    # train_set_json_filtered = "/cluster/project/infk/courses/3d_vision_21/group_14/Scan2CAD-training-data/trainset_pn_filtered.json"

    # validation set
    # train_set_json = "/cluster/project/infk/courses/3d_vision_21/group_14/1_data/Scan2CAD-training-data/validationset_pn.json"
    # train_set_json_filtered = "/cluster/project/infk/courses/3d_vision_21/group_14/1_data/Scan2CAD-training-data/validationset_pn_filtered.json"
    
    # visualization set
    train_set_json = "/cluster/project/infk/courses/3d_vision_21/group_14/1_data/Scan2CAD-training-data/visualizationset.json"
    train_set_json_filtered = "/cluster/project/infk/courses/3d_vision_21/group_14/1_data/Scan2CAD-training-data/visualizationset_filtered.json"


    valid_sample_count = 0
    invalid_sample_count = 0
    count =0
    json_data= []

    for r in JSONHelper.read(train_set_json):

        count +=1

        if count%100 ==0:
            print(valid_sample_count, ":", invalid_sample_count)

        # load data
        filename_center = r["filename_vox_center"]
        filename_heatmap = r["filename_vox_heatmap"]

        if os.path.isfile(filename_heatmap) and os.path.isfile(filename_center):
            json_data.append(r)
            valid_sample_count+=1
        else:
            invalid_sample_count+=1


    print("Valid sample percentage:", 100.0*valid_sample_count/count, " %")

    JSONHelper.write(train_set_json_filtered, json_data)
    print("Training json-file (needed from conv) saved in:", train_set_json_filtered)

