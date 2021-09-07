import os
from typing import List, Tuple
from torch.utils.data import Dataset

import data
import numpy as np

class InferenceDataset(Dataset):
    def __init__(self, data_root: str, file_list: List[str], file_extension: str, data_type: str, transformation=None, 
                 scan_dataset: str = "scannet", cad_dataset: str = "shapenet", input_only_mask: bool = False,
                 mask_folder_extension: str = "_mask_voxel", object_folder_extension: str = "_object_voxel"):
        super().__init__()

        self.data_root = data_root
        self.file_list = file_list # scannet object list
        self.transformation = transformation
        self.file_extension = file_extension # sdf
        self.data_type = data_type
        self.scan_dataset = scan_dataset
        self.cad_dataset = cad_dataset
        self.input_only_mask = input_only_mask
        self.mask_folder_extension = mask_folder_extension
        self.object_folder_extension = object_folder_extension

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int) -> Tuple[str, np.array]:
        name = self.file_list[index]
        element_path = ""
        rotation_ranking_on = False
        # edit here

        parts = name.split("_")
        if self.data_type == "scan": # scannet

            if self.scan_dataset == "scannet":
                # example scan name: scene0238_01__8_03211117_4812245f2f9fa2c953ed9ce120377769
                # or scene0238_01__8
                scan_name = parts[0]+"_"+parts[1]
                object_name = scan_name+"__"+parts[3]
                if self.input_only_mask:
                    element_path = os.path.join(self.data_root, scan_name, scan_name + self.mask_folder_extension, object_name + self.file_extension)
                else:
                    element_path = os.path.join(self.data_root, scan_name, scan_name + self.object_folder_extension, object_name + self.file_extension)
            
            elif self.scan_dataset == "2d3ds":
                # example scan name: Area_3_office_2_chair_38_3_03001627_94e289c89059106bd8f74b0004a598cd 
                # or Area_3_office_2_chair_38_3
                area_name = parts[0]+"_"+parts[1]
                room_name = parts[2]+"_"+parts[3]
                obj_global_name =  parts[4] + "_" + parts[5] + "_" + parts[6]
                object_name = area_name + "_" + room_name + "_" + obj_global_name
                if self.input_only_mask:              
                    element_path = os.path.join(self.data_root, area_name, room_name, room_name + self.mask_folder_extension, object_name + self.file_extension)
                else:
                    element_path = os.path.join(self.data_root, area_name, room_name, room_name + self.object_folder_extension, object_name + self.file_extension)
        
        elif self.data_type == "cad": # shapenet
            
            if self.cad_dataset == "shapenet":
                # example cad name: 03211117/e2fab6391d388d1971863f1e1f0d28a4
                if len(parts) == 1:
                    element_path = os.path.join(self.data_root, name + self.file_extension)
                else: 
                # example cad name with rotation: 03211117/e2fab6391d388d1971863f1e1f0d28a4_rot
                    element_path = os.path.join(self.data_root, parts[0] + self.file_extension)
                    rotation_ranking_on = True
                    rot = int(parts[1])
        
        if self.input_only_mask:
            element = data.load_mask(element_path).tdf
        else:
            element = self.transformation(data.load_raw_df(element_path)).tdf

        # apply rotation (only for the shapenet cad)
        if rotation_ranking_on:
            element = data.rotation_augmentation_interpolation_v3(element, "dummy", aug_rotation_z = rot)     
       
        return name, element

# deprecated
    @staticmethod
    def _load(path, input_object_mask):
        if input_object_mask:
            sample = data.load_mask(path)
        else:
            sample = data.load_raw_df(path)
             
        return sample

