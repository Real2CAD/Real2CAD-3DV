import json
import math
import os
import random
from collections import defaultdict
import numpy as np
from numpy.linalg import inv
import quaternion

from torch.utils.data import Dataset
from typing import List, Tuple, Any, Dict
# import src.data as data
import data


class TrainingDataset(Dataset):
    def __init__(self, scan2cad_file: str, scan_base_path: str, cad_base_path: str, quat_file: str, full_annotation_file: str,
                 model_pool_file: str = None, scan_dataset_name: str = "scannet", splits=None, input_only_mask: bool = False,
                 transformation=None, align_to_scan=False, rotation_aug=None, flip_aug=False, jitter_aug=False,
                 rotation_aug_count: int = 12, add_negatives=False, negative_sample_strategy: str = "mix",
                 scan_obj_format: str = ".sdf", cad_format: str = ".df", scan_mask_format: str = ".mask",
                 scan2cad_key_name: str = "scan2cad_objects", mask_folder_extension: str = "_mask_voxel",
                 object_folder_extension: str = "_object_voxel") -> None:
        
        super().__init__()

        if splits is None:
            splits = ["train", "validation", "test"]
       
        self.scan_dataset_name = scan_dataset_name
        self.scan_base_path = scan_base_path
        self.cad_base_path = cad_base_path

        self.splits = splits
        self.input_only_mask = input_only_mask

        self.dataset_file = scan2cad_file
        self.quat_file = quat_file
        self.model_pool = self.load_model_pool(model_pool_file)
        
        self.align_to_scan = align_to_scan
        self.rotation_interval = 360 / rotation_aug_count #(deg)
        self.rotation_aug_count = rotation_aug_count

        # augmentation
        self.rotation_augmentation = rotation_aug
        self.flip_augmentation = flip_aug
        self.jitter_augmentation = jitter_aug
        
        self.scan_obj_format = scan_obj_format
        self.cad_format = cad_format
        self.scan_mask_format = scan_mask_format
        self.mask_folder_extension = mask_folder_extension
        self.object_folder_extension = object_folder_extension
       
        if transformation is None:
            transformation = data.truncation_normalization_transform

        self.transformation = transformation

        self.fixed_rotation_augmentation = 0

        self.samples, self.quats = self.load_splitset_from_json(self.dataset_file, self.quat_file, scan2cad_key_name)

        self.formatted_full_annotation = self.load_full_annotation(full_annotation_file)

        self.has_negatives = add_negatives

        if negative_sample_strategy == "other_cat":
            self.negatives = self.add_negatives(self.samples, self.model_pool)
        elif negative_sample_strategy == "same_cat":
            self.negatives = self.add_negatives_same_category(self.samples, self.model_pool)
        else: # mix
            self.negatives = self.add_negatives_mix(self.samples, self.model_pool)
    
    # TODO: unify to a single function for negative generation
    @staticmethod
    def add_negatives(samples: List[str], model_pool) -> List[str]:
        # TODO: update the version for 2d3ds
        if model_pool is None:
            per_category = defaultdict(list)
            for sample in samples:
                category = sample.split("_")[4]
                model_id = sample.split("_")[5]
                shape_object = category+"/"+model_id
                per_category[category].append(shape_object)
        else:
            per_category = model_pool
            # print("Sampling negative cad models from the specified model pool")

        negatives = []
        for sample in samples:
            category = sample.split("_")[4]
            # select a random model from a random category (except for the current one)
            neg_categories = list(per_category.keys())
            if category in neg_categories:
                neg_categories.remove(category)
            neg_category = np.random.choice(neg_categories)
            neg_cad = np.random.choice(per_category[neg_category])        
            negatives.append(neg_cad)

        # print("first positive sample: ", samples[0].split("_")[4]+"/"+samples[0].split("_")[5])
        # print("first negative sample: ", negatives[0])

        return negatives
    
    # need to be static method
    @staticmethod
    def add_negatives_same_category(samples: List[str], model_pool) -> List[str]:

        # TODO: update the version for 2d3ds
        if model_pool is None:
            per_category = defaultdict(list)
            for sample in samples:
                category = sample.split("_")[4]
                model_id = sample.split("_")[5]
                shape_object = category+"/"+model_id
                per_category[category].append(shape_object)
        else:
            per_category = model_pool
            # print("Sampling negative cad models from the specified model pool")

        negatives = []
        for sample in samples:
            category = sample.split("_")[4]
            gt_model_id = sample.split("_")[5]
            gt_shape_object = category+"/"+gt_model_id
            if gt_shape_object in per_category[category]:
                per_category[category].remove(gt_shape_object) # make sure there's gt_model_id inside the pool
            # select a random model from the same category
            neg_cad = np.random.choice(per_category[category])        
            negatives.append(neg_cad)
            per_category[category].append(gt_shape_object) # it's removed, but we still want it for other samples, so that we append it again

        # print("first positive sample: ", samples[0].split("_")[4]+"/"+samples[0].split("_")[5])
        # print("first negative sample: ", negatives[0])

        return negatives

    @staticmethod
    def add_negatives_mix(samples: List[str], model_pool) -> List[str]:
        # TODO: update the version for 2d3ds

        if model_pool is None:
            per_category = defaultdict(list)
            for sample in samples:
                category = sample.split("_")[4]
                model_id = sample.split("_")[5]
                shape_object = category+"/"+model_id
                per_category[category].append(shape_object)
        else:
            per_category = model_pool
            # print("Sampling negative cad models from the specified model pool")

        negatives = []
        for idx, sample in enumerate(samples):
            category = sample.split("_")[4]
            gt_model_id = sample.split("_")[5]
            gt_shape_object = category+"/"+gt_model_id
            if idx % 3 == 0: # sample negative cad model from a different category
                neg_categories = list(per_category.keys())
                if category in neg_categories:
                    neg_categories.remove(category)
                neg_category = np.random.choice(neg_categories)
                neg_cad = np.random.choice(per_category[neg_category])        
                negatives.append(neg_cad)
            elif idx % 3 == 1: # sample negative cad model from the same category
                if gt_shape_object in per_category[category]:
                    per_category[category].remove(gt_shape_object) # make sure there's gt_model_id inside the pool
                neg_cad = np.random.choice(per_category[category])
                negatives.append(neg_cad) 
                per_category[category].append(gt_shape_object)  
            else:  # sample the cad model, but we will apply a different rotation
                negatives.append(gt_shape_object)
        
        # print("first positive sample: ", samples[0].split("_")[4]+"/"+samples[0].split("_")[5])
        # print("first negative sample: ", negatives[0])

        return negatives


    def regenerate_negatives(self) -> None:
            self.negatives = self.add_negatives(self.samples, self.model_pool)

    def regenerate_negatives_same_category(self) -> None:
            self.negatives = self.add_negatives_same_category(self.samples, self.model_pool)

    def regenerate_negatives_mix(self) -> None:
            self.negatives = self.add_negatives_mix(self.samples, self.model_pool)

    def reset_rotation(self, rotation_deg) -> None:
        self.fixed_rotation_augmentation = rotation_deg # unit: deg
    
    def load_splitset_from_json(self, filename_file: str, quat_file: str, sample_key_name: str) -> Any:
        
        with open(filename_file) as f:
            content = json.load(f)
            json_list = content[sample_key_name]
            
        with open(quat_file) as qf: # rotation quaternion of the cad models relative to the scan object
            quat_content = json.load(qf)
            json_quat = quat_content[sample_key_name]
                
        samples=[]
        quats = []

        for k, v in json_list.items():
            # if os.path.exists(os.path.join(self.scan_base_path, k + ".mask")):
            if v in self.splits:
                samples.append(k)
                quats.append(json_quat[k])
        
        return samples, quats

    def load_model_pool(self, file: str) -> Any:
        if file is None:
            return None

        with open(file) as f:
            content = json.load(f)
            
            # delete duplicate elements from each list
            for cat, cat_list in content.items():
                cat_list_filtered = list(set(cat_list))
                content[cat] = cat_list_filtered

        return content

    def load_full_annotation(self, file: str) -> Dict:
        print("loading full annotation")
        with open(file) as f:
            loaded_list = json.load(f)  # this json has only one List

        formatted_full_annotation = {}
        # for key, values in full_dataset_dict["scan2cad_objects"].items():
        for key in self.samples:
            _temp_split = key.split('_')
            # TODO: consider 2d3ds has different name
            scan_id = '_'.join(_temp_split[:2])
            cad_id = _temp_split[-1]
            # find scan in full_annotation
            try:
                scan = next(_scan for _scan in loaded_list if scan_id == _scan['id_scan'])
            except StopIteration:
                raise IndexError(f"Can not find scan_id={scan_id} in {file}")
            # find cad
            try:
                aligned_model = next(_aligned for _aligned in scan['aligned_models'] if cad_id == _aligned['id_cad'])
            except StopIteration:
                raise IndexError(f"Can not find cad_id={cad_id} in scan {scan_id}, in file {file}")

            formatted_full_annotation[key] = aligned_model
            # print(f"key = {key}, scale = {aligned_model['trs']['scale']}, sym = {aligned_model['sym']}")

        print("full annotation loaded")
        return formatted_full_annotation

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Any:
        objects = {}

        cur_sample = self.samples[index]
        parts = str.split(cur_sample, "_")

        if self.scan_dataset_name == "scannet":
            assert len(parts)==6, "Wrong sample name"
            # exmaple: 'scene0568_02__7_02871439_c214e1d190cb87362a9cd5247487b619'
            # parts: ['scene0568', '02', '', '7', '02871439', 'c214e1d190cb87362a9cd5247487b619']
            # scan_object_name: scene0568_02__7
            # cad_object_name: 02871439/c214e1d190cb87362a9cd5247487b619

            scan_id = parts[0] + "_" + parts[1]
            scan_object_name = scan_id + "__" + parts[3]
            category = parts[4]
            model_id = parts[5]

            if not self.input_only_mask:
                scan_obj_path = os.path.join(self.scan_base_path, scan_id, scan_id + self.object_folder_extension, scan_object_name + self.scan_obj_format)
            
            scan_mask_path = os.path.join(self.scan_base_path, scan_id, scan_id + self.mask_folder_extension, scan_object_name + self.scan_mask_format)

        elif self.scan_dataset_name == "2d3ds":       
            # assert len(parts)==9, "Wrong sample name"

            # example: 'Area_3_office_2_chair_38_3_03001627_94e289c89059106bd8f74b0004a598cd'
            # parts: ['Area', '3', 'office', '2', 'chair', '38', '3', '03001627', '94e289c89059106bd8f74b0004a598cd']
            area_name = parts[0]+"_"+parts[1]
            room_name = parts[2]+"_"+parts[3]  
            obj_global_name =  parts[4] + "_" + parts[5] + "_" + parts[6]
            scan_object_name = area_name + "_" + room_name + "_" + obj_global_name
            category = parts[7]
            model_id = parts[8]
            
            if not self.input_only_mask:            
                scan_obj_path = os.path.join(self.scan_base_path, area_name, room_name, room_name + self.object_folder_extension, scan_object_name + self.scan_obj_format)
    
            scan_mask_path = os.path.join(self.scan_base_path, area_name, room_name, room_name + self.mask_folder_extension, scan_object_name + self.scan_mask_format)

        cad_object_name = category + "/" + model_id
        cad_path = os.path.join(self.cad_base_path, cad_object_name + self.cad_format)
        
        # Load scan object (mask)
        if not self.input_only_mask:
            objects["scan"] = self.transformation(data.load_raw_df(scan_obj_path)).tdf

        sample_obj = data.load_mask(scan_mask_path)
        scan_matrix = sample_obj.matrix
        objects["mask"] = sample_obj.tdf

        # Load CAD model
        objects["cad"] = self.transformation(data.load_raw_df(cad_path)).tdf
        objects["positive"] = objects["cad"]

        # Load negative CAD sample
        if self.has_negatives:
            negative_cad_name = self.negatives[index]
            negative_cad_path = os.path.join(self.cad_base_path, negative_cad_name + self.cad_format)
            objects["negative"] = self.transformation(data.load_raw_df(negative_cad_path)).tdf 
        else:
            negative_cad_name = ""

        # Apply augmentations (same augmentations to the scan, mask, positive and negative cad model)
        # Issue: do we need to apply rotation to the negative models?

        # get relative rotation between cad model and scan object 
        # take care of the coordinate system convertion
        quat = self.quats[index]
        Rsc = data.get_rot_cad2scan(quat)

        # cur_quat = np.quaternion(quat[0], quat[1], quat[2], quat[3])
        # Rwc = quaternion.as_rotation_matrix(cur_quat)
        # Rsw = np.zeros((3,3))
        # Rsw[0,0]=1
        # Rsw[1,2]=-1
        # Rsw[2,1]=1
        # Rzx = np.zeros((3,3))
        # Rzx[0,2]=1
        # Rzx[1,1]=1
        # Rzx[2,0]=1
        # Rsc=Rsw @ Rwc @ inv(Rsw)
        # Rsc=inv(Rzx).dot(Rsw).dot(Rzx)
        # print(cur_sample ,":", Rwc)

        if self.rotation_augmentation == "random": # random in batch
            #rotations = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330] # rotation around y axis for a random angle within the list (0:30:330), 12 possibilities 
            rotations = np.linspace(0,360,self.rotation_aug_count,endpoint=False).tolist()
            degree = random.choice(rotations) # unit: deg
            # angle = degree * math.pi / 180 # rad
            # use rotation matrix instead of angle as the input
        elif self.rotation_augmentation == "fixed":
            degree = self.fixed_rotation_augmentation
        else: # "none"
            degree = 0
        
        if negative_cad_name == cad_object_name: # [negative case 1] Same CAD as the ground truth, but with different rotation around z
            random_for_neg = True
            #print("Same name")
        else:                                # [negative case 2] different CAD as the ground truth, but with almost (only pertubation) same rotation around z
            random_for_neg = False

        if self.align_to_scan:
            objects = {k: data.rotation_augmentation_interpolation_v3(o, k, aug_rotation_z = degree, pre_rot_mat = Rsc,
                       pertub_deg_max = int(self.rotation_interval/2), random_for_neg = random_for_neg) for k, o in objects.items()} # same rotation for all the objects since the degree is already chosen randomly    
        # for the positive samples, there will still be a pertubation angle
        
        else: # align_to_cad (canonical learning)
            objects = {k: data.rotation_augmentation_interpolation_v5(o, k, aug_rotation_z = degree, pre_rot_mat = Rsc) for k, o in objects.items()} # same rotation for all the objects since the degree is already chosen randomly
            #objects = {k: data.rotation_augmentation_interpolation_v4(o, k, angle, Rsc) for k, o in objects.items()}
            #print ("Apply rotation augmentation (interpolation)")
            #print(degree)

        # elif self.rotation_augmentation == "fixed":
        #     objects = {k: data.rotation_augmentation_fixed(o) for k, o in objects.items()}
        #     #print ("Apply rotation augmentation (fixed)")

        if self.flip_augmentation:
            objects = {k: data.flip_augmentation(o) for k, o in objects.items()}
            #print ("Apply flipping augmentation")

        if self.jitter_augmentation:
            objects = {k: data.jitter_augmentation(o) for k, o in objects.items()}
            #print ("Apply jitter augmentation")

        objects = {k: np.ascontiguousarray(o) for k, o in objects.items()}

        # Define final outputs (dictionary)
        scan_data = {"name": scan_object_name}
        if "scan" in objects:
            scan_data["content"] = objects["scan"]

        if "mask" in objects:
            scan_data["mask"] = objects["mask"]

        scan_data["scale"] = np.array(self.formatted_full_annotation[cur_sample]["trs"]["scale"])
        scan_data["symmetry"] = self.formatted_full_annotation[cur_sample]["sym"]
        scan_data["matrix"] = scan_matrix[0, 0]

        cad_data = {"name": cad_object_name, "content": objects["cad"]}

        if self.has_negatives:
            negative_data = {"name": negative_cad_name, "content": objects["negative"]}
            # added
            positive_data = {"name": cad_object_name, "content": objects["positive"]}
            #
            return scan_data, cad_data, negative_data, positive_data
        else:
            return scan_data, cad_data
    
    # deprecated
    # load all kinds of data
    def _load(self, path, mask_path=None):
        model, info = self._load_df(path) # model is the tdf, info is all the metadata of the voxel representation

        if self.mask_scans and mask_path is not None: # actually false and None
            # if not so
            mask, mask_info = self._load_mask(mask_path) # mask is the binary voxel
            info.tdf = self.mask_object(model, mask) #mask the model voxel (if mask=0, let model voxel=inf)

        info = self.transformation(info) # in our case, transformation is to_occupancy_grid, so it's tdf -> occupancy voxel (for both model and mask)
        #data.print_info(info) # check the dimension
        return info.tdf, info # return the occupancy voxel (info.tdf)

    # deprecated
    @staticmethod
    def mask_object(model, mask):
        masked = np.where(mask, model, np.NINF)

        return masked

    # deprecated
    @staticmethod
    def _load_mask(filepath: str) -> Tuple[np.array, np.array]:
        mask = data.load_mask(filepath)

        return mask.tdf, mask

    # deprecated
    @staticmethod
    def _load_df(filepath: str) -> Tuple[np.array, np.array]:
        if os.path.splitext(filepath)[1] == ".mask":
            sample = data.load_mask(filepath) # return tdf as binary occupancy mask
            sample.tdf = 1.0 - sample.tdf.astype(np.float32) # binary occupancy (foreground_maksed=1) -> tdf (distance=0)
        else: # no difference for df and sdf but actually not
            sample = data.load_raw_df(filepath)  # return tdf if the input is just tdf instead of mask
        patch = sample.tdf
        return patch, sample

    # deprecated
    @staticmethod
    def _load_sdf(filepath: str) -> Tuple[np.array, np.array]:
        sample = data.load_sdf(filepath)
        patch = sample.tdf
        return patch, sample

    # @staticmethod
    # def get_shapenet_object(object_name: str) -> str:
    #
    #     # exmaple: 'scene0568_02__7_02871439_c214e1d190cb87362a9cd5247487b619'
    #     # parts: ['scene0568', '02', '', '7', '02871439', 'c214e1d190cb87362a9cd5247487b619']
    #     # shapenet_object: 02871439/c214e1d190cb87362a9cd5247487b619
    #     parts = str.split(object_name, "_")
    #
    #     if len(parts) == 6:
    #         return parts[4] + "/" + parts[5] + "__0__"
    #     else:
    #         return None
    #
    # @staticmethod
    # def get_scannet_object(object_name: str) -> Tuple[str, str]:
    #
    #     # exmaple: 'scene0568_02__7_02871439_c214e1d190cb87362a9cd5247487b619'
    #     # parts: ['scene0568_02', '__7_02871439_c214e1d190cb87362a9cd5247487b619']
    #     # scannet_object: scene0568_02
    #     parts = str.split(object_name, "_")
    #
    #     if len(parts) == 6:
    #         scan_name = parts[0] + "_" + parts[1]
    #         scan_object_name = scan_name + "__" + parts[3]
    #         return scan_name, scan_object_name
    #     else:
    #         return object_name, object_name
