import os
import struct
import numpy as np
import copy
import open3d as o3d
from typing import Tuple, AnyStr

# the IO interface, quite important

class Sample:
    filename: str = ""
    dimx: int = 0
    dimy: int = 0
    dimz: int = 0
    size: float = 0.0
    matrix: np.array = None
    tdf: np.array = None
    sign: np.array = None
    rot: np.array = None

def print_info(sample: Sample):
    print(sample.dimx, 'x', sample.dimy, 'x', sample.dimz)
    print('Voxel resolution:',sample.size)
    print(sample.matrix)
    occupied_voxel_count = np.count_nonzero(sample.tdf)
    print(occupied_voxel_count, 'occupied voxels')

def save_tdf(filename: str, tdf: np.array, dimx: int, dimy: int, dimz: int, voxel_size: float, matrix: np.array) -> None:
    with open(filename, 'wb') as f:
        f.write(struct.pack('I', dimx))
        f.write(struct.pack('I', dimy))
        f.write(struct.pack('I', dimz))
        f.write(struct.pack('f', voxel_size))
        f.write(struct.pack("={}f".format(16), *matrix.flatten("F")))

        num_elements = dimx * dimy * dimz
        f.write(struct.pack("={}f".format(num_elements), *tdf.flatten("F")))


def load_sample_info(filename: str) -> Tuple[Sample, AnyStr, int]:
    assert os.path.isfile(filename), "File not found: %s" % filename
    content = open(filename, "rb").read()

    # load the meta-information: dimx(I), dimy(I), dimz(I), voxel size(f), and transformation matrix (4x4,f)
    sample = Sample()
    sample.filename = filename
    sample.dimx = struct.unpack('I', content[0:4])[0]
    sample.dimy = struct.unpack('I', content[4:8])[0]
    sample.dimz = struct.unpack('I', content[8:12])[0]
    sample.size = struct.unpack('f', content[12:16])[0] #voxel size (we need this one for scale initial guess)

    matrix_size = int(16 * 4)
    sample.matrix = struct.unpack('f' * 16, content[16:16 + matrix_size])
    sample.matrix = np.asarray(sample.matrix, dtype=np.float32).reshape([4, 4]) # transformation matrix

    start_index = 16 + matrix_size

    return sample, content, start_index


def load_raw_df(filename: str) -> Sample:
    # first load the meta-data
    s, raw, start_index = load_sample_info(filename)
    n_elements = s.dimx * s.dimy * s.dimz

    # Load distance values
    s.tdf = struct.unpack('f' * n_elements, raw[start_index:start_index + n_elements * 4])
    s.tdf = np.asarray(s.tdf, dtype=np.float32).reshape([1, s.dimz, s.dimy, s.dimx])

    if os.path.splitext(filename)[1] == ".df": 
        s.tdf = s.tdf.transpose((0, 2, 1, 3))# change to the dimx, dimz, dimy order (this should be correct, make CAD model aligned with scan object, only allow rotation around z axis
        # actually flipped (reflected)

    #else: # ".sdf" 
        #s.tdf = s.tdf.transpose((0, 1, 2, 3))# change to the standard dimx, dimy, dimz order , originally (0,3,2,1), but do not know why

    return s

def load_mask(filename: str) -> Sample:
    # first load the meta-data
    s, raw, start_index = load_sample_info(filename)
    n_elements = s.dimx * s.dimy * s.dimz

    # Load binary values (maybe binary occupancy voxel or frontground/backgroud mask)
    s.tdf = struct.unpack('B' * n_elements, raw[start_index:start_index + n_elements])
    #s.tdf = np.asarray(s.tdf, dtype=np.dtype("?")).reshape([1, s.dimz, s.dimy, s.dimx], order='C') # bool version

    s.tdf = np.asarray(s.tdf, dtype=np.dtype("float32")).reshape([1, s.dimz, s.dimy, s.dimx], order='C')

    #s.tdf = s.tdf.transpose((0, 1, 2, 3))# change to the standard dimx, dimy, dimz order , originally (0,3,2,1), but do not know why

    # binary mask, used or not
    return s


# Encode sign in a separate channel
def load_sdf(filename: str) -> Sample:
    s = load_raw_df(filename)

    truncation = 3 * s.size # set truncation as 3 * voxel size
    s.sign = np.sign(s.tdf)  # Encode sign in separate channel
    s.sign[s.sign >= 0] = 1
    s.sign[s.sign < 0] = 0
    s.tdf = np.abs(s.tdf)  # Omit sign
    s.tdf = np.clip(s.tdf, 0, truncation)  # Truncate
    s.tdf = s.tdf / truncation  # Normalize
    s.tdf = 1 - s.tdf  # flip (why ?) -> make sdf a bit like the occupancy voxel
    s.tdf = s.tdf[:, :s.dimx, :s.dimy, :s.dimz]
    s.tdf = np.concatenate((s.tdf, s.sign), axis=0)

    return s

# TODO
# calculate bbx length in voxel space (unit: voxel) 
def cal_voxel_bbx_length(voxel_in: np.array) -> np.array:

    occupied_idx = np.where(voxel_in > 0.75) # 0.5 or 0.75

    bbx_lz = np.max(occupied_idx[0]) - np.min(occupied_idx[0]) # z
    bbx_ly = np.max(occupied_idx[1]) - np.min(occupied_idx[1]) # y
    bbx_lx = np.max(occupied_idx[2]) - np.min(occupied_idx[2]) # x

    bbx_length = np.array([bbx_lx, bbx_ly, bbx_lz], dtype="float") #[int]
    return bbx_length
    

def voxel2point(voxel: np.array, category: int = -1, name: str = "", color_mode: str = "height",
                random_densify: bool = False, visualize_prob_threshold = 0.5):

    visualize_with_category = False
    
    if category >= 0:
        visualize_with_category = True
    
    point_cloud = []

    voxel_shape = np.shape(voxel)

    half_size = voxel_shape[0] / 2
    axis_size = half_size / 4

    point_cloud.append([half_size, half_size, half_size, 0, 0, 0])  # dummy center point

    for z in range(voxel_shape[0]): #Z
        for y in range(voxel_shape[1]): #Y
            for x in range(voxel_shape[2]): #X
                cur_prob = voxel[z, y, x]
                if  cur_prob > visualize_prob_threshold:
                    if visualize_with_category:
                        cur_point = [x, y, z, category]
                    else:
                    # TODO: update the colormap for visualization
                        if color_mode == "height":
                            cur_point = [x, y, z, (0.2 + 0.8 * z/ voxel_shape[0]) * 255, (1.0 - 0.8 * z / voxel_shape[0]) * 255, 0]
                        elif color_mode == "prob":
                            cur_point = [x, y, z, cur_prob * 255, (1.0 - cur_prob) * 255 , 0]
                        else:
                            cur_point = [x ,y ,z, 0]

                    point_cloud.append(cur_point)
                    # if random_densify:
                    #     for n in range(5): # TODO: densify # TODO: change order
                    #         if visualize_with_category:
                    #             densified_point = [x+ 0.5 - np.random.random(1), y + 0.5 - np.random.random(1),  z + 0.5 - np.random.random(1), category]
                    #         else:
                    #             densified_point = [x+ 0.5 - np.random.random(1), y + 0.5 - np.random.random(1),  z + 0.5 - np.random.random(1), (0.2 + 0.8 * z / voxel_shape[0]) * 255, (1.0 - 0.8 * z / voxel_shape[0]) * 255, 0]
                    #         point_cloud.append(densified_point)

    point_cloud_array = np.asarray(point_cloud)
    axis = np.array([{"start": [half_size,half_size,half_size], "end": [half_size+axis_size,half_size,half_size]}, {"start": [half_size,half_size,half_size], "end": [half_size,half_size+axis_size,half_size]}, {"start": [half_size,half_size,half_size], "end": [half_size,half_size,half_size+axis_size]}])
    bbx = np.array([{"corners": [ [0,0,0], [0,voxel_shape[1],0], [0,0,voxel_shape[2]], [voxel_shape[0],0,0], [voxel_shape[0], voxel_shape[1], 0], [0, voxel_shape[1], voxel_shape[2]],[voxel_shape[0],0,voxel_shape[2]],[voxel_shape[0],voxel_shape[1],voxel_shape[2]]],
                                "label": name,
                                "color": [123,321,111]}])
    point_cloud_object = {"type": "lidar/beta", "points": point_cloud_array, "boxes": bbx, "vectors": axis}

    #point_cloud_object = {"type": "lidar/beta", "points": point_cloud_array}
                         
    return point_cloud_object


def o3dpcd2point(o3d_pcd, category: int = -1, name: str = "", color_mode: str = "unique", down_rate = 5):

    o3d_pcd_down = o3d_pcd.uniform_down_sample(down_rate)
    
    if category == -1:
        o3d_pcd_down.paint_uniform_color([1, 0.706, 0])
    elif category == -2:
        o3d_pcd_down.paint_uniform_color([0, 0.651, 0.929])
    else:
        o3d_pcd_down.paint_uniform_color([0.8, 0, 0])

    point_cloud_array = np.asarray(o3d_pcd_down.points)
    point_cloud_color = 255 * np.asarray(o3d_pcd_down.colors)

    point_cloud_array = np.hstack((point_cloud_array, point_cloud_color))
    
    if category >= 0:
        visualize_with_category = True
    
    # TODO:
    # if visualize_with_category:
    #         cur_point = [x, y, z, category]


    point_cloud_object = {"type": "lidar/beta", "points": point_cloud_array}
    return point_cloud_object


def o3dreg2point(o3d_source, o3d_target, trans2t, name: str = "", color_mode: str = "unique", down_rate = 10):

    # o3d_pcd_source = copy.deepcopy(o3d_source)
    # o3d_pcd_target = copy.deepcopy(o3d_target)
    
    o3d_pcd_source = o3d_source.uniform_down_sample(down_rate)
    o3d_pcd_target = o3d_target.uniform_down_sample(down_rate)
    
    o3d_pcd_source.paint_uniform_color([1, 0.706, 0])
    o3d_pcd_target.paint_uniform_color([0, 0.651, 0.929])

    o3d_pcd_source.transform(trans2t)
    
    source_point_cloud_array = np.asarray(o3d_pcd_source.points)
    target_point_cloud_array = np.asarray(o3d_pcd_target.points)

    source_point_cloud_color = 255 * np.asarray(o3d_pcd_source.colors)
    target_point_cloud_color = 255 * np.asarray(o3d_pcd_target.colors)

    source_point_cloud_array = np.hstack((source_point_cloud_array, source_point_cloud_color))
    target_point_cloud_array = np.hstack((target_point_cloud_array, target_point_cloud_color))

    point_cloud_array = np.vstack((source_point_cloud_array, target_point_cloud_array))

    point_cloud_object = {"type": "lidar/beta", "points": point_cloud_array}
    return point_cloud_object