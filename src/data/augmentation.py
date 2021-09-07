import math
import random
import numpy as np
import scipy.ndimage
import torch
from typing import List

# Generating more training data by data augmentation

# N.B. explicit define the data type

# TODO: add them to another math subfolder
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)

    return n < 1e-6

# rotation matrix to euler angles (unit: rad)
def rot2eul(R):

    assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-4

    if not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

# euler angles (unit: rad) to rotation matrix 
def eul2rot(theta) :

    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])

    return R

def flip_rotmat (mat: np.array) -> np.array:
    fliped_mat = mat
    fliped_mat[0, 1] = mat[0, 2]
    fliped_mat[0, 2] = mat[0, 1]
    fliped_mat[1, 0] = - mat[2, 0]
    fliped_mat[1, 1] = - mat[2, 2]
    fliped_mat[1, 2] = - mat[2, 1]
    fliped_mat[2, 0] = mat[1, 0]
    fliped_mat[2, 1] = mat[1, 2]
    fliped_mat[2, 2] = mat[1, 1]
    return fliped_mat

# generating random number
# 0,1,2,3 (axis)
def random_rotation() -> int:
    return np.random.randint(0, 4)
# example -180 - 179
def random_degree(angle_min: int, angle_max: int) -> int:
    return np.random.randint(angle_min, angle_max)
# 0 - 2pi
def random_angle() -> float:
    return np.random.random() * 2 * math.pi

# [-4,4],[-4,4],[-4,4]
def random_jitter(max_jitter: int = 4) -> List[int]:
    jitter = [np.random.randint(-max_jitter, max_jitter) for _ in range(3)]
    return jitter

# rotate around y axis (why y axis)
def get_rotations_y(angles: np.array) -> np.array:
    cos_angle = np.cos(angles)
    sin_angle = np.sin(angles)
    rots = np.eye(3)[np.newaxis, :]
    rots = np.tile(rots, [angles.shape[0], 1, 1])
    rots[:, 0, 0] = cos_angle
    rots[:, 0, 2] = sin_angle
    rots[:, 2, 0] = -sin_angle
    rots[:, 2, 2] = cos_angle
    return rots.astype(np.float32)

# z aixs is the first dimension (take care that now the dimension sequence is z,y,x)
def get_rotations_z(angles: np.array) -> np.array:
    cos_angle = np.cos(angles)
    sin_angle = np.sin(angles)
    rots = np.eye(3)[np.newaxis, :]
    rots = np.tile(rots, [angles.shape[0], 1, 1])
    rots[:, 1, 1] = cos_angle
    rots[:, 1, 2] = sin_angle
    rots[:, 2, 1] = -sin_angle
    rots[:, 2, 2] = cos_angle
    return rots.astype(np.float32)

def get_rotations_from_mat(mat: np.array, count) -> np.array:
    rots = mat[np.newaxis, :]
    rots = np.tile(rots, [count, 1, 1])
    return rots.astype(np.float32)

def rotation_augmentation_fixed(grid: np.array, num_rotations=None) -> np.array:
    if num_rotations is None:
        angle = np.random.randint(0, 4)
    else:
        angle = num_rotations

    grid = np.rot90(grid, k=angle, axes=(1, 3))
    return grid


def rotate_grid(grid: np.array, num_rotations: int) -> np.array:
    patch = np.rot90(grid, k=num_rotations, axes=(1, 3))
    return patch

# used, figure out why this works (unit: degree)
# take the scan as the target, align models to the scan
def rotation_augmentation_interpolation_v3(grid: np.array, key: str, aug_rotation_z = None, pre_rot_mat = np.eye(3), 
                                           random_for_neg = False, pertub_for_pos = True, pertub_deg_max: int = 15) -> np.array:
    
    # TODO: figure out which is better: rotate negative cad with the same angle or not (trun on/off by setting the argument random_for_neg)
    # False is better because we'd like to have harder negative samples

    if key == "positive" and pertub_for_pos: # also add this line for v5
        angle = aug_rotation_z + random_degree(-pertub_deg_max, pertub_deg_max) # unit: degree, -15 - 15
    elif key == "negative" and random_for_neg:
        angle = random_degree(0, 359) # unit: degree, 0 - 359
    else:
        angle = aug_rotation_z # unit: degree (the augmentation random rotation)
    
    euler_angle = rot2eul(pre_rot_mat) * 180.0 / math.pi # unit: degree (the alignment rotation)
    
    model_to_align = ["cad", "negative", "positive"] 

    if key in model_to_align:
        #print("apply additional rotation:", angle*180/math.pi,  euler_angle[2]*180/math.pi)
        angle = angle + 180.0 - euler_angle[2] # unit: degree
    
    if angle != 0:
        grid = scipy.ndimage.rotate(grid, angle, (2, 3), False, prefilter=True, order=3, cval=0, mode="nearest") # rotate in x-y plane, unit: degree
    
    interpolate_thre = 0.5 # TODO: figure out what's the suitable value and the principle of the grid interpolation

    grid[grid<interpolate_thre] = 0
    
    return grid

# take the model as the target, align scan to the model at the canonical coordinate system
def rotation_augmentation_interpolation_v5(grid: np.array, key: str, aug_rotation_z = None, pre_rot_mat = np.eye(3), random_for_neg = False, pertub_for_pos = True) -> np.array:
        
    angle = aug_rotation_z # unit: degree (the augmentation random rotation)

    euler_angle = rot2eul(pre_rot_mat) * 180.0 / math.pi # unit: degree (the alignment rotation)
    
    model_to_align = ["scan", "mask"] 

    if key in model_to_align:
        #print("apply additional rotation:", angle*180/math.pi,  euler_angle[2]*180/math.pi)
        angle = angle - 180.0 + euler_angle[2] # unit: degree
    
    grid = scipy.ndimage.rotate(grid, angle, (2, 3), False, prefilter=True, order=3, cval=0, mode="nearest") # rotate in x-y plane, unit: degree
    
    # interpolate_thre = 0.5 # TODO: figure out what's the suitable value and the principle of the grid interpolation

    # grid[grid<interpolate_thre] = 0
    
    return grid


# figure out why this does not work
def rotation_augmentation_interpolation_v4(grid: np.array, key: str, aug_rotation_z = None, pre_rot_mat = np.eye(3)) -> np.array:
    if aug_rotation_z is None:
        aug_rotation_z = random_angle()


    scans = torch.from_numpy(np.expand_dims(grid, axis=0))
    num = scans.shape[0]

    # print(aug_rotation_z)

    #rots = np.asarray([aug_rotation_z])

    rots = np.asarray([0.0])
    #rotations_y = torch.from_numpy(get_rotations_y(rots))
    aug_rot_mat = get_rotations_z(rots)

    total_rot = aug_rot_mat.dot(pre_rot_mat)

    model_to_rot = ["cad", "negative"]
    if key in model_to_rot:
        rot_mat = torch.from_numpy(total_rot).float()
        #rot_mat = torch.from_numpy(get_rotations_from_mat(pre_rot_mat.astype(np.float32),rots.shape[0]))
        #print(pre_rot_mat)
    else:
        rot_mat = torch.from_numpy(aug_rot_mat).float()
        
    # apply rotation and keep the voxel's structure
    max_size = np.array(scans.shape[2:], dtype=np.int32)
    center = (max_size - 1).astype(np.float32) * 0.5
    center = np.tile(center.reshape(3, 1), [1, max_size[0] * max_size[1] * max_size[2]])
    grid_coords = np.array(
        np.unravel_index(np.arange(max_size[0] * max_size[1] * max_size[2]), [max_size[0], max_size[1], max_size[2]]),
        dtype=np.float32) - center
    grid_coords = np.tile(grid_coords[np.newaxis, :], [num, 1, 1])
    # get grid coordinates (decentralized) before rotation
    grid_coords = torch.from_numpy(grid_coords)
    center = torch.from_numpy(center).unsqueeze(0).repeat(scans.shape[0], 1, 1)
    #grid_coords = torch.bmm(rotations_y, grid_coords) + center
    grid_coords = torch.bmm(rot_mat, grid_coords) + center
    grid_coords = torch.clamp(grid_coords, 0, max_size[0] - 1).long()
    grid_coords = grid_coords[:, 0] * max_size[1] * max_size[2] + grid_coords[:, 1] * max_size[2] + grid_coords[:, 2]
    mult = torch.arange(num).view(-1, 1) * max_size[0] * max_size[1] * max_size[2]
    grid_coords = grid_coords + mult
    grid_coords = grid_coords.long()
    scan_rots = scans.permute(0, 2, 3, 4, 1).contiguous().view(-1, 1)[grid_coords]
    scan_rots = scan_rots.view(scans.shape[0], scans.shape[2], scans.shape[3], scans.shape[4], scans.shape[1]).permute(
        0, 4, 1, 2, 3)
    scan_rots = scan_rots.numpy()

    return scan_rots[0]

def rotation_augmentation_interpolation_v2(grid: np.array, rotation=None) -> np.array:
    if rotation is None:
        rotation = random_angle()

    scans = torch.from_numpy(np.expand_dims(grid, axis=0))
    num = scans.shape[0]
    rots = np.asarray([rotation])
    #rotations_y = torch.from_numpy(get_rotations_y(rots))
    rotations_z = torch.from_numpy(get_rotations_z(rots))

    max_size = np.array(scans.shape[2:], dtype=np.int32)
    center = (max_size - 1).astype(np.float32) * 0.5
    center = np.tile(center.reshape(3, 1), [1, max_size[0] * max_size[1] * max_size[2]])
    grid_coords = np.array(
        np.unravel_index(np.arange(max_size[0] * max_size[1] * max_size[2]), [max_size[0], max_size[1], max_size[2]]),
        dtype=np.float32) - center
    grid_coords = np.tile(grid_coords[np.newaxis, :], [num, 1, 1])
    grid_coords = torch.from_numpy(grid_coords)
    center = torch.from_numpy(center).unsqueeze(0).repeat(scans.shape[0], 1, 1)
    #grid_coords = torch.bmm(rotations_y, grid_coords) + center
    grid_coords = torch.bmm(rotations_z, grid_coords) + center
    grid_coords = torch.clamp(grid_coords, 0, max_size[0] - 1).long()
    grid_coords = grid_coords[:, 0] * max_size[1] * max_size[2] + grid_coords[:, 1] * max_size[2] + grid_coords[:, 2]
    mult = torch.arange(num).view(-1, 1) * max_size[0] * max_size[1] * max_size[2]
    grid_coords = grid_coords + mult
    grid_coords = grid_coords.long()
    scan_rots = scans.permute(0, 2, 3, 4, 1).contiguous().view(-1, 1)[grid_coords]
    scan_rots = scan_rots.view(scans.shape[0], scans.shape[2], scans.shape[3], scans.shape[4], scans.shape[1]).permute(
        0, 4, 1, 2, 3)
    scan_rots = scan_rots.numpy()
    return scan_rots[0]


def rotation_augmentation_interpolation(grid: np.array, rotation=None) -> np.array:
    if rotation is None:
        angle = random_degree()
    else:
        angle = rotation
    grid = scipy.ndimage.rotate(grid, angle, (1, 3), False, prefilter=True, order=3, cval=0, mode="nearest")
    return grid


def flip_augmentation(grid: np.array, flip=None) -> np.array:
    if flip is None:
        chance = random.random() < 0.5
    else:
        chance = flip

    if chance:
        grid = np.flip(grid, (1, 3))

    return grid


def jitter_augmentation(grid: np.array, jitter=None) -> np.array:
    if jitter is None:
        jitter = random_jitter()

    start = [max(0, j) for j in jitter]
    end = [max(0, -j) for j in jitter]
    pad = np.pad(grid, ((0, 0),
                        (start[0], end[0]),
                        (start[1], end[1]),
                        (start[2], end[2])), "constant", constant_values=(0, 0))

    offset_start = [max(0, -j) for j in jitter]
    offset_end = [None if max(0, j) == 0 else -j for j in jitter]
    grid = pad[:, offset_start[0]:offset_end[0], offset_start[1]:offset_end[1], offset_start[2]:offset_end[2]]

    return grid
