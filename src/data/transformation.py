import numpy as np
from numpy.linalg import inv
import quaternion
import math
from data.sample import Sample


def truncation_normalization_transform(s: Sample) -> Sample:
    truncation = 3 * s.size # 3
    s.tdf = np.abs(s.tdf)  # Omit sign
    s.tdf = np.clip(s.tdf, 0, truncation)  # Truncate
    s.tdf = s.tdf / truncation  # Normalize [0-1]
    s.tdf = 1 - s.tdf  # flip [1-0]

    return s


def to_flipped_occupancy_grid(s: Sample) -> Sample:
    s.tdf = np.abs(s.tdf) > s.size  # Omit sign
    s.tdf = s.tdf.astype(np.float32)
    s.tdf = s.tdf[:, :s.dimx, :s.dimy, :s.dimz]

    return s


def to_occupancy_grid(s: Sample) -> Sample:
    s.tdf /= s.size
    s.tdf = np.less_equal(np.abs(s.tdf), 1).astype(np.float32) # 1 only when abs(tdf) <= voxel_size
    s.tdf = s.tdf[:, :s.dimx, :s.dimy, :s.dimz]

    return s

def get_rot_cad2scan(quat_cad2world):
    quat_wc = np.quaternion(quat_cad2world[0], quat_cad2world[1], quat_cad2world[2], quat_cad2world[3])
    Rwc = quaternion.as_rotation_matrix(quat_wc)
    Rsw = np.zeros((3,3))
    Rsw[0,0]=1
    Rsw[1,2]=-1
    Rsw[2,1]=1
    Rzx = np.zeros((3,3))
    Rzx[0,2]=1
    Rzx[1,1]=1
    Rzx[2,0]=1
    Rsc=Rsw @ Rwc @ inv(Rsw)

    return Rsc


# used for cad model to scan registration 
def get_tran_init_guess(scan_grid2world, predicted_rotation_deg, flip_on: bool = True, direct_scale: bool = False, 
                        voxel_size = 32, cad_scale_multiplier: np.array = np.ones(3)):

    if direct_scale: # directly use the predicted scale of the neural network. In such case, cad_scale_multiplier should be the predicted scale itself
        s_init = cad_scale_multiplier
    else: # use voxel resolution and bounding box length to estimate the scale. In such case, cad_scale_multiplier should be bounding box length ratio (bbx_l_scan / bbx_l_cad)
        scan_voxel_res = scan_grid2world[0,0]
        #print("scan_voxel_res:", scan_voxel_res)
        cad_voxel_res = 1.0 / voxel_size
        # s_init = scan_voxel_res / cad_voxel_res * cad_scale_multiplier 
        s_x_init = scan_voxel_res / cad_voxel_res * cad_scale_multiplier[0]
        s_y_init = scan_voxel_res / cad_voxel_res * cad_scale_multiplier[1]
        s_z_init = scan_voxel_res / cad_voxel_res * cad_scale_multiplier[2]
        s_init = np.asarray([s_x_init, s_y_init, s_z_init])
    
    R_init = get_rot_mat_around_z((180.0 -predicted_rotation_deg) / 180.0 * math.pi) # unit: rad
    t_init = scan_grid2world[3, 0:3] + scan_voxel_res * voxel_size / 2
    T_init = make_T_from_trs(t_init, R_init, s_init)
    
    T_flip = np.zeros((4,4))
    T_flip[0,0]=1
    T_flip[1,2]=-1
    T_flip[2,1]=1
    T_flip[3,3]=1
    
    if flip_on:
        T_init = T_init.dot(T_flip)              
    
    return T_init


def make_T_from_trs(t, R, s):
    t_mat = np.eye(4)
    t_mat[0:3, 3] = t
    R_mat = np.eye(4)
    R_mat[0:3, 0:3] = R
    s_mat = np.eye(4)
    s_mat[0:3, 0:3] = np.diag(s)
    
    # first scale, then rotation, finally translation
    Tran = t_mat.dot(R_mat).dot(s_mat)
    return Tran

def decompose_mat4(Tran):
    R = Tran[0:3, 0:3]
    sx = np.linalg.norm(R[0:3, 0])
    sy = np.linalg.norm(R[0:3, 1])
    sz = np.linalg.norm(R[0:3, 2])

    s = np.array([sx, sy, sz])

    R[:, 0] /= sx
    R[:, 1] /= sy
    R[:, 2] /= sz

    t = Tran[0:3, 3]
    return t, R, s

def get_rot_mat_around_z(angle): # unit: rad
    R_mat = np.eye(3)
    R_mat[0,0] = np.cos(angle)
    R_mat[0,1] = -np.sin(angle)
    R_mat[1,0] = np.sin(angle)
    R_mat[1,1] = np.cos(angle)
    return R_mat