import math
import random
import numpy as np
from numpy.linalg import inv
import torch
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import copy
from typing import List

import data


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def filter_tran_only_z_rot(T_mat):
    t, R, s = data.decompose_mat4(T_mat)
    euler_angle = data.rot2eul(R)
    euler_angle[0] = 0.0
    euler_angle[1] = 0.0
    R_only_z_rot = data.eul2rot(euler_angle)
    T_only_z_rot = data.make_T_from_trs(t, R_only_z_rot, s)
    return T_only_z_rot

def reg_icp_p2p_o3d(source, target, threshold = 0.1, trans_init = np.eye(4), estimate_scale: bool = True, 
                    only_z_rot: bool = False, max_iter = 30, verbose: bool = True):
    
    threshold_list = [threshold, threshold*0.8, threshold*0.6, threshold*0.4, threshold*0.2, threshold*0.1]
    #threshold_list = [threshold, threshold*0.9, threshold*0.8, threshold*0.7, threshold*0.6, threshold*0.5, threshold*0.4]

    trans_cur = trans_init

    counter = 0
    for thre in threshold_list:
        counter += 1
        if counter > 3 and estimate_scale:
            estimate_with_scale = True
        else:
            estimate_with_scale = False
        reg_p2p = o3d.registration.registration_icp(
                source, target, float(thre), trans_cur,
                o3d.registration.TransformationEstimationPointToPoint(with_scaling=estimate_with_scale),
                o3d.registration.ICPConvergenceCriteria(max_iteration = 5))
        trans_cur = reg_p2p.transformation

    tran_reg = reg_p2p.transformation @ inv(trans_init)
    
    if only_z_rot:
        if verbose:
            print("T before filtering:")
            print(tran_reg)

        tran_reg = filter_tran_only_z_rot(tran_reg)

        if verbose:
            print("T after filtering:")
            print(tran_reg)

        tran_result = tran_reg @ trans_init
    else:
        tran_result = reg_p2p.transformation

    evaluation = o3d.registration.evaluate_registration(source, target,
                                                        float(threshold_list[-1]), reg_p2p.transformation)

    if verbose:
        print("Apply point-to-point ICP")
        print(reg_p2p)
        print("Transformation is:")
        print(tran_result)
        #print(evaluation)
    # draw_registration_result(source, target, tran_result)
    return tran_result, evaluation


def reg_icp_p2l_o3d(source, target, threshold = 0.1, trans_init = np.eye(4), estimate_scale: bool = False, 
                    only_z_rot: bool = False, max_iter = 30, verbose: bool = True):
    
    threshold_list = [threshold, threshold*0.9, threshold*0.8, threshold*0.7, threshold*0.6, threshold*0.5, threshold*0.4]

    trans_cur = trans_init 

    for thre in threshold_list:
        reg_p2l = o3d.registration.registration_icp(
                source, target, float(thre), trans_cur,
                o3d.registration.TransformationEstimationPointToPlane(),
                o3d.registration.ICPConvergenceCriteria(max_iteration = 5))
        trans_cur = reg_p2l.transformation

    tran_reg = reg_p2l.transformation @ inv(trans_init)
    
    if only_z_rot:
        if verbose:
            print("T before filtering:")
            print(tran_reg)

        tran_reg = filter_tran_only_z_rot(tran_reg)

        if verbose:
            print("T after filtering:")
            print(tran_reg)

        tran_result = tran_reg @ trans_init
    else:
        tran_result = reg_p2l.transformation
    
    evaluation = o3d.registration.evaluate_registration(source, target,
                                                        float(threshold_list[-1]), reg_p2l.transformation)

    if verbose:
        print("Apply point-to-plane ICP")
        print(reg_p2l)
        print("Transformation is:")
        print(tran_result)                                                    
    # draw_registration_result(source, target, tran_result)
    return tran_result, evaluation



"""
ref: https://github.com/ClayFlannigan/icp/blob/master/icp.py
"""

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def reg_icp_p2p_np(A, B, init_pose=None, max_iterations=30, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i

    