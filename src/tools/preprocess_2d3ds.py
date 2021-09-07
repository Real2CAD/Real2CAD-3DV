# Script used to preprocess 2d3ds dataset for Real2CAD task
# By Yue Pan and Yuanwen Yue from ETH Zurich
# 2021. 7
#
# about 2D3DS dataset (http://buildingparser.stanford.edu/dataset.html) (https://github.com/alexsax/2D-3D-Semantics)
#
# Input:
# the point_cloud.mat matlab workspace file for each area
# (Firstly you need to convert the original mat file "pointcloud.mat" to a ealier version format that can directly imported by python
# You can simply import it into Matlab and exported it (given the data is not too large the format or you need to split it into two parts)
#
# Output:
# 1. mesh with semantic label of each room (Disjoint space) of each area [*.ply]
# 2. point cloud with rgb color and semantic label of each room (Disjoint space) of each area [*.pcd]
# 3. point cloud and bounding box (bbx) of each interested object (chair, table, sofa and bookcase) of each room (Disjoint space) of each area [*.pcd for point cloud, *.ply for bbx]
# 4. 32*32*32 binary occupancy voxel of each interested object (chair, table, sofa and bookcase) of each room (Disjoint space) of each area, which can acts as the foreground mask of the object [*.mask, *.ply for visualization]

import os
import scipy.io as sio
import open3d as o3d
import colorsys
import copy
import pathlib
import json
import numpy as np

# for ply format related utilities
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
# for voxel representation data structure
from Vox import *

def process_area(base_path, area_id, mat_file_name, bbox=True, pc=True, mesh=True, voxel=True):

    # constant
    interested_cat = ["chair", "table", "bookcase", "sofa"]

    # issue: sofa in 2d3ds would be regarded as chair in Scan2CAD

    ignore_obj_cat = ["ceiling", "stairs"]
    voxel_size = 32 # You may change this value here to make the voxel much denser or sparser for downstream tasks

    data_path = os.path.join(base_path, mat_file_name)
    room_base_path = os.path.join(base_path, "rooms")
    pathlib.Path(room_base_path).mkdir(parents=True, exist_ok=True)

    # load raw data of the area
    raw_data = sio.loadmat(data_path, squeeze_me=True, struct_as_record=False)
    # print(raw_data)

    # load bounding box template
    with open("./bbox.ply", 'rb') as bbox_file:
        mesh_bbox = PlyData.read(bbox_file)
    assert mesh_bbox, "Could not read bbox template."

    with open("lut_2d3ds_scannet.json", 'r') as injson:
        lut = json.load(injson)

    area = raw_data[area_id]
    area_points = []
    area_name = area.name
    print("Area:", area_name)
    print("-----------------------")

    begin_now = False

    # for each room in the area
    for space in area.Disjoint_Space:

        room_points = o3d.geometry.PointCloud()
        room_points_label_rgb = o3d.geometry.PointCloud()
        room_points_label = o3d.geometry.PointCloud()

        room_name = space.name
        print("-----------------------")
        print("Room:", room_name)

        room_path = os.path.join(room_base_path, room_name)

        pathlib.Path(room_path).mkdir(parents=True, exist_ok=True)

        mask_path = os.path.join(room_path, room_name) + "_mask_voxel"
        pc_path = os.path.join(room_path, room_name) + "_mask_pc"
        pathlib.Path(mask_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(pc_path).mkdir(parents=True, exist_ok=True)

        # if begin_now or room_name == "office_20":
        #     begin_now = True
        # else:
        #     continue

        faces_bbox_all = []
        verts_bbox_all = []
        faces_bbox_obj = []
        verts_bbox_obj = []

        # for each object in the room
        for object in space.object:

            obj_name = object.name
            obj_cat = obj_name.split("_")[0]

            if obj_cat in ignore_obj_cat:
                continue

            print("Object:", obj_name)

            semantic_label = lut[obj_cat]["idx"][0]
            semantic_color = lut[obj_cat]["color"]

            points = object.points
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            # RGB color
            point_cloud.colors = o3d.utility.Vector3dVector(object.RGB_color / 255)
            # Semantic color
            point_cloud_label_rgb = copy.deepcopy(point_cloud)
            point_cloud_label_rgb.paint_uniform_color([semantic_color[0]/255, semantic_color[1]/255, semantic_color[2]/255])

            # for assigning the vertex label of the mesh
            point_cloud_label = copy.deepcopy(point_cloud)
            point_cloud_label.paint_uniform_color([semantic_label / 100, semantic_label / 100, semantic_label / 100])

            Bbox = np.reshape(object.Bbox, (6, 1))
            boundingBox = o3d.geometry.AxisAlignedBoundingBox(Bbox[:3], Bbox[3:])
            boundingBox.color = (1, 1, 1)
            # room_points.append(boundingBox)
            area_points.append(boundingBox)
            len_x = Bbox[3] - Bbox[0]
            len_y = Bbox[4] - Bbox[1]
            len_z = Bbox[5] - Bbox[2]
            len_max = max(len_x, len_y, len_z)
            voxel_res = len_max / (voxel_size - 2)  # 30
            cp_x = 0.5 * (Bbox[0] + Bbox[3])
            cp_y = 0.5 * (Bbox[1] + Bbox[4])
            cp_z = 0.5 * (Bbox[2] + Bbox[5])
            cp = [cp_x, cp_y, cp_z]
            minp = [cp_x - len_max / 2, cp_y - len_max / 2, cp_z - len_max / 2]

            Msbbox = np.eye(4)
            Msbbox[0, 0] = len_x / 2
            Msbbox[1, 1] = len_y / 2
            Msbbox[2, 2] = len_z / 2
            Msbbox[0, 3] = cp_x
            Msbbox[1, 3] = cp_y
            Msbbox[2, 3] = cp_z

            if bbox:
                # load object proposal bbx
                for f in mesh_bbox["face"]:
                    faces_bbox_all.append((np.array(f[0]) + len(verts_bbox_all),))
                for v in mesh_bbox["vertex"]:  # not just the 8 vertices for this bbx ply
                    v1 = np.array([v[0], v[1], v[2], 1])  # bbx vertex coordinate
                    v1 = np.dot(Msbbox, v1)[0:3]  # actual bbx vertex coordinate
                    verts_bbox_all.append(tuple(v1) + (semantic_color[0], semantic_color[1], semantic_color[2]))

                if obj_cat in interested_cat:
                    for f in mesh_bbox["face"]:
                        faces_bbox_obj.append((np.array(f[0]) + len(verts_bbox_obj),))
                    for v in mesh_bbox["vertex"]:  # not just the 8 vertices for this bbx ply
                        v1 = np.array([v[0], v[1], v[2], 1])  # bbx vertex coordinate
                        v1 = np.dot(Msbbox, v1)[0:3]  # actual bbx vertex coordinate
                        verts_bbox_obj.append(tuple(v1) + (semantic_color[0], semantic_color[1], semantic_color[2]))  # (50,50,200) original bbx color

            #room_points.append(point_cloud)
            room_points += point_cloud
            room_points_label_rgb += point_cloud_label_rgb
            room_points_label += point_cloud_label

            # Voxelization
            if voxel:
                if obj_cat not in interested_cat:
                    continue

                obj_global_name = object.global_name

                voxels = o3d.geometry.VoxelGrid.create_dense(cp, voxel_res, len_max, len_max, len_max)
                voxels_idx = []
                unique_idx_list = []
                for point in point_cloud.points:
                    voxel_idx_o = voxels.get_voxel(point)
                    voxel_idx = voxel_idx_o + int(voxel_size / 2 - 1)  # +15
                    voxel_idx = voxel_idx[::-1]  # x,y,z -> z,y,x
                    unique_idx = voxel_idx[0] * voxel_size * voxel_size + voxel_idx[1] * voxel_size + voxel_idx[2]
                    if unique_idx not in unique_idx_list:
                        voxels_idx.append(voxel_idx)
                        unique_idx_list.append(unique_idx)

                voxels_idx = np.array(voxels_idx)
                print(voxels_idx.shape[0], " occupied voxels")

                # Generate foreground mask
                mask_vox = Vox(dims=[voxel_size, voxel_size, voxel_size], res=voxel_res)
                mask_vox.grid2world = np.array(
                    [[voxel_res, 0, 0, minp[0]], [0, voxel_res, 0, minp[1]],
                    [0, 0, voxel_res, minp[2]], [0, 0, 0, 1]])
                mask_vox.mask = np.zeros((mask_vox.dims[0], mask_vox.dims[1], mask_vox.dims[2]), dtype=bool)
                for i in range(voxels_idx.shape[0]):
                    mask_vox.mask[voxels_idx[i, 0], voxels_idx[i, 1], voxels_idx[i, 2]] = True

                mask_file_name = area_name + "_" + room_name + "_" + obj_global_name + '.mask'
                mask_file_path = os.path.join(mask_path, mask_file_name)

                point_cloud_file_name = area_name + "_" + room_name + "_" + obj_global_name + '.pcd'
                pc_file_path = os.path.join(pc_path, point_cloud_file_name)

                # Generate corresponding mask (32 x 32) for background/foreground segmentation
                write_mask(mask_file_path, mask_vox)  # order = "C"

                # Convert mask to ply for visualization
                os.system('../Vox2Mesh/main --in ' + mask_file_path + ' --out ' + mask_file_path + '.ply')
                # Save the point cloud segment as pcd
                o3d.io.write_point_cloud(pc_file_path, point_cloud)

        if pc:
            room_pointcloud_rgb_path = os.path.join(room_path, room_name) + "_pc_rgb.pcd"
            room_pointcloud_label_path = os.path.join(room_path, room_name) + "_pc_sem.pcd"
            o3d.io.write_point_cloud(room_pointcloud_rgb_path, room_points)
            o3d.io.write_point_cloud(room_pointcloud_label_path, room_points_label_rgb)

        room_points_label_tree = o3d.geometry.KDTreeFlann(room_points_label)

        # area_points.append(point_cloud)

        # write the object proposal for interested objects together
        if bbox:
            verts_bbox_all = np.asarray(verts_bbox_all,
                                    dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
            faces_bbox_all = np.asarray(faces_bbox_all, dtype=[('vertex_indices', 'i4', (3,))])
            bbox_all_plydata = PlyData(
                    [PlyElement.describe(verts_bbox_all, 'vertex', comments=['vertices']),
                     PlyElement.describe(faces_bbox_all, 'face')], comments=['faces'])

            bbx_all_file_path = os.path.join(room_path, room_name) + "_bbox_all.ply"
            print("gt bbx of the room saved:", bbx_all_file_path)
            with open(bbx_all_file_path, mode='wb') as f:
                PlyData(bbox_all_plydata).write(f)

            verts_bbox_obj = np.asarray(verts_bbox_obj,
                                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                                               ('blue', 'u1')])
            faces_bbox_obj = np.asarray(faces_bbox_obj, dtype=[('vertex_indices', 'i4', (3,))])
            bbox_obj_plydata = PlyData(
                [PlyElement.describe(verts_bbox_obj, 'vertex', comments=['vertices']),
                 PlyElement.describe(faces_bbox_obj, 'face')], comments=['faces'])
            bbx_obj_file_path = os.path.join(room_path, room_name) + "_bbox_obj.ply"
            #print("gt bbx of the room saved:", bbx_obj_file_path)
            with open(bbx_obj_file_path, mode='wb') as f:
                PlyData(bbox_obj_plydata).write(f)

        # Reconstruct the mesh of the room from the point cloud and assign the semantic label to mesh vertices
        if mesh:
            print("Normal estimation")
            room_points.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.08, max_nn=50))

            print("Mesh reconstruction from point cloud (Possion)")
            room_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(room_points, depth=10)

            vertices_to_remove = densities < np.quantile(densities, 0.08)
            room_mesh.remove_vertices_by_mask(vertices_to_remove)

            print(
                f'Original mesh has {len(room_mesh.vertices)} vertices and {len(room_mesh.triangles)} triangles'
            )

            mesh_sim_voxel_size = 0.025
            #print(f'voxel_size = {mesh_sim_voxel_size:e}')
            room_mesh = room_mesh.simplify_vertex_clustering(
                voxel_size=mesh_sim_voxel_size,
                contraction=o3d.geometry.SimplificationContraction.Average)

            print(
                f'Simplified mesh has {len(room_mesh.vertices)} vertices and {len(room_mesh.triangles)} triangles'
            )

            room_mesh_file = os.path.join(room_path, room_name) + "_mesh.ply"
            # print(mesh)
            o3d.io.write_triangle_mesh(room_mesh_file, room_mesh, write_vertex_normals=False, print_progress=True)

            # import the labeled scan
            print("Assign vertex label")
            with open(room_mesh_file, 'rb') as f:
                mesh_scan_label = PlyData.read(f)

                vertices = mesh_scan_label.elements[0]
                faces = mesh_scan_label.elements[1]

                # Create the new vertex data with label dtype
                vertices_label = np.empty(len(vertices.data), vertices.data.dtype.descr + [('label', 'i4')])
                for name in vertices.data.dtype.fields:
                    vertices_label[name] = vertices[name]
                vertices_label['label'] = 0 # figure out how to add the actual label

                # Recreate the PlyElement instance
                vertices = PlyElement.describe(vertices_label, 'vertex')

                for v in vertices:
                    vertex_pt =  np.array([v[0], v[1], v[2]])
                    [k, idx, _] = room_points_label_tree.search_knn_vector_3d(vertex_pt, 1)
                    label = room_points_label.colors[idx[0]][0] * 100
                    v[6] = int(label)

                # Recreate the PlyData instance
                mesh_scan_label = PlyData([vertices, faces], text=True)
                #print(vertices)
                mesh_scan_label.write(room_mesh_file)

            #o3d.visualization.draw_geometries([room_mesh])

        #o3d.visualization.draw_geometries([room_points])

    # o3d.visualization.draw_geometries(area_points)

if __name__ == "__main__":

    # example
    base_path = "/media/edward/SeagateNew/1_data/2d3ds/noXYZ_area_4_no_xyz/area_4/3d"
    area_id = "Area_4"

    # Firstly you need to convert the original mat file "pointcloud.mat" to a ealier version format that can directly imported by python
    # You can simply import it into Matlab and exported it (given the data is not too large the format or you need to split it into two parts)
    mat_file_name = "pointcloud_new.mat"

    process_area(base_path, area_id, mat_file_name, voxel=True, pc=True, bbox=True, mesh=True)
