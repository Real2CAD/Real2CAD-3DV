# Script used to visualize 2d3ds dataset

import scipy.io as sio
import open3d as o3d
import colorsys
import numpy as np

def generate_semantic_color(area):
    category = []
    for space in area.Disjoint_Space:
        for object in space.object:
            category.append(object.name.split("_")[0])

    category = list(set(category))
    color_dict = {}

    for x in range(len(category)):
        HSV = [x * 1.0 / len(category), 0.5, 0.5]
        RGB = colorsys.hsv_to_rgb(*HSV)
        color_dict[category[x]] = RGB

    return color_dict


def visualize(raw_data, area, color, bbox=True, voxel=False):

    area = raw_data[area]
    area_points = []
    for space in area.Disjoint_Space:
        for object in space.object:
            points = object.points
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)

            # Semantic color
            if color == "semantic":
                color_dict = generate_semantic_color(area)
                point_cloud.paint_uniform_color(color_dict.get(object.name.split("_")[0]))
            else:
                # RGB color
                point_cloud.colors = o3d.utility.Vector3dVector(object.RGB_color / 255)

            if bbox:
                Bbox = np.reshape(object.Bbox, (6, 1))
                boundingBox = o3d.geometry.AxisAlignedBoundingBox(Bbox[:3], Bbox[3:])
                boundingBox.color = (1, 1, 1)
                area_points.append(boundingBox)

            # Voxelization
            if voxel:
                point_cloud = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud,
                                                                         voxel_size=0.05)

            area_points.append(point_cloud)

    o3d.visualization.draw_geometries(area_points)

if __name__ == "__main__":
    raw_data = sio.loadmat('pointcloud.mat', squeeze_me=True, struct_as_record=False)
    visualize(raw_data, 'Area_3', 'semantic', bbox=True, voxel=False)

