import sys
assert sys.version_info >= (3, 5)

import os
import pathlib
import json
import open3d as o3d

if __name__ == '__main__':

    base_folder = "/cluster/project/infk/courses/3d_vision_21/group_14/1_data"
    
    model_base_folder = os.path.join(base_folder, "ShapeNetCore.v2")

    pcd_base_folder = os.path.join(base_folder, "ShapeNetCore.v2-pcd")

    model_pool_json = os.path.join(base_folder, "ScanCADJoint", "model_pool_large.json")

    down_point: int = 10000

    with open(model_pool_json) as f:
        model_pool = json.load(f)
            
        for cat, cat_model_pool in model_pool.items():
            print("------------ Begin Category [", cat, "] ------------")
            
            pathlib.Path(os.path.join(pcd_base_folder, cat)).mkdir(parents=True, exist_ok=True)
            
            for cad_str in cat_model_pool:
                cad_path = os.path.join(model_base_folder, cad_str, "models", "model_normalized.obj")
                cad_mesh = o3d.io.read_triangle_mesh(cad_path)

                cad_pcd = cad_mesh.sample_points_uniformly(down_point)
                #o3d.visualization.draw_geometries([cad_pcd])
                pcd_path = os.path.join(pcd_base_folder, cad_str + ".pcd")
                o3d.io.write_point_cloud(pcd_path, cad_pcd)

                print("Output [", pcd_path, "]")
