import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)
import json
import numpy as npvisual_pcd
import cv2
import open3d as o3d
import numpy as np


from transform_utils import read_loc, read_calib, visual_pcd, create_depth_map_from_lidar, render_depth_on_image, \
    back_project_pixel_to_lidar, get_coord


pcd_class_dict = {
    0: 'default',
    1: 'drivable_space',
    2: 'road_curb',
    3: 'wall',
    4: 'ceiling',
    5: 'noise',
    # 6: 'pathway',
    # 7: 'wall_fence'
}

cam_topics = ["cam_around_front",
              "cam_around_right",
              "cam_around_back",
              "cam_around_left",
              # "cam_front",
              # "cam_front_right",
              # "cam_front_left",
              # "cam_back",
              # "cam_back_left",
              # "cam_back_right"
              ]

cam_calib_dict = {
    "cam_around_front": "CAM__a_front",
    "cam_around_right": "CAM__a_right",
    "cam_around_back": "CAM__a_back",
    "cam_around_left": "CAM__a_left",
    "cam_front": "CAM__front",
    "cam_front_right": "CAM__front_right",
    "cam_front_left": "CAM__front_left",
    "cam_back": "CAM__back",
    "cam_back_left": "CAM__back_left",
    "cam_back_right": "CAM__back_right"
}

if __name__ == "__main__":
    data_dir = './Datasets/sjt20/annotation_pc_seman/2024-12-20-09-32-37_hangzhouwan-diku_UG_n_i_1131_10V__label'
    calib_path = os.path.join(data_dir, 'calib.json').replace('/annotation_pc_seman/', '/annotation/')
    location_path = os.path.join(data_dir, 'loc.json').replace('/annotation_pc_seman/', '/annotation/')
    visualize = True

    # pcd_clips = read_semantic_pcd_clips(data_dir)
    cam_calib = read_calib(calib_path, cam_calib_dict)
    loc_clips = read_loc(location_path)

    for dirpath, dirname, filenames in os.walk(data_dir):
        for scene_name in sorted(dirname):
            scene_dir = os.path.join(dirpath, scene_name)

            print(os.listdir(scene_dir))
            pcd_path = os.path.join(scene_dir, f'{scene_name}.pcd')

            # pcd = o3d.io.read_point_cloud(pcd_path)
            pcd = o3d.t.io.read_point_cloud(pcd_path) # tensor
            # points = pcd.point['positions'].numpy()
            points = pcd.point['positions'].numpy()  # 获取点的坐标 (N x 3)
            try:
                labels = pcd.point['label'].numpy()
            except:
                labels = pcd.point['dt_label'].numpy()

            assert points.shape[0] == labels.shape[0], "points and label shape diff！"

            # visual_pcd(points, labels, pcd_class_dict)

            # visual_pcd(points, labels)
            for cam_name in cam_topics:
                # print(cam_name)
                img_name = f"{scene_name}{cam_calib_dict[cam_name].replace('CAM', '')}.jpg"
                img_path = os.path.join(dirpath, scene_name, img_name).replace('/annotation_pc_seman/', '/annotation/')
                img = cv2.imread(img_path)
                resolution = img.shape

                # get calib param of camera
                K, D = cam_calib[cam_name]['K'], cam_calib[cam_name]['D']
                T = cam_calib[cam_name]['T']  # lidar2cam

                # ------------------- pixel-coord to ego-coord-------------------
                # fov_pixel, fov_label = project_pcd_with_cv(points, labels, K, D, T, resolution=resolution)
                depth_map = create_depth_map_from_lidar(points, K, D, T, resolution)

                pixel_coord = get_coord(resolution)

                vcs_points = back_project_pixel_to_lidar(pixel_coord, depth_map, K=K, D=D, trans_lidar2cam=T)

                if visualize:
                    render_image = render_depth_on_image(img=img,
                                                         depth=np.array(depth_map.copy()).astype(np.int32),
                                                         depth_min=0, depth_max=255, transparency=1.0)
                    cv2.imshow('depth_map', render_image)
                    cv2.waitKey()

                    labels_origin = np.ones((points.shape[0], 1)) * 1
                    labels_new = np.ones((vcs_points.shape[0], 1)) * 3

                    merge_points = np.concatenate((points, vcs_points), axis=0)
                    merge_labels = np.concatenate((labels_origin, labels_new), axis=0)

                    visual_pcd(merge_points, merge_labels, pcd_class_dict)
