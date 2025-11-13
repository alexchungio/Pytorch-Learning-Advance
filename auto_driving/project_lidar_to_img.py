import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)
import json
import cv2
import open3d as o3d
import yaml
import glob
import matplotlib.pyplot as plt
import numpy as np

from transform_utils import read_loc, read_calib, visual_pcd, project_pcd_with_cv, project_pcd_with_custom


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


def get_color_map(class_label):
    """
    return: RGB
    """
    # unique_labels = np.unique(labels)  # 获取所有类别
    unique_labels = np.array(list(class_label.keys()))
    colormap = plt.get_cmap("tab20")  # 使用 matplotlib 的 tab20 颜色映射
    # colors = colormap(filtered_labels / unique_labels.max()).reshape(filtered_labels.shape[0], 4)[:, :3]  # 归一化标签并分配颜色 (取 RGB)
    colors_map = colormap(unique_labels / unique_labels.max()).reshape(unique_labels.shape[0], 4)[:, :3]
    colors_map = {label: tuple(int(round(c * 255)) for c in color) for label, color in zip(unique_labels, colors_map)}

    return colors_map


if __name__ == "__main__":
    data_dir = './Datasets/sjt20/annotation_pc_seman/2024-12-20-09-32-37_hangzhouwan-diku_UG_n_i_1131_10V__label'
    calib_path = os.path.join(data_dir, 'calib.json').replace('/annotation_pc_seman/', '/annotation/')
    location_path = os.path.join(data_dir, 'loc.json').replace('/annotation_pc_seman/', '/annotation/')

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

            # generate_color
            # color_map = get_color_map(pcd_class_dict)
            color_map = get_color_map(pcd_class_dict)
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
                T = cam_calib[cam_name]['T']
                # ------------------- ego-coord to pixel-coord -------------------
                # fov_pixel, fov_label = project_pcd_with_cv(points, labels, K, D, T, resolution=resolution)
                fov_pixel, fov_label = project_pcd_with_custom(points, labels, K, D, T, resolution=resolution)

                # visual point
                for pixel, label in zip(fov_pixel, fov_label):
                    y, x = pixel
                    # import pdb;pdb.set_trace()
                    img[x, y] = color_map[label[0]][::-1]

                cv2.imshow('image', img)
                cv2.waitKey()