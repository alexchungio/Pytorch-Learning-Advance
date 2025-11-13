import os
import yaml
import numpy as np
import open3d as o3d
import json
import warnings
import tqdm
import copy
import cv2
import matplotlib.pyplot as plt


__all__ = ['read_calib', 'read_loc', 'read_semantic_pcd_clips', 'project_pcd_with_cv', 'project_pcd_with_custom',
           'visual_pcd', 'get_color_map', 'pcd_class_dict', 'visual_occ', 'back_project_pixel_to_lidar',
           'create_depth_map_from_lidar', 'render_depth_on_image']


pcd_class_dict = {
    0: 'default',
    1: 'drivable_space',
    2: 'road_curb',
    3: 'wall',
    4: 'ceiling',
    5: 'noise',
    # 6: 'pathway',
    # 7: 'roadboundary'
    6: 'car',
    7: 'person',
    8: 'cyclist',
    9: 'pillar',

}

pcd_colors = np.array(
    [[31, 119, 194],  # 0 undefined  black
    [0, 207, 191],  # 1 drivable_space
    [75, 0, 75],  # 2 road_curb
    [222, 184, 135],  # 3 wall
    [44, 160, 44],  # 4 ceiling
    [255, 0, 0],  # 5 noise
    # [75, 0, 75]  # 6 pathway
    [255, 158, 0],  # 6 car
    [165, 42, 42],  # 7 person
    [112, 128, 114],  # 8 cyclist
    [158, 218, 229],  # pillar
    [0, 0, 0]]  # abnormal noise
)


occ_colors = np.array(
    [
        [0, 0, 0, 255],
        [0, 175, 0, 255],  # vegetation           green
        [255, 120, 50, 255],  # barrier              orangey
        [255, 192, 203, 255],  # bicycle              pink
        [255, 255, 0, 255],  # bus                  yellow
        [0, 150, 245, 255],  # car                  blue
        [0, 255, 255, 255],  # construction_vehicle cyan
        [200, 180, 0, 255],  # motorcycle           dark orange
        [255, 0, 0, 255],  # pedestrian           red
        [255, 240, 150, 255],  # traffic_cone         light yellow
        [135, 60, 0, 255],  # trailer              brown
        [160, 32, 240, 255],  # truck                purple
        [255, 0, 255, 255],  # driveable_surface    dark pink
        # [175,   0,  75, 255],       # other_flat           dark red
        [139, 137, 137, 255],
        [75, 0, 75, 255],  # sidewalk             dard purple
        [150, 240, 80, 255],  # terrain              light green
        [230, 230, 250, 255],  # manmade              white
        [0, 175, 0, 255],  # vegetation           green
        [0, 255, 127, 255],  # ego car              dark cyan
        [255, 99, 71, 255],
        [0, 191, 255, 255]
    ]
).astype(np.uint8)

CLASSES_OD={
    'car': 'car',
    'truck': 'truck',
    'bus' : 'bus',
    'engineering_vehicle' : 'engineering_vehicle',
    'person':'person',
    'pillar':'pillar',
    'bicycle':'cyclist',
    'electromobile':'cyclist',
    'motorcycle':'cyclist',
    'barricade__traffic_cone':'traffic_cone',
    'barricade__noparking_pillar':'noparking_pillar',
    'barricade__water_horse':'water_horse',
    'barricade__gate':'gate',
    'barricade__anti_collision_bucket':'anti_collision_bucket',
    'barricade__noparking_board':'noparking_board',
    'stroller': 'stroller',
    'Shopping_cart': 'Shopping_cart'
}

class_map_od_to_occ = {
    'car': 6,
    'truck': 6,
    'bus': 6,
    'engineering_vehicle': 6,
    'person': 7,
    'cyclist': 8,
    'pillar': 9,
    'traffic_cone': 0,
    'noparking_pillar': 0,
    'water_horse': 0,
    'gate': 0,
    'anti_collision_bucket': 0,
    'noparking_board': 0,
    'stroller': 0,
    'Shopping_cart': 0
}


class_map_thresh_points_in_box = {
    'car': 200,
    'truck': 200,
    'bus': 200,
    'engineering_vehicle': 200,
    'person': 50,
    'cyclist': 50,
    'pillar': 0,
    'traffic_cone': 10,
    'noparking_pillar': 10,
    'water_horse': 20,
    'gate': 100,
    'anti_collision_bucket': 30,
    'noparking_board': 10,
    'stroller': 20,
    'Shopping_cart': 20,
    'other_non_vehicle__other_non_vehicle': 10
}


# ################################# load data ##################################
def read_calib(calib_path, cam_calib_dict):
    """
    read camera intra, extra and distort param
    Args:
        calib_path:

    Returns:

    """
    if not os.path.exists(calib_path):
        calib_path = calib_path.replace('calib.json', 'cali.json')

    with open(calib_path, "r", encoding="utf-8") as file:
        calib_info = yaml.safe_load(file)

    cam_calib = {}
    for cam_name, calib_name in cam_calib_dict.items():
        calib_param = calib_info[calib_name]
        K = np.array(calib_param['intr']).reshape(3, 3)
        D = np.array(calib_param['D'])[:4]
        T = np.array(calib_param['extr']).reshape(4, 4)
        cam_calib[cam_name] = {
            'K': K,
            'D': D,
            'T': T
        }

    return cam_calib


def read_loc(loc_path):
    """
    read location of frame
    Args:
        loc_path:

    Returns:

    """
    if not os.path.exists(loc_path):
        loc_path = loc_path.replace('/annotation', '/sensor')
    with open(loc_path) as f:
        loc_info = json.load(f)
    all_loc = {}
    for scene_name, loc in loc_info.items():
        all_loc[scene_name] = np.array(loc).reshape(4, 4)

    return all_loc


def remove_invalid_points(pcd):
    invalid_index = np.where(np.isfinite(pcd.point.positions.numpy()).all(axis=1))
    # pcd = pcd.select_by_index(invalid_index)
    pcd.point.positions = pcd.point.positions[invalid_index]
    pcd.point.dt_label = pcd.point.dt_label[invalid_index]
    pcd.point.intensity = pcd.point.intensity[invalid_index]
    pcd.point.dt_ID = pcd.point.dt_ID[invalid_index]
    if 'colors' in dir(pcd.point):
        pcd.point.colors = pcd.point.colors[invalid_index]
    pcd.point.timestamp = pcd.point.timestamp[invalid_index]

    return pcd


def read_semantic_pcd_clips(pcd_dir, exchange_label=False):
    """
    read semantic pcd in a clip and filter with location
    Args:
        pcd_dir:
        exchange_label: repair error label

    Returns:

    """
    pass

    # get frame name
    frame_list = [frame_name for frame_name in os.listdir(pcd_dir) if os.path.isdir(os.path.join(pcd_dir, frame_name))]

    # load loc info
    loc_path = os.path.join(pcd_dir, 'loc.json')
    if not os.path.exists(loc_path):
        loc_path = loc_path.replace('/annotation_pc_seman/', '/annotation/')
    all_loc = read_loc(loc_path)

    # remove frame do not have loc info
    frame_list = [frame_name for frame_name in frame_list if frame_name in all_loc.keys()]
    frame_list = sorted(frame_list)

    pcd_clips = {}
    pbar = tqdm.tqdm(frame_list)
    for frame_name in pbar:
        pcd_path = os.path.join(pcd_dir, frame_name, f'{frame_name}.pcd')
        pbar.set_postfix_str(f'loading pcd data from {frame_name}.pcd')
        if not os.path.exists(pcd_path):
            warnings.warn(f'{pcd_path} do not exist')
            continue
        else:
            # pcd_content = pypcd.PointCloud.from_path(pcd_path)
            pcd_info = o3d.t.io.read_point_cloud(pcd_path)
            pcd_info = remove_invalid_points(pcd_info)

            points = pcd_info.point['positions'].numpy()  # 获取点的坐标 (N x 3)
            try:
                labels = pcd_info.point['label'].numpy()
            except:
                labels = pcd_info.point['dt_label'].numpy()

            # bd label class is error about wall and noise and need to exchange
            if exchange_label:
                noise_mask = (labels == 3)
                wall_mask = (labels == 5)
                labels[noise_mask] = 5
                labels[wall_mask] = 3

            assert points.shape[0] == labels.shape[0], "points and label shape diff！"

            # concat pcd and label
            pcd_label = np.hstack((points, labels))

            pcd_clips[frame_name] = pcd_label

    return pcd_clips


def get_obj_type(obj_type):
    obj_type_s = obj_type.split('__')
    if obj_type_s[0] in CLASSES_OD:
        return CLASSES_OD[obj_type_s[0]]
    elif obj_type in CLASSES_OD:
        return CLASSES_OD[obj_type]
    else:
        return 'invalid'


def read_od(label_dir):
    """
    static_od & untrack_od[0:8]: [[x, y, z, x_size, y_size, z_size, yaw, track_id, od_type]]
    dynamic_od[0:10]:             [[x, y, z, x_size, y_size, z_size, yaw, track_id, od_type, v_0, v_1]]
    (x,y,z) locate in box center
    """
    label_lists = sorted(os.listdir(label_dir))
    all_dynamic_od = {}
    all_static_od = {}
    all_untracked_od = {}
    for od_folder in label_lists:
        if not os.path.isdir(os.path.join(label_dir, od_folder)):
            continue
        od_label_file = os.path.join(label_dir, od_folder, od_folder + ".json")
        with open(od_label_file) as f:
            od_content = json.load(f)

        # TODO filter box of low number points in box
        # label_semantic_dir = label_dir.replace("/annotation", "/annotation_pc_seman")
        # pcd_path = os.path.join(label_semantic_dir, od_folder, od_folder+'.pcd')
        # if not os.path.exists(pcd_path):
        #     warnings.warn(f'{pcd_path} do not exist, load {od_folder} od failed')
        #     continue
        # pcd_info = o3d.t.io.read_point_cloud(pcd_path)
        # pcd_info = remove_invalid_points(pcd_info)
        # points = pcd_info.point['positions'].numpy()  # 获取点的坐标 (N x 3)
        # points = points[points[:, 2] <= 1.8

        # parse od data
        dynamic_od_curr = []
        static_od_curr = []
        untracked_od_curr = []
        for single_od in od_content:
            if get_obj_type(single_od['obj_type']) == 'invalid':
                continue
            x = float(single_od['psr']['position']['x'])
            y = float(single_od['psr']['position']['y'])
            z = float(single_od['psr']['position']['z'])
            x_size = float(single_od['psr']['scale']['x'])
            y_size = float(single_od['psr']['scale']['y'])
            z_size = float(single_od['psr']['scale']['z'])
            yaw = float(single_od['psr']['rotation']['z'])
            od_type = int(class_map_od_to_occ[get_obj_type(single_od['obj_type'])])
            id = int(single_od['obj_id'])
            if od_type in [6, 7, 8]:
                # judge dynamic od with velocity
                if 'velocity' in single_od:
                    vel = single_od['velocity']
                    dynamic_od_curr.append(
                        [x, y, z, x_size, y_size, z_size, yaw, id, od_type, float(vel[0]), float(vel[1])]
                    )
                elif not (id ==-1):
                    static_od_curr.append(
                        [x, y, z, x_size, y_size, z_size, yaw, id, od_type, 0, 0]
                    )
                else:
                    untracked_od_curr.append(
                        [x, y, z, x_size, y_size, z_size, yaw, -1, od_type, 0, 0]
                    )
            else:
                static_od_curr.append(
                    [x, y, z, x_size, y_size, z_size, yaw, id, od_type, 0, 0]
                )

        # if len(dynamic_od_curr)==0 and len(static_od_curr)==0 and len(untracked_od_curr)==0 :
        #     continue

        all_dynamic_od[od_folder] = np.array(dynamic_od_curr)
        all_static_od[od_folder] = np.array(static_od_curr)
        all_untracked_od[od_folder] = np.array(untracked_od_curr)

    return all_dynamic_od, all_static_od, all_untracked_od


def read_pcd_loc_calib(label_dir, label_semantic_dir,  cam_calib_dict):
    """
    read pcd, location and camera calibration param of a clips
    Args:
        label_dir:
        label_semantic_dir:
        cam_calib_dict:

    Returns:

    """

    calib_path = os.path.join(label_dir, 'calib.json')
    location_path = os.path.join(label_dir, 'loc.json')
    if not os.path.exists(calib_path):
        calib_path = calib_path.replace('/annotation/', '/annotation_pc_seman/')
    if not os.path.exists(location_path):
        location_path = location_path.replace('/annotation/', '/annotation_pc_seman/')

    pcd_clips = read_semantic_pcd_clips(label_semantic_dir)
    all_dynamic_od, all_static_od, all_untracked_od = read_od(label_dir)
    loc_clips = read_loc(location_path)
    cam_calib = read_calib(calib_path, cam_calib_dict)

    return pcd_clips, all_dynamic_od, all_static_od, all_untracked_od, loc_clips, cam_calib


# ################################# ego2cam ###########################

def get_distort_v1(x_norm, y_norm, D):
    '''
    get r_dist gradient function
    fisheye distort model: r = theta * (1 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8)
    '''
    k_1, k_2, k_3, k_4 = D
    # # calculate distance to zero pint
    r = np.sqrt(x_norm ** 2 + y_norm ** 2)
    # # calculate theta(from optical axis to the point)
    theta = np.arctan(r)
    # theta = np.pi - theta  # !!!

    theta_2 = theta * theta
    theta_4 = theta_2 * theta_2
    theta_6 = theta_4 * theta_2
    theta_8 = theta_6 * theta_2
    theta_d = theta * (1 + k_1*theta_2 + k_2*theta_4 + k_3*theta_6 + k_4*theta_8)

    # r<0, maintain origin direction
    r[r == 0] = 1e-8
    scale = theta_d / r
    x_dist = x_norm * scale
    y_dist = y_norm * scale

    return x_dist, y_dist


def get_distort_v2(x_norm, y_norm, D):
    '''

    '''
    k_1, k_2, k_3 = D[0], D[1], 0 if len(D) < 5 else D[4]  # radial distort and
    p_1, p_2 = D[2], D[3]  # tangential distort param

    r_2 = x_norm ** 2 + y_norm **  2
    r_4 = r_2 * r_2
    r_6 = r_4 * r_2

    # calculate radial param
    cdist = 1.0 + k_1 * r_2 + k_2 * r_4 + k_3 * r_6

    x_dist = x_norm * cdist + 2 * p_1 * x_norm * y_norm + p_2 * (r_2 + 2 * x_norm ** 2)
    y_dist = y_norm * cdist + 2 * p_2 * x_norm * y_norm + p_1 * (r_2 + 2 * y_norm ** 2)

    return x_dist, y_dist


def lidar2cam(points, trans_lidar2cam):
    """
    convert point from lidar-coord to camara-coord
    Args:
        points: (N, 3)
        trans_lidar2cam: (4, 4)

    Returns:
        points_cam: (N, 3)

    """


    points_cam = copy.deepcopy(points)

    points_cam = points_cam.T
    one_array = np.ones((points_cam.shape[0], 1), dtype=np.float64)
    points_cam = np.hstack((points_cam, one_array))
    points_cam = trans_lidar2cam @ points_cam.T
    points_cam = points_cam.T[:, :3]

    return points_cam


def cam2pixel(points, K, D):
    """
    convert point from camera coord to pixel coord
    Args:
        point:
        K: (3, 3)
        D: (4, )

    Returns:
        pixels: (N, 2)

    """
    points_cam = points.copy()

    # cal normalize coord
    x_norm = points_cam[..., 0] / abs(points_cam[..., 2])  # x/z
    y_norm = points_cam[..., 1] / abs(points_cam[..., 2])  # y/z
    #
    # # calculate distort param
    x_dist, y_dist = get_distort_v1(x_norm, y_norm, D)
    # # x_dist, y_dist = get_distort_v2(x_norm, y_norm, D)
    #
    # # convert to image-coord with intra-param
    fx, cu, fy, cv = K[0, 0], K[0, 2], K[1, 1], K[1, 2]
    u = fx * x_dist + cu
    v = fy * y_dist + cv
    # u = fx * x_norm + cu
    # v = fy * y_norm + cv
    # generate pixel
    pixels = np.hstack((u.reshape(-1, 1), v.reshape(-1, 1)))

    return pixels


def project_pcd_with_custom(points, labels, K, D, T, resolution):
    """
    project pcd from lidar to pixel
    Args:
        points:
        labels:
        K:
        D:
        T:
        resolution:

    Returns:

    """

    points_cam = lidar2cam(points, trans_lidar2cam=T)

    # filter behind point of camera (restrict with fov)
    fov_mask = points_cam[:, 2] > 0
    points_cam = points_cam[fov_mask]
    labels = labels[fov_mask]

    # cam to pixel
    pixels = cam2pixel(points_cam, K=K, D=D)

    # filter point with image shape
    valid = (pixels[:, 0] >= 0) & (pixels[:, 0] < resolution[1]) & (pixels[:, 1] >= 0) & (
            pixels[:, 1] < resolution[0])

    fov_pixel = pixels[valid].astype(np.int32)
    fov_label = labels[valid].astype(np.int32)

    return fov_pixel, fov_label


def project_pcd_with_cv(points, labels, K, D, T, resolution):
    """
    project with opencv interface
    Args:
        points:
        labels:
        K:
        D:
        T:
        resolution:

    Returns:

    """
    points_cam = copy.deepcopy(points)

    # calculate fov
    # filter behind point of camera (restrict with fov)
    points_tmp = lidar2cam(points, trans_lidar2cam=T)
    fov_mask = points_tmp[:, 2] > 0
    # fov_mask = (labels[:, 0] == 1) & (points_tmp[:, 2] > 0) & (points_tmp[:, 2] < 5)
    points_cam = points_cam[fov_mask]
    labels = labels[fov_mask]

    # world-coord -> camera-coord -> image-coord
    rvec = cv2.Rodrigues(T[:3, :3])[0]
    tvec = T[:3, 3]
    if D is None:
        points_cam = cv2.projectPoints(points_cam.reshape(-1, 1, 3), rvec, tvec, K[:3, :3], D)[0].reshape(-1, 2)
    else:
        points_cam = cv2.fisheye.projectPoints(points_cam.reshape(-1, 1, 3), rvec, tvec, K[:3, :3], D)[
            0].reshape(-1, 2)

    valid = (points_cam[:, 0] >= 0) & (points_cam[:, 0] < resolution[1]) & (points_cam[:, 1] >= 0) & (
            points_cam[:, 1] < resolution[0])

    fov_pixel = points_cam[valid].astype(np.int32)[:, :2]
    fov_label = labels[valid].astype(np.int32)

    return fov_pixel, fov_label


# ################################ convert pixel-coord to lidar-coord ###########################################

def get_coord(resolution):
    """
    Args:
        resolution: (height, width)

    Returns:
        pixel_coord (N, 2)

    """
    height, width = resolution[:2]
    u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
    pixel_coord = np.column_stack((u_coords.ravel(), v_coords.ravel())).astype(np.float32)

    return pixel_coord


def get_theta_with_newton_iter(r_dist, D, max_iter=20, tolerance=1e-8):
    """
    get unit ray angle of per pixel in camera coord by fisheye reverse-distort
    Args:
       r_dist: (N, 1)
       D: (4, )
       max_iter: int
       tolerance: float

    Returns:
        theta: (N, 1)
    """
    k1, k2, k3, k4 = D
    distort_model = lambda theta: theta * (1 + k1*np.power(theta, 2)
                                             + k2 * np.power(theta, 4)
                                             + k3 * np.power(theta, 6)
                                             + k4 * np.power(theta, 8))

    gradient_distort_model = lambda theta: 1 + 3 * k1 * np.power(theta, 2) \
                                             + 5 * k2 * np.power(theta, 4) \
                                             + 7 * k3 * np.power(theta, 6) \
                                             + 9 * k4 * np.power(theta, 8)

    theta = r_dist.copy()  # initial theta value
    for _ in range(max_iter):
        # calculate the distortion radius based on distort model
        r_dist_theta = distort_model(theta)
        r_dist_grad = gradient_distort_model(theta)
        # calculate error
        error = r_dist_theta - r_dist

        theta_new = theta - error / r_dist_grad

        if np.max(np.abs((theta_new - theta))) < tolerance:
            break
        theta = theta_new

    return theta


def pixel2cam(coord, depth,  K, D):
    """
    pixel-coord to  cam-coord
    Args:
        coord: (N,2)
        depth: (height, width)
        K: (3, 3)
        D: (4, 0)

    Returns:
        cam_coord: (N, 3)
    """
    fx, cx, fy, cy = K[0, 0], K[0, 2], K[1, 1], K[1, 2]

    u, v = coord[:, 0], coord[:, 1]

    # step 1: cal normalized image coordinate after distortion
    x_dist = (u - cx) / fx
    y_dist = (v - cy) / fy
    # step 2: cal the radial distance after distortion
    r_dist = np.sqrt(x_dist ** 2 + y_dist ** 2)  # radial distance

    # cal the angel of incidence by distortion model
    theta = get_theta_with_newton_iter(r_dist, D)

    # cal the undistorted radial distance of normalize plane
    # (the distance from normalized plane to the optical center is 1)
    r = np.tan(theta)

    # cal undistorted normalized coordinate
    scale = np.where(r_dist<1e-8, 1, r/r_dist)
    x_norm = x_dist * scale
    y_norm = y_dist * scale
    z_norm = np.ones_like(x_norm)
    coord_norm = np.column_stack((x_norm, y_norm, z_norm))

    # get camera coordinate with depth
    depth = depth.reshape(-1, 1)
    cam_coord = coord_norm * depth

    return cam_coord


def cam2lidar(points, trans_cam2lidar):
    """
    camera-coord to lidar-coord
    Args:
        points: (N, 3)
        trans_cam2lidar: (4,4)

    Returns:
        points_vcs: (N, 3)
    """

    points_cam = copy.deepcopy(points)

    points_cam = points_cam
    one_array = np.ones((points_cam.shape[0], 1), dtype=np.float64)
    points_cam = np.hstack((points_cam, one_array))
    points_cam = trans_cam2lidar @ points_cam.T
    points_vcs = points_cam.T[:, :3]

    return points_vcs


def back_project_pixel_to_lidar(coords, depth, K, D, trans_lidar2cam):
    """

    Args:
        coords: coord of pixel (N, 2)
        depth: depth correspond to pixel (height, width)
        K: (3X3)
        D: (4,)
        T: (4X4)

    Returns:
        vsc_coord: (N, 3)

    """

    trans_cam2lidar = np.linalg.inv(trans_lidar2cam)
    cam_coord = pixel2cam(coords, depth=depth, K=K, D=D)
    vsc_coord = cam2lidar(cam_coord, trans_cam2lidar=trans_cam2lidar)

    return vsc_coord


def create_depth_map_from_lidar(lidar_points, K, D, trans_lidar2cam, img_shape):
    """

    Args:
        lidar_points: (N, 2)
        K: (3, 3)
        D: (4, )
        trans_lidar2cam: (4, 4)
        img_shape: (height, width)

    Returns:
        depth_map: (height, width)

    """
    height, width = img_shape[:2]

    # vcs -> cam
    points_cam = lidar2cam(lidar_points, trans_lidar2cam)
    fov_mask = points_cam[:, 2] > 0
    points_cam = points_cam[fov_mask]

    # cam to pixel
    pixels = cam2pixel(points_cam, K=K, D=D)  # (u, v)

    # filter point with image shape
    valid_mask = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (
                   pixels[:, 1] < height)
    valid_pixel = pixels[valid_mask].astype(np.int32)

    depth_map = np.zeros((height, width), dtype=np.float32)
    depth_valid = points_cam[valid_mask][:, 2]

    depth_map[valid_pixel[:, 1], valid_pixel[:, 0]] = depth_valid

    return depth_map


# ################################ visualize ############################################

def rotation_3d_in_aixs(corners, yaw):
    rot_sin = np.sin(yaw)
    rot_cos = np.cos(yaw)
    ones    = np.ones_like(rot_sin)
    zeros   = np.zeros_like(rot_sin)

    rot_mat_T = np.stack([
        np.stack([rot_cos, rot_sin, zeros]),
        np.stack([-rot_sin, rot_cos,  zeros]),
        np.stack([zeros,   zeros,    ones])
    ])

    return np.einsum('aij, jka->aik', corners, rot_mat_T)


def get_corners(boxes):
    if len(boxes)<=0:
        return []

    # recover bbox position to center
    boxes[:, 2] = boxes[:, 2] + 0.1 + boxes[:, 5] / 2

    corners_norm = np.stack(np.unravel_index(np.arange(8), [2]*3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = boxes[:, 3:6].reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
    corners = rotation_3d_in_aixs(corners, boxes[:, -1])
    corners += boxes[:, :3].reshape(-1, 1, 3)
    return corners


def visual_pcd(points, labels, class_label, boxes=None, boxes_color=(1.0, 0, 0), window_name='pcd_vis'):
    """
    point: (x, y, z)
    class_label: (label)
    boxes: (x, y, z, l, w, h, yaw)
    """
    # filter point with z axis
    filtered_mask = points.copy()[:, 2] <= 2.5
    filtered_points = points.copy()[filtered_mask]
    filtered_labels = labels.copy()[filtered_mask]

    if boxes is not None:
        assert boxes.shape[-1] == 7, "invalid box shape, the box coord must be PSR mode"

    # render color
    # unique_labels = np.unique(np.array(list(class_label.keys())))  # 获取所有类别
    # colormap = plt.get_cmap("tab20")  # 使用 matplotlib 的 tab20 颜色映射
    # colors = colormap(filtered_labels / unique_labels.max()).reshape(filtered_labels.shape[0], 4)[:, :3]  # 归一化标签并分配颜色 (取 RGB)
    colors = np.array(list(map(lambda x: pcd_colors[int(x)], filtered_labels))) / 255

    # construct pcd object
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(colors)

    # print(pcd)
    # o3d.visualization.draw_geometries_with_editing([filtered_pcd])  # check coordinate direction

    # create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name)

    # visual pcd
    vis.add_geometry(filtered_pcd)
    # visual box
    if boxes is not None:
        corners = get_corners(boxes.copy())
        boxes_color = np.array(boxes_color)
        colors = np.tile(boxes_color, (corners.shape[0], 1)) if boxes_color.shape == (3, ) else boxes_color
        assert corners.shape[0] == colors.shape[0]
        for corner, color in zip(corners, colors):
            # color = np.array(boxes_color)
            # color = np.array(colorMap[class_names[label]]) / 255.0
            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7],
                [4, 6], [5, 7]
            ]
            line_colors = np.tile(color, len(lines)).reshape(-1, 3)
            line_set = o3d.geometry.LineSet()
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(line_colors)
            line_set.points = o3d.utility.Vector3dVector(corner)
            vis.add_geometry(line_set)

    # create coordinate
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0,
                                                                         origin=[0., 0., 0.])  # size可以根据点云大小调整
    vis.add_geometry(coordinate_frame)
    #
    # # visual pcd and coordinate
    # o3d.visualization.draw_geometries([filtered_pcd, coordinate_frame],
    #                                   window_name="Point Cloud by Category",
    #                                   # front-left-up coord
    #                                   front=[1, 0, 0],  # x -> front
    #                                   lookat=[0, 0, 0],  # zero point
    #                                   up=[0, 0, 1],  # z -> up
    #                                   zoom=0.8)  # scale factor
    view_control = vis.get_view_control()
    view_control.set_front([1, 0, 0])
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, 0, 1])
    view_control.set_zoom(0.8)

    vis.run()
    vis.destroy_window()


def visual_occ(occ_label, voxel_size, pc_range, mask_camera=None, fill_z=False, predict_mode=False):
    import mayavi.mlab as mlab

    if len(occ_label.shape) == 4:
        occ_label = occ_label[0]
    print("occ data shape:", occ_label.shape)

    fov_voxels = []

    empty_index = 8 if predict_mode else 17

    if fill_z:
        for i in range(occ_label.shape[0]):
            for j in range(occ_label.shape[1]):
                for k in range(occ_label.shape[2]):
                    category = occ_label[i][j][k]
                    if category in [empty_index]:
                        continue
                    else:
                        for m in range(int(voxel_size[2] / voxel_size[0])):
                            fov_voxels.append([i+0.5, j+0.5, k + (m + 0.5) * (voxel_size[0] / voxel_size[2]), category])

        fov_voxels = np.array(fov_voxels).astype(np.float32)

        print("fov_voxels size:", fov_voxels.shape)

        fov_voxels[:, 0] = (fov_voxels[:, 0]) * voxel_size[0]
        fov_voxels[:, 1] = (fov_voxels[:, 1]) * voxel_size[1]
        fov_voxels[:, 2] = (fov_voxels[:, 2]) * voxel_size[2]
    else:
        for i in range(occ_label.shape[0]):
            for j in range(occ_label.shape[1]):
                for k in range(occ_label.shape[2]):
                    classi = occ_label[i][j][k]

                    if classi in [empty_index] or (mask_camera is not None and mask_camera[i][j][k]==0) :
                        continue
                    else:
                        fov_voxels.append([i, j, k, classi])

        fov_voxels = np.array(fov_voxels).astype(np.float32)
        # print("fov_voxels:", fov_voxels.shape, fov_voxels)

        # fov_voxels = fov_voxels[fov_voxels[..., 3] > 0]
        fov_voxels[:, :1] = (fov_voxels[:, :1] + 0.5) * voxel_size[0]
        fov_voxels[:, 1:2] = (fov_voxels[:, 1:2] + 0.5) * voxel_size[1]
        fov_voxels[:, 2:3] = (fov_voxels[:, 2:3] + 0.5) * voxel_size[2]

    fov_voxels[:, 0] += pc_range[0]
    fov_voxels[:, 1] += pc_range[1]
    fov_voxels[:, 2] += pc_range[2]

    # figure = mlab.figure(size=(600, 600), bgcolor=(1, 1, 1))
    # figure = mlab.figure(visual_path.split("/")[-2],size=(2560, 1440), bgcolor=(1, 1, 1))
    # figure = mlab.figure(visual_path, size=(600, 200), bgcolor=(1, 1, 1))
    # pdb.set_trace()
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size[0] - 0.05 * voxel_size[0],
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=19,
    )
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"

    plt_plot_fov_car = mlab.points3d(
        [0],
        [0],
        [0],
        [1],
        scale_factor=1,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=19,
    )
    # plt_plot_fov_car.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = occ_colors
    # mlab.savefig('temp_ly/mayavi_ly.png')
    mlab.show()


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


def gray2rainbow(value):
    pixel = [255, 255, 0]
    if value <= 51:
        pixel = [255, value * 5, 0]
    elif value <= 102:
        value -= 51
        pixel = [255 - value * 5, 255, 0]
    elif value <= 153:
        value -= 102
        pixel = [0, 255, value * 5]
    elif value <= 204:
        value -= 153
        pixel = [0, 255 - int(value * 128 / 51 + 0.5), 255]
    elif value <= 255:
        value -= 204
        pixel = [0, 127 - int(value * 127 / 51 + 0.5), 255]
    return np.array(pixel)


def render_depth_on_image(img, depth, depth_min=0, depth_max=0, transparency=1.0, mode='circle'):
    assert img.shape[:2] == depth.shape
    coords = np.where(depth > 0)
    value = depth[coords]
    if depth_max == 0:
        depth_max = value.max()
    value = np.clip(value, depth_min, depth_max)
    value = (value - depth_min) / (depth_max - depth_min)
    for x, y, v in zip(*coords, value):
        v = int(v * 255)
        col = gray2rainbow(v)
        col = img[x, y, :] * (1 - transparency) + col * transparency
        col = [int(x) for x in col]
        if mode == 'circle':
            cv2.circle(img, (y, x), 0, col, -1)
        elif mode == 'pixel':
            img[x, y, :] = col
        else:
            raise NotImplementedError('invalid render mode.')

    return img


if __name__ == "__main__":
    # pcd_range = [-20, -20, -1.4, 20, 20, 2.2]  # [x_min, y_min, z_min, x_max, y_max, z_max](m)
    # voxel_size = [0.05, 0.05, 0.4]
    predict_mode = True
    occ_output_dir = './Datasets/issue_record_2025_07_08_16_16_00_extract_4/_2025-07-08-16-16-03_0/occ'
    for occ_data_name in sorted(os.listdir(occ_output_dir))[-100:]:
        occ_data_path = os.path.join(occ_output_dir, occ_data_name, f'{occ_data_name}.npz')
        occ_data = np.load(occ_data_path)
        occ_label = occ_data['semantics']
        pc_range = occ_data['pc_range']
        voxel_size = occ_data['voxel_size']
        mask_camera = occ_data['mask_camera']
        # import pdb;pdb.set_trace()
        visual_occ(occ_label=occ_label, pc_range=pc_range, voxel_size=voxel_size, mask_camera=mask_camera, fill_z=True,
                   predict_mode=predict_mode)
解