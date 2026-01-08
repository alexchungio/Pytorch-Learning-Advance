import numpy as np


def get_cam_coord_in_ego_coord(ego2cam_mat):
    """
    get camera coord in the ego coord system
    Args:
        ego2cam_mat:

    Returns:

    """
    ego2cam_mat = np.array(ego2cam_mat, dtype=np.float32)
    cam2ego_mat = np.linalg.inv(ego2cam_mat)

    # cam_coord: x -> right, y -> down, z -> front
    cam_coord_optical_center = [0, 0, 0, 1]
    ego_coord_optical_center = cam2ego_mat @ cam_coord_optical_center

    return ego_coord_optical_center


def get_optical_coord_in_img_coord(optical_coord, K, D, undistort=True):
    """
    get optical coord in the image coord system
    Args:
        optical_coord:
        K:
        D:
        undistort:

    Returns:

    """
    K = np.array(K)
    D = np.array(D)
    # normalize
    optical_coord_norm = optical_coord / optical_coord[:, 2][:, np.newaxis]  # (x/z, y/z, 1)
    # TODO ignore distortion
    if undistort:
        pass
    uv_1 = optical_coord_norm @ K.T
    uv = uv_1[:, :2]

    return uv


if __name__ == "__main__":
    # camera coordinate system: x -> right, y -> down, z -> front

    # fisheye camera intra-param and extra-param
    # image_size: [1920, 1536] # (W, H)
    cam_a_front = {
        'intra':
            [[444.551, 0.0, 958.056],
             [0.0, 444.436, 766.658],
             [0.0, 0.0, 1.0]],
        'D': [0.151969, -0.0729416, 0.0223519, -0.0027434],
        'extra':  # T
            [[-0.0018607662921296072, -0.9999135168347716, 0.013021628941955169, -0.14269328394559694],
             [-0.32374123081309586, -0.011718021140193728, -0.9460731246023264, 1.8589260403675762],
             [0.946143837746221, -0.005976060337372186, -0.32369142729277434, -3.7327890430187822],
             [0.0, 0.0, 0.0, 1.0]]
    }
    cam_a_back = {
        'intra':
            [[445.912, 0.0, 956.874],
             [0.0, 446.029, 769.107],
             [0.0, 0.0, 1.0]],
        'D': [0.152265, -0.0727098, 0.0222699, -0.002708],
        'extra':  # T
            [[-0.01906966382888496, 0.999787543611002, -0.007828897930117082, 0.09312958702874971],
             [0.4541657825891613, 0.0016862596649677658, -0.8909156206436581, 1.146303237529569],
             [-0.8907130817011412, -0.020545084006618457, -0.45410144415188614, -0.6513270008726006],
             [0.0, 0.0, 0.0, 1.0]]
    }

    cam_a_left = {
        'intra':
            [[446.433, 0.0, 958.42],
             [0.0, 446.275, 768.358],
             [0.0, 0.0, 1.0]],
        'D': [0.149372, -0.0699979, 0.0209462, -0.00251038],
        'extra':  # T
            [[0.9993971683762107, -0.013573127854624794, 0.03195430947173704, -2.3270317106952625],
             [0.020008661269951857, -0.5269908737432965, -0.8496354136517967, 1.3781319028791912],
             [0.02837183828532727, 0.8497625883595619, -0.5264016017327777, -0.2794546987905724],
             [0.0, 0.0, 0.0, 1.0]],
    }
    cam_a_right = {
        'intra':
            [[449.145, 0.0, 959.927],
             [0.0, 446.402, 768.656],
             [0.0, 0.0, 1.0]],
        'D': [0.149796, -0.0707195, 0.0214003, -0.00262933],
        'extra':
            [[-0.9984118667447346, -0.007435417369693265, -0.05584326062536746, 2.3230406619019544],
             [0.04517448527966062, 0.48659945085894196, -0.8724564786739653, 1.4584584852901117],
             [0.033660374742162956, -0.8735935874274143, -0.48549076925811646, -0.6092559153799021],
             [0.0, 0.0, 0.0, 1.0]]
    }

    # get camera coord
    ego_coord_cam_front = get_cam_coord_in_ego_coord(cam_a_front['extra'])
    print(f'cam_front: {ego_coord_cam_front}')
    ego_coord_cam_back = get_cam_coord_in_ego_coord(cam_a_back['extra'])
    print(f'cam_back: {ego_coord_cam_back}')
    ego_coord_cam_left = get_cam_coord_in_ego_coord(cam_a_left['extra'])
    print(f'cam_left: {ego_coord_cam_left}')
    ego_coord_cam_right = get_cam_coord_in_ego_coord(cam_a_right['extra'])
    print(f'cam_right: {ego_coord_cam_right}')

    # cam_front: [ 4.13330078 -0.14320536  0.5522663   1.]
    # cam_back: [-1.09898126 - 0.10842432  0.72622001  1.]
    # cam_left: [2.30598283 0.93214792 1.09816289 1.]
    # cam_right: [2.27397394 - 1.22465432  1.10637951  1.]

    # point vertical to optical center, not affected by distortion
    optical_cam_coord = np.array([[0, 0, 0.2],
                                  [0, 0, 1]])  # (x, Y, Z)
    img_coord_cam_front = get_optical_coord_in_img_coord(optical_cam_coord,
                                                         K=cam_a_front['intra'],
                                                         D=cam_a_front['D'],
                                                         undistort=False)
    print(img_coord_cam_front)

    # [[958.056 766.658]
    # [958.056 766.658]]