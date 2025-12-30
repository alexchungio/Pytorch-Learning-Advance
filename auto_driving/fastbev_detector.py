# -*- coding: utf-8 -*-
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import numpy as np

from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmseg.models import build_head as build_seg_head
from mmdet.models.detectors import BaseDetector
from mmdet3d.core import bbox3d2result
from mmseg.ops import resize
from mmcv.runner import get_dist_info, auto_fp16

import copy


class KittiSetOrigin:
    def __init__(self, point_cloud_range):
        """

        Args:
            point_cloud_range: boundary coord at ego coord (left, rear, down, right, front, up)
        """

        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.origin = (point_cloud_range[:3] + point_cloud_range[3:]) / 2.

    def __call__(self, results):
        results['lidar2img']['origin'] = self.origin.copy()
        return results


@DETECTORS.register_module()
class FastBEV(BaseDetector):
    def __init__(
            self,
            backbone,
            neck,
            neck_fuse,
            neck_3d,
            bbox_head,
            seg_head,
            n_voxels,  # n_voxels=[[200, 200, 4]],
            voxel_size,  # voxel_size=[[0.5, 0.5, 1.5]],
            bbox_head_2d=None,
            train_cfg=None,
            test_cfg=None,
            train_cfg_2d=None,
            test_cfg_2d=None,
            pretrained=None,
            init_cfg=None,
            extrinsic_noise=0,
            seq_detach=False,
            multi_scale_id=None,  # multi_scale_id=[0]
            multi_scale_3d_scaler=None,
            with_cp=False,
            backproject='inplace',
            style='v1',
    ):
        super().__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.neck_3d = build_neck(neck_3d)
        if isinstance(neck_fuse['in_channels'], list):
            for i, (in_channels, out_channels) in enumerate(zip(neck_fuse['in_channels'], neck_fuse['out_channels'])):
                self.add_module(
                    f'neck_fuse_{i}',
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        else:
            self.neck_fuse = nn.Conv2d(neck_fuse["in_channels"], neck_fuse["out_channels"], 3, 1, 1)

        # style
        # v1: fastbev wo/ ms
        # v2: fastbev + img ms
        # v3: fastbev + bev ms
        # v4: fastbev + img/bev ms
        self.style = style
        assert self.style in ['v1', 'v2', 'v3', 'v4'], self.style
        self.multi_scale_id = multi_scale_id
        self.multi_scale_3d_scaler = multi_scale_3d_scaler

        if bbox_head is not None:
            bbox_head.update(train_cfg=train_cfg)
            bbox_head.update(test_cfg=test_cfg)
            self.bbox_head = build_head(bbox_head)
            self.bbox_head.voxel_size = voxel_size
        else:
            self.bbox_head = None

        if seg_head is not None:
            self.seg_head = build_seg_head(seg_head)
        else:
            self.seg_head = None

        if bbox_head_2d is not None:
            bbox_head_2d.update(train_cfg=train_cfg_2d)
            bbox_head_2d.update(test_cfg=test_cfg_2d)
            self.bbox_head_2d = build_head(bbox_head_2d)
        else:
            self.bbox_head_2d = None

        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # test time extrinsic noise
        self.extrinsic_noise = extrinsic_noise
        if self.extrinsic_noise > 0:
            for i in range(5):
                print("### extrnsic noise: {} ###".format(self.extrinsic_noise))

        # detach adj feature
        self.seq_detach = seq_detach
        self.backproject = backproject
        # checkpoint
        self.with_cp = with_cp

    @staticmethod
    def _compute_projection(img_meta, stride, noise=0):
        """
        get porjection of ego-coord to img-feat-coord for front camera(no suitable for fisheye cameras)
        """
        projection = []
        intrinsic = torch.tensor(img_meta["lidar2img"]["intrinsic"][:3, :3])
        intrinsic[:2] /= stride
        extrinsics = map(torch.tensor, img_meta["lidar2img"]["extrinsic"])
        # get ego2image of per camera view
        for extrinsic in extrinsics:
            if noise > 0:
                projection.append(intrinsic @ extrinsic[:3] + noise)
            else:
                projection.append(intrinsic @ extrinsic[:3])  # ego -> image

        return torch.stack(projection)

    @staticmethod
    def _compute_projection_5v(img_meta, stride, noise=0):
        """

        Args:
            stride:
            noise:

        Returns:

        """

        projection = []
        orin_extrinsics = []
        orin_intrinsics = []
        orin_distortion_params = []

        post_rts = np.array(img_meta['lidar2img']['aug_post_rt'])
        post_rts = torch.from_numpy(post_rts)

        intrinsic = torch.tensor(img_meta["lidar2img"]["intrinsic"][:3, :3])
        intrinsic[:2] /= stride
        extrinsics = map(torch.tensor, img_meta["lidar2img"]["extrinsic"])

        for index, extrinsic in enumerate(extrinsics):
            if noise > 0:
                projection.append(intrinsic @ extrinsic[:3] + noise)
            else:
                projection.append(intrinsic @ extrinsic[:3])

            # get origin param for fisheye projection
            rotation_matrix = img_meta['lidar2img']['lidar2img_aug'][index]['rot']
            translation_vector = img_meta['lidar2img']['lidar2img_aug'][index]['tran']  # 【3
            lidar2cam_r = np.linalg.inv(rotation_matrix)
            lidar2cam_t = translation_vector @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            orin_extrinsic = torch.from_numpy(lidar2cam_rt.T[:3])  # (3, 4)
            orin_extrinsics.append(orin_extrinsic)

            orin_intrinsic = torch.from_numpy(
                np.array(img_meta['lidar2img']['lidar2img_extra'][index]['cam_intrinsic']))  # (3, 3)
            # orin_intrinsic[:2, :2] /= stride  # adapt feat_size
            orin_intrinsics.append(orin_intrinsic)
            distortion_params = torch.from_numpy(
                np.array(img_meta['lidar2img']['lidar2img_extra'][index]['distortion_params']))

            orin_distortion_params.append(distortion_params)

        return torch.stack(projection), torch.stack(orin_extrinsics), torch.stack(orin_intrinsics), torch.stack(
            orin_distortion_params), post_rts


    def extract_feat(self, img, img_metas, mode):
        """
        extract bev feat
        """
        batch_size = img.shape[0]
        img = img.reshape(
            [-1] + list(img.shape)[2:]
        )  # [1, 6, 3, 928, 1600] -> [6, 3, 928, 1600]

        # -------------------- extract image feature --------------------
        x = self.backbone(
            img
        )  # [6, 256, 232, 400]; [6, 512, 116, 200]; [6, 1024, 58, 100]; [6, 2048, 29, 50]

        # use for vovnet
        if isinstance(x, dict):
            tmp = []
            for k in x.keys():
                tmp.append(x[k])
            x = tmp

        # -------------------- fuse multi level features -------------------
        def _inner_forward(x):
            out = self.neck(x)
            return out  # [6, 64, 232, 400]; [6, 64, 116, 200]; [6, 64, 58, 100]; [6, 64, 29, 50])

        if self.with_cp and x.requires_grad:
            mlvl_feats = cp.checkpoint(_inner_forward, x)
        else:
            mlvl_feats = _inner_forward(x)  # mlvl(multi-level)
        mlvl_feats = list(mlvl_feats)

        features_2d = None
        if self.bbox_head_2d:
            features_2d = mlvl_feats

        if self.multi_scale_id is not None:
            mlvl_feats_ = []
            for msid in self.multi_scale_id:
                # fpn output fusion
                if getattr(self, f'neck_fuse_{msid}', None) is not None:
                    fuse_feats = [mlvl_feats[msid]]
                    for i in range(msid + 1, len(mlvl_feats)):
                        resized_feat = resize(
                            mlvl_feats[i],
                            size=mlvl_feats[msid].size()[2:],
                            mode="bilinear",
                            align_corners=False)
                        fuse_feats.append(resized_feat)

                    if len(fuse_feats) > 1:
                        fuse_feats = torch.cat(fuse_feats, dim=1)
                    else:
                        fuse_feats = fuse_feats[0]
                    fuse_feats = getattr(self, f'neck_fuse_{msid}')(fuse_feats)
                    mlvl_feats_.append(fuse_feats)
                else:
                    mlvl_feats_.append(mlvl_feats[msid])
            mlvl_feats = mlvl_feats_
        # v3 bev ms
        if isinstance(self.n_voxels, list) and len(mlvl_feats) < len(self.n_voxels):
            pad_feats = len(self.n_voxels) - len(mlvl_feats)
            for _ in range(pad_feats):
                mlvl_feats.append(mlvl_feats[0])

        # --------------- visual transform to get bev volume of bev (2D -> 3D) --------------
        mlvl_volumes = []
        for lvl, mlvl_feat in enumerate(mlvl_feats):
            stride_i = math.ceil(img.shape[-1] / mlvl_feat.shape[-1])  # P4 880 / 32 = 27.5
            # [bs*seq*nv, c, h, w] -> [bs, seq*nv, c, h, w]
            mlvl_feat = mlvl_feat.reshape([batch_size, -1] + list(mlvl_feat.shape[1:]))
            # [bs, seq*nv, c, h, w] -> list([bs, nv, c, h, w])
            mlvl_feat_split = torch.split(mlvl_feat, 6, dim=1)

            volume_list = []
            for seq_id in range(len(mlvl_feat_split)):
                volumes = []
                for batch_id, seq_img_meta in enumerate(img_metas):
                    feat_i = mlvl_feat_split[seq_id][batch_id]  # [nv, c, h, w]
                    img_meta = copy.deepcopy(seq_img_meta)
                    img_meta["lidar2img"]["extrinsic"] = img_meta["lidar2img"]["extrinsic"][seq_id * 6:(seq_id + 1) * 6]
                    if isinstance(img_meta["img_shape"], list):
                        img_meta["img_shape"] = img_meta["img_shape"][seq_id * 6:(seq_id + 1) * 6]
                        img_meta["img_shape"] = img_meta["img_shape"][0]

                    # --------------- get shape of image feature ---------------------
                    height = math.ceil(img_meta["img_shape"][0] / stride_i)
                    width = math.ceil(img_meta["img_shape"][1] / stride_i)

                    # -------- get projection from ego coord to image coord ------------
                    projection = self._compute_projection(
                        img_meta, stride_i, noise=self.extrinsic_noise).to(feat_i.device)
                    if self.style in ['v1', 'v2']:
                        # wo/ bev ms
                        n_voxels, voxel_size = self.n_voxels[0], self.voxel_size[0]
                    else:
                        # v3/v4 bev ms
                        n_voxels, voxel_size = self.n_voxels[lvl], self.voxel_size[lvl]

                    # ------------- get all coord points at ego coord ------------
                    points = get_points(  # [3, vx, vy, vz]
                        n_voxels=torch.tensor(n_voxels),
                        voxel_size=torch.tensor(voxel_size),
                        origin=torch.tensor(img_meta["lidar2img"]["origin"]),# origin coord is center of ego(KittiSetOrigin)
                    ).to(feat_i.device)

                    # ------------- get bev feat volume ------------------
                    if self.backproject == 'inplace':
                        volume = backproject_inplace(
                            feat_i[:, :, :height, :width], points, projection)  # [c, vx, vy, vz]
                    else:
                        volume, valid = backproject_vanilla(
                            feat_i[:, :, :height, :width], points, projection)

                        # fusion all bev volume to all camera view with average
                        # sum bev feat along of all camera
                        volume = volume.sum(dim=0)
                        valid = valid.sum(dim=0)
                        # average feat per bev voxel by divide number of valid feat
                        volume = volume / valid  # average
                        valid = valid > 0
                        volume[:, ~valid[0]] = 0.0

                    volumes.append(volume)
                volume_list.append(torch.stack(volumes))  # list([bs, c, vx, vy, vz])

            mlvl_volumes.append(torch.cat(volume_list, dim=1))  # list([bs, seq*c, vx, vy, vz])

        # if self.style in ['v1', 'v2']:
        mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, lvl*seq*c, vx, vy, vz]

        x = mlvl_volumes

        def _inner_forward(x):
            # v1/v2: [bs, lvl*seq*c, vx, vy, vz] -> [bs, c', vx, vy]
            # v3/v4: [bs, z1*c1+z2*c2+..., vx, vy, 1] -> [bs, c', vx, vy]
            out = self.neck_3d(x)
            return out

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x, None, features_2d


@auto_fp16(apply_to=('img',))
def forward(self, img, img_metas, return_loss=True, **kwargs):
    """Calls either :func:`forward_train` or :func:`forward_test` depending
    on whether ``return_loss`` is ``True``.

    Note this setting will change the expected inputs. When
    ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
    and List[dict]), and when ``resturn_loss=False``, img and img_meta
    should be double nested (i.e.  List[Tensor], List[List[dict]]), with
    the outer list indicating test time augmentations.
    """
    if torch.onnx.is_in_onnx_export():
        if kwargs["export_2d"]:
            return self.onnx_export_2d(img, img_metas)
        elif kwargs["export_3d"]:
            return self.onnx_export_3d(img, img_metas)
        else:
            raise NotImplementedError

    if return_loss:
        return self.forward_train(img, img_metas, **kwargs)
    else:
        return self.forward_test(img, img_metas, **kwargs)


def forward_test(self, img, img_metas, **kwargs):
    if not self.test_cfg.get('use_tta', False):
        return self.simple_test(img, img_metas)
    return self.aug_test(img, img_metas)


def simple_test(self, img, img_metas):
    bbox_results = []
    feature_bev, _, features_2d = self.extract_feat(img, img_metas, "test")
    if self.bbox_head is not None:
        x = self.bbox_head(feature_bev)
        bbox_list = self.bbox_head.get_bboxes(*x, img_metas, valid=None)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list
        ]

    else:
        bbox_results = [dict()]

    # BEV semantic seg
    if self.seg_head is not None:
        x_bev = self.seg_head(feature_bev)
        bbox_results[0]['bev_seg'] = x_bev

    return bbox_results


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    """
    get cloud point at ego points, notes the origin coord at center of ego
    """
    points = torch.stack(
        torch.meshgrid(
            [
                torch.arange(n_voxels[0]),
                torch.arange(n_voxels[1]),
                torch.arange(n_voxels[2]),
            ]
        )
    )
    new_origin = origin - n_voxels / 2.0 * voxel_size  # orgin_coord
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)  # trans coord to ego-coord
    return points


def backproject_vanilla(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [6, 64, 200, 200, 12]
        valid: [6, 1, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    #  convert to homogeneous coordinates: [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # --------- ego_to_img(3D -> 2D): project ego-point to image coord -------------
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img

    # --------- convert to pixel coord -----------------
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000] #

    # --------- remove invalid feat index --------------
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]
    volume = torch.zeros(
        (n_images, n_channels, points.shape[-1]), device=features.device
    ).type_as(features)  # [6, 64, 480000]

    # --------- fill bev feat volume with all camera valid index ---------
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    # [6, 64, 480000] -> [6, 64, 200, 200, 12]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    # [6, 480000] -> [6, 1, 200, 200, 12]
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)

    return volume, valid


def backproject_inplace(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [64, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]

    # method2：特征填充，只填充有效特征，重复特征直接覆盖
    volume = torch.zeros(
        (n_channels, points.shape[-1]), device=features.device
    ).type_as(features)
    for i in range(n_images):
        volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]

    volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume


def backproject_vanilla_5v(features, points, projection, orin_extr, orin_intr, orin_D, orin_post_rt, stride):
    """
    function: 2d feature + predefined point cloud -> 3d volume for 5v
    Args:
        features:
        points:
        projection:
        orin_extr:
        orin_intr:
        orin_D:
        orin_post_rt:
        stride:

    Returns:

    """
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam

    # computer projection of cam_front
    points_3d_front = torch.bmm(projection[4][None], points[4][None])
    x_front = (points_3d_front[:, 0] / points_3d_front[:, 2]).round().long()  # [4, 480000]
    y_front = (points_3d_front[:, 1] / points_3d_front[:, 2]).round().long()  # [4, 480000]
    z_front = points_3d_front[:, 2]  # [4, 480000]
    valid_front = (x_front >= 0) & (y_front >= 0) & (x_front < width) & (y_front < height - height * 0.30) & (
                z_front > 0)  # [4, 480000]

    # computer projection of cam_a_front, cam_a_back, cam_a_left, cam_right
    points_3d_cam = torch.bmm(orin_extr[:4], points[:4])
    points_2d_aug = []
    for cam in range(n_images - 1):
        corners_2d_single_cam = cam2pixel_fisheye(points_3d_cam[cam][0:3].T, orin_intr[cam], orin_D[cam],
                                                  features.device)
        points_2d_aug_single_cam = torch.matmul(corners_2d_single_cam, orin_post_rt[cam].T)
        points_2d_aug.append(points_2d_aug_single_cam)
    points_2d_aug = torch.stack(points_2d_aug).permute(0, 2, 1)

    # coord compatible img_feat size
    x = (points_2d_aug[:, 0] / stride).round().long()  #
    y = (points_2d_aug[:, 1] / stride).round().long()  # [4, 480000]

    x = points_2d_aug[:, 0]
    y = points_2d_aug[:, 1]
    z = points_3d_cam[:, 2]  # [4, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]

    # concat cam_around and cam_front coord
    valid_all = torch.cat([valid, valid_front], dim=0)
    x_all = torch.cat([x, x_front], dim=0)
    y_all = torch.cat([y, y_front], dim=0)

    # gen feat volume for each camera view
    volume = torch.zeros(
        (n_images, n_channels, points.shape[-1]), device=features.device
    ).type_as(features)  # [6, 64, 480000]
    for i in range(n_images):
        volume[i, :, valid_all[i]] = features[i, :, y_all[i, valid_all[i]], x_all[i, valid_all[i]]]

    # [6, 64, 480000] -> [6, 64, 200, 200, 12]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    # [6, 480000] -> [6, 1, 200, 200, 12]
    valid_all = valid_all.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)

    return volume, valid_all


def cam2pixel_fisheye(point_in_cam, intra, dist, device):
    """

    Args:
        point_in_cam:
        intra:
        dist:
        device:

    Returns:

    """
    k1, k2, k3, k4 = dist
    fx, fy, cu, cv = intra[0][0], intra[1][1], intra[0][2], intra[1][2]

    # computer coord at normalized plane
    norm_x = point_in_cam[:, 0] / point_in_cam[:, 2]
    norm_y = point_in_cam[:, 1] / point_in_cam[:, 2]
    r = torch.sqrt(norm_x * norm_x + norm_y * norm_y)

    # computer distorted coord
    theta = torch.atan(r)
    theta_d = theta + k1 * torch.pow(theta, 3) + k2 * torch.pow(theta, 5) + \
              k3 * torch.pow(theta, 7) + k4 * torch.pow(theta, 9)
    dis_x = theta_d / r * norm_x
    dis_y = theta_d / r * norm_y

    tmp = torch.ones_like(dis_x, device=dis_x.device)
    pts = torch.stack([dis_x, dis_y, tmp], dim=-1)
    points_2d = torch.matmul(pts, intra[:3, :3].T)
    # pt0 = fx*disX + cu
    # pt1 = fy*disY + cv
    return points_2d


