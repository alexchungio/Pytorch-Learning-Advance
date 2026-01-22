import logging
import math
from typing import Any, Dict, Tuple
import os

import numpy as np
import torch
from torch import nn, Tensor
from typing import Tuple, Collection, List, Union, Optional
import torch.nn.functional as F
from torch.quantization import DeQuantStub

# import horizon_plugin_pytorch.nn as hnn
# from horizon_plugin_pytorch.qtensor import QTensor
# from hat.registry import OBJECT_REGISTRY
# from hat.utils.model_helpers import fx_wrap


__all__ = [
    "TemporalFusion",
    "GAddTemporalFusion",
]

def adjust_coords_delta(coords: torch.Tensor, grid_size: Tuple[int]) -> torch.Tensor:
    """adjust coords to delta of coords for gridsample with delta

    Args:
        coords: Coords for grid_sample.
        # coords: now is the uv in img feature
        grid_size: Grid size.
    """

    W = grid_size[0]
    H = grid_size[1]

    bev_x = (torch.linspace(0, W - 1, W).reshape((1, W)).repeat(H, 1)).float()
    bev_y = (torch.linspace(0, H - 1, H).reshape((H, 1)).repeat(1, W)).float()

    bev_coords = torch.stack([bev_x, bev_y], axis=-1).to(device=coords.device)
    coords = coords - bev_coords
    return coords


class GridsampleDelta(nn.Module):

    """Refine this docstring in the future.

    Given an input and a flow-field grid, computes the output using
    input values and pixel locations from grid.

    Note that the grid required by this function is DIFFERENT from
    torch.nn.functional.grid_sample !!!

    Args:
        mode (str, optional): Interpolation mode to calculate output values.
            Only "bilinear" and "nearest" supported now.
            Defaults to "bilinear".
        padding_mode (str, optional): Padding mode for outside grid values.
            Only "zeros" and "border" is supported now.
            Defaults to "zeros".
        align_corners ([type], optional): Since the grid format is
            different with torch.nn.functional.grid_sample, this param
            does not have any effect now.
            Defaults to None.
    """

    def __init__(
        self, mode="bilinear", padding_mode="zeros", align_corners=False
    ):
        super(GridsampleDelta, self).__init__()

        assert mode in (
            "bilinear",
            "nearest",
        ), "GridSample only support 'bilinear' and 'nearest' mode now"
        assert padding_mode in (
            "zeros",
            "border",
        ), "GridSample only support 'zeros' and 'border' padding_mode now"
        assert isinstance(
            align_corners, (bool, type(None))
        ), "param 'align_corners' must be bool or None"

        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, x, grid):
        # type: (Tensor, Tensor) -> Tensor
        """
        Forward pass of GridSample.

        Args:
            x (Tensor[N, C, H, W]): Input data.
            grid (Tensor[N, H_out, W_out, (dx, dy)]): Flow-field. This param
                is different with torch.nn.functional.grid_sample. In this
                function, the sample point of output point (x, y) is computed
                by (x + dx, y + dy).
        """
        r = self.warp(x, grid, self.mode, self.padding_mode)
        return r

    def warp(self,
             x: Tensor,
             grid: Tensor,
             mode: str = "bilinear",
             padding_mode: str = "zeros",
    ):
        """Refine this docstring in the future.

        Given an input and a flow-field grid, computes the output using
        input values and pixel locations from grid.

        Note that the grid required by this function is DIFFERENT from
        torch.nn.functional.grid_sample !!!

        Args:
            mode (str, optional): Interpolation mode to calculate output values.
                Only "bilinear" and "nearest" supported now.
                Defaults to "bilinear".
            padding_mode (str, optional): Padding mode for outside grid values.
                Only "zeros" and "border" is supported now.
                Defaults to "zeros".
        """
        input_dtype = x.dtype
        x = x.float()
        grid = grid.float()

        # convert grid format from 'delta' to 'norm'
        n = grid.size(0)
        h = grid.size(1)
        w = grid.size(2)
        base_coord_y = (
            torch.arange(h, dtype=grid.dtype, device=grid.device)
            .unsqueeze(-1)
            .unsqueeze(0)
            .expand(n, h, w)
        )
        base_coord_x = (
            torch.arange(w, dtype=grid.dtype, device=grid.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(n, h, w)
        )
        absolute_grid_x = grid[:, :, :, 0] + base_coord_x
        absolute_grid_y = grid[:, :, :, 1] + base_coord_y
        norm_grid_x = absolute_grid_x * 2 / (x.size(3) - 1) - 1
        norm_grid_y = absolute_grid_y * 2 / (x.size(2) - 1) - 1
        norm_grid = torch.stack((norm_grid_x, norm_grid_y), dim=-1)

        r = F.grid_sample(x, norm_grid, mode, padding_mode, self.align_corners)

        return r.to(input_dtype)


class GridsampleNorm(nn.Module):
    """
    grid_sample with pytorch, interface
    grid` has values outside the range of ``[-1, 1]``
    """
    def __init__(
            self, mode="bilinear", padding_mode="zeros", align_corners=False
    ):
        super(GridsampleNorm, self).__init__()

        assert mode in (
            "bilinear",
            "nearest",
        ), "GridSample only support 'bilinear' and 'nearest' mode now"
        assert padding_mode in (
            "zeros",
            "border",
        ), "GridSample only support 'zeros' and 'border' padding_mode now"
        assert isinstance(
            align_corners, (bool, type(None))
        ), "param 'align_corners' must be bool or None"

        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, x, grid):

        """
        Forward pass of GridSample.

        Args:
            x (Tensor[N, C, H, W]): Input data.
            grid (Tensor[N, H_out, W_out, (dx, dy)]): specifies the sampling pixel locations normalized by the
                `input` spatial dimensions. Therefore, it should have most values in the range of ``[-1, 1]``.
                 For example, values ``x = -1, y = -1`` is the left-top pixel of :attr:`input`, and values
                  ``x = 1, y = 1`` is the right-bottom pixel of :attr:`input`.
        """
        input_dtype = x.dtype
        out = F.grid_sample(x, grid, self.mode, self.padding_mode, self.align_corners)

        return out.to(input_dtype)


class ConvModule2d(nn.Module):
    """
    A conv block that bundles conv/norm/activation layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        dilation (int | tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool): Same as nn.Conv2d.
        padding_mode (str): Same as nn.Conv2d.
        norm_layer (nn.Module): Normalization layer.
        act_layer (nn.Module): Activation layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = None,
        **kwargs
    ):
        super(ConvModule2d, self).__init__()
        self.conv_2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        # conv_list = [conv, norm_layer, act_layer]
        # self.conv_list = [layer for layer in conv_list if layer is not None]
        # super(ConvModule2d, self).__init__(*self.conv_list)

    def forward(self, x):

        x = self.conv_2d(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        if self.act_layer is not None:
            x = self.act_layer
        out = x
        # out = super().forward(x)
        return out


class SeparableConv2D(nn.Module):
    """
    depth separable conv2d
    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        dilation (int | tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool): Same as nn.Conv2d.
        padding_mode (str): Same as nn.Conv2d.
        dw_norm_layer (nn.Module): Normalization layer in depth-wise conv.
        dw_act_layer (nn.Module): Activation layer in depth-wise  conv.
        pw_norm_layer (nn.Module): Normalization layer in point-wise  conv.
        pw_act_layer (nn.Module): Activation layer in point-wise  conv.
    """
    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernel_size: Union[int, Tuple[int, int]],
                stride: Union[int, Tuple[int, int]] = 1,
                padding: Union[int, Tuple[int, int]] = 0,
                dilation: Union[int, Tuple[int, int]] = 1,
                bias: bool = True,
                padding_mode: str = "zeros",
                dw_norm_layer: Union[None, nn.Module] = None,
                dw_act_layer: Union[None, nn.Module] = None,
                pw_norm_layer: Union[None, nn.Module] = None,
                pw_act_layer: Union[None, nn.Module] = None,
    ):
        super(SeparableConv2D, self).__init__()
        self.dw_conv = ConvModule2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                in_channels,
                bias,
                padding_mode,
                dw_norm_layer,
                dw_act_layer,
            )
        self.pw_conv = ConvModule2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=bias,
                norm_layer=pw_norm_layer,
                act_layer=pw_act_layer,
            )

    def forward(self, x):
        x = self.dw_conv(x)
        out = self.pw_conv(x)
        return out


# @OBJECT_REGISTRY.register
class TemporalFusion(nn.Module):
    """Temporal fusion for bev feats.
    Args:
        in_channels: Channels for input.
        out_channels: Channels for ouput.
        num_seq: Number of sequence for multi frames.
        bev_size: Bev size.
        grid_size: Grid size.
        num_encoder: Number of encoder layers.
        num_project: Number of project layers.
        mode: Mode for grid sample.
        padding_mode: Padding mode for grid sample.
        grid_quant_scale: Quanti scale for grid sample.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_seq: int,
            bev_size: Tuple[float],
            grid_size: Tuple[float],
            num_encoder: int = 2,
            num_project: int = 1,
            mode: str = "bilinear",
            padding_mode: str = "zeros",
            align_corners: bool = True,
            grid_quant_scale: float = 1 / 512,
            grid_sample_with_delta=False,
    ):
        super(TemporalFusion, self).__init__()
        self.num_seq = num_seq
        self.bev_size = bev_size
        self.grid_size = grid_size
        self.in_channels = in_channels
        self.grid_sample_with_delta = grid_sample_with_delta

        encoder = nn.ModuleList()
        for i in range(num_encoder):
            encoder.append(
                SeparableConv2D(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    pw_norm_layer=nn.BatchNorm2d(out_channels),
                    pw_act_layer=nn.ReLU(inplace=True),
                )
            )
        self.encoder = nn.Sequential(*encoder)
        project = nn.ModuleList()
        for i in range(num_project):
            project.append(
                SeparableConv2D(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    pw_norm_layer=nn.BatchNorm2d(out_channels),
                    pw_act_layer=nn.ReLU(inplace=True),
                )
            )
        self.project = nn.Sequential(*project)

        self.coords = nn.Parameter(self._gen_coords(), requires_grad=False)

        self.offset = nn.Parameter(self._gen_offset(), requires_grad=False)
        if self.grid_sample_with_delta:
            self.grid_sample = GridsampleDelta(
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners
            )
        else:
            self.grid_sample = GridsampleNorm(
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
            )
        # self.quant_stub = QuantStub(grid_quant_scale)
        # self.quant = QuantStub()
        # self.dequant = DeQuantStub()


    # def _set_scale(self, feat: Tensor, prev_feat) -> Tensor:
    #     """Set the scale factor of a feature for activation quantization.
    #
    #     Args:
    #         feat: The input feature.
    #         prev_feat: The prev input feature.
    #
    #     Returns:
    #         feat: The input feature after setting the scale factor.
    #     """
    #
    #     if self.training and isinstance(feat, QTensor):
    #         self.quant.activation_post_process.scale = prev_feat.scale
    #     return feat, prev_feat

    def forward(
            self, feats: Tensor, meta: Dict, compile_model: bool, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Perform the forward pass through the modules.

        Args:
            feats: The input features.  bev_feat: (B*num_seq)*C*H*W
            meta: The meta information. ego2global: (B*num_seq)*4*4
            compile_model: A flag indicating whether to compile the model.

        Returns:
            feat: The output feature.
            prev_feat: The dequantized feature.
        """

        n, c, h, w = feats.shape
        feats = self.encoder(feats)
        fused_feat, prev_feat = self._fusion(feats, meta, compile_model)
        fused_feat, prev_feat = self._set_scale(fused_feat, prev_feat)
        prev_feat = self.dequant(prev_feat)
        return fused_feat, prev_feat

    def get_min_max_coords(self, bev_size: Tuple[float], ) -> Tuple[float, float, float, float]:
        """Get min and max coord of bev grid.

        Args:
            bev_size: [x_down_limit, x_up_limit, y_left_limit, y_right_limit, x_step, y_step]
        """

        min_x = -bev_size[0] + bev_size[4] / 2
        max_x = bev_size[1] - bev_size[4] / 2
        min_y = -bev_size[2] + bev_size[5] / 2
        max_y = bev_size[3] - bev_size[5] / 2

        return min_x, max_x, min_y, max_y


    def _gen_offset(self) -> None:
        """Generate a tensor for the offset values of a bird's eye view grid.

        Returns:
            The tensor containing the offset values.
        """

        W = self.grid_size[0]
        H = self.grid_size[1]

        # Generate a tensor for the x-coordinates of the bird's eye view grid
        bev_x = (
            torch.linspace(0, W - 1, W).reshape((1, W)).repeat(H, 1)
        ).double()
        bev_y = (
            torch.linspace(0, H - 1, H).reshape((H, 1)).repeat(1, W)
        ).double()

        bev_offset = torch.stack([bev_x, bev_y], axis=-1) * -1
        bev_offset = bev_offset.unsqueeze(0)
        return bev_offset

    def _gen_coords(self) -> None:
        """Generate a tensor for the coordinates of a bird's eye view grid.

        Returns:
            The tensor containing the world space coordinates.
        """

        # Get the minimum and maximum x and y coordinates
        # for the bird's eye view grid
        bev_min_x, bev_max_x, bev_min_y, bev_max_y = self.get_min_max_coords(
            self.bev_size
        )

        W = self.grid_size[0]
        H = self.grid_size[1]

        # Generate a tensor for the x-coordinates of the bird's eye view grid
        x = (
            torch.linspace(bev_min_x, bev_max_x, H)
            .reshape((H, 1))
            .repeat(1, W)
        ).double()
        y = (
            torch.linspace(bev_min_y, bev_max_y, W)
            .reshape((1, W))
            .repeat(H, 1)
        ).double()

        coords = torch.stack([x, y], dim=-1).unsqueeze(0)

        return coords

    def _get_matrix(
            self, meta: Dict, idx: int, bev_x: float, bev_y: float
    ) -> Tuple[np.array, np.array]:
        """Compute the transformation matrix.

        Components for warping a point on the bird's eye view grid
        between consecutive frames based on the provided meta information
        and the corresponding index.
        ego_pose: new_points = prev_g2e @ cur_e2g @ points
        bev_pose: [new_points - (min_x, min_y)} / scale

        Args:
            meta: The meta information.
            idx: The index corresponding to the frame of interest.
            bev_x: The x-coordinate on the bird's eye view grid.
            bev_y: The y-coordinate on the bird's eye view grid.

        Returns:
            wrap_r: The rotation component of the transformation matrix.
            wrap_t: The translation component of the transformation matrix.
        """

        # Get the ego to global transformation matrix
        ego2global = meta["ego2global"]

        if isinstance(ego2global, List) or isinstance(ego2global, np.ndarray):
            ego2global = np.array(ego2global).astype(np.float64)
        else:
            ego2global = ego2global.cpu().numpy().astype(np.float64)
            ego2global = ego2global.reshape(-1, self.num_seq, 4, 4)

        # Get the ego to global transformation matrix for the previous frame
        prev_e2g = ego2global[:, idx + 1]
        # Compute the inverse transformation matrix

        prev_g2e = np.linalg.inv(prev_e2g)
        # Get the ego to global transformation matrix for the current frame
        cur_e2g = ego2global[:, idx]
        # Compute the transformation matrix
        # by multiplying the inverse with the current
        wrap_m = prev_g2e @ cur_e2g
        # Extract the rotation component
        wrap_r = wrap_m[:, :2, :2].transpose((0, 2, 1))
        # Extract the translation component
        wrap_t = wrap_m[:, :2, 3]
        # Adjust the translation component
        # with the bird's eye view grid coordinates
        wrap_t = wrap_t + np.array([-bev_x, -bev_y])
        wrap_r /= self.bev_size[4]
        wrap_t /= self.bev_size[5]
        return wrap_r, wrap_t

    def _get_reference_points(
            self, feat: Tensor, meta: Dict, idx: int
    ) -> Tensor:
        """Compute the warped reference points on the bird's eye view grid.

        Args:
            feat: The feature tensor.
            meta: The meta information.
            idx: The index corresponding to the frame of interest.

        Returns:
            new_coords: The warped reference points
                        on the bird's eye view grid.
        """

        bev_min_x, bev_max_x, bev_min_y, bev_max_y = self.get_min_max_coords(
            self.bev_size
        )
        feat_hw = feat.shape[2:]
        wrap_r, wrap_t = self._get_matrix(meta, idx, bev_min_x, bev_min_y)
        wrap_r = torch.tensor(wrap_r).to(device=feat.device)
        wrap_t = torch.tensor(wrap_t).to(device=feat.device)

        # Compute the transformed coordinates
        new_coords = []
        batch = wrap_r.shape[0]
        for i in range(batch):
            new_coord = torch.matmul(self.coords, wrap_r[i]).float()
            new_coord += wrap_t[i]
            u = new_coord[..., 1:2]
            v = new_coord[..., 0:1]
            new_coord = torch.cat([u, v], dim=-1)
            new_coord = self.adjust_coord_grid(new_coord, feat_size=feat_hw)
            new_coords.append(new_coord)
        new_coords = torch.cat(new_coords)

        return new_coords

    def adjust_coord_grid(self, coords, feat_size=None):
        """
        adjust coords for gridsample
        Args:
            coords: (N, grid_h, grid_w, 2)
            feat_size: (feat_h, feat_w)

        Returns:

        """
        # grid sample with delta coords
        if self.grid_sample_with_delta:
            coords_grid = coords + self.offset
        # grid sample with norm coords
        else:
            norm_grid_x = coords[:, :, :, 0] * 2 / (feat_size[1] - 1) - 1
            norm_grid_y = coords[:, :, :, 1] * 2 / (feat_size[0] - 1) - 1
            coords_grid = torch.stack((norm_grid_x, norm_grid_y), dim=-1)

        return coords_grid

    def export_reference_points(self, feat, meta):

        prev_point = self._get_reference_points(feat, meta, 0)

        return {"prev_points": prev_point}

    def _transform(self, feat: Tensor, points: Tensor) -> Tensor:
        """Apply a spatial transformation to a feature tensor.

        Args:
            feat: The feature tensor to be transformed.
            points: The reference points for the transformation.

        Returns:
            feat: The transformed feature tensor.
        """

        feat = self.grid_sample(
            feat,
            points,
        )
        return feat

    def fuse_model(self) -> None:
        """Perform model fusion on the modules."""
        for mod_list in [self.encoder, self.project]:
            for mod in mod_list:
                if hasattr(mod, "fuse_model"):
                    mod.fuse_model()

    def _process_input(self, inputs: Any) -> Tensor:
        """Process the input data before further operations.

        Args:
            inputs: The input data to be processed.

        Returns:
            inputs: The processed input data.
        """

        # if placeholder is not None and isinstance(inputs, placeholder):
        #     inputs = inputs.sample
        return inputs

    def _fuse(self, cur, prev, new_coords):
        prev = self._transform(prev, new_coords)
        prev = self._fuse_op(prev, cur)
        prev = self.project(prev)
        return prev

    def _fusion(
            self, feats: Tensor, meta: Dict, compile_model: bool
    ) -> Tensor:
        """Perform the fusion operation on the input features.

        Args:
            feats: The input features.
            meta: The meta information.
            compile_model: A flag indicating whether to compile the model.

        Returns:
            fused_feat: The fused features.
        """
        if compile_model is True or ('prev_feats' in meta and 'prev_points' in meta):
            prev_feats = self._process_input(meta["prev_feats"])
            prev_feats = self.quant(prev_feats)
            prev_points = self._process_input(meta["prev_points"])
            n, c, h, w = feats.shape
            cur_feat = feats
        else:
            prev_points = []
            for i in range(0, self.num_seq - 1):
                new_coords = self._get_reference_points(feats, meta, i)
                prev_points.append(new_coords)
            prev_points = self._warp_stack(prev_points)  # [(T-1)*B]*H*W*2
            n, c, h, w = feats.shape
            feats = feats.view(-1, self.num_seq, c, h, w)
            prev_feats = (
                feats[:, 1:]
                .permute(1, 0, 2, 3, 4)
                .contiguous()
                .view(-1, c, h, w)
            )  # ([(T-1)*B]*c*h*w)
            cur_feat = feats[:, 0]  # (B)*c*h*w
        bs = cur_feat.shape[0]
        prev = prev_feats[(self.num_seq - 2) * bs:]  # last one
        prev_points = self.quant_stub(prev_points)

        for i in reversed(range(0, self.num_seq - 2)):
            cur = prev_feats[i * bs: (i + 1) * bs]  # cur
            new_coords = prev_points[(i + 1) * bs: (i + 2) * bs]
            prev = self._fuse(cur, prev, new_coords)
        fused_feat = self._fuse(cur_feat, prev, prev_points[0:bs])
        return fused_feat, cur_feat

    # @fx_wrap()
    def _warp_stack(self, prev_points):
        return torch.stack(prev_points, dim=0).flatten(0, 1)


# @OBJECT_REGISTRY.register
class GAddTemporalFusion(TemporalFusion):
    """Simple Add Temporal fusion for bev feats."""

    def __init__(self, **kwargs):
        super(GAddTemporalFusion, self).__init__(**kwargs)
        # self.floatFs = FloatFunctional()

    def _fuse_op(self, prev: Tensor, cur: Tensor) -> Tensor:
        """Fuse the previous and the current features.

        Args:
            prev: The previous features.
            cur: The current features.
        Returns:
            fused_features: The fused features.
        """

        # return self.floatFs.add(prev, cur)
        return self.torch.add(prev, cur)


if __name__ == "__main__":
    import cv2

    # bev coordinate: origin -> rear axle, x -> front, y -> left, z -> up
    # ego coordinate: origin -> rear axle, x -> front, y -> left, z -> up
    # gt coordinate: origin -> (back, right, down), x -> front, y -> left, z -> up
    # img coordinate: origin -> (up, left), u -> right, v -> down

    """
                                up z    x front (yaw=0)
                                    ^   ^
                                    |  /
                                    | /
                     left y <------ 0
    """

    # x-y bev range
    bev_size = (10, 22, 10.4, 10.4, 0.2, 0.2)  # (m)
    # grid-size (w, h, z)
    grid_size = (104, 160, 8)  # (w, h, z)
    # z bev range
    z_range = (-1.3, 1.8, 0.4)  # (m)

    # bev coord correspond to coordinate origin of ego
    # bev_coord: x -> front, y -> left, z -> up
    ori_x = int(bev_size[0] / (bev_size[0] + bev_size[1]) * grid_size[1])
    ori_y = int(bev_size[2] / (bev_size[2] + bev_size[3]) * grid_size[0])

    fusion_model = TemporalFusion(
        64, 64, 2, bev_size, grid_size,
        grid_sample_with_delta=False
    )

    # fake datasets
    ego2global_curr = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    feat_cur = torch.zeros(1, 1, grid_size[1], grid_size[0])

    rot_degree = math.radians(45)
    ego2global_prev = [
        [np.cos(rot_degree), -np.sin(rot_degree), 0, 0],
        [np.sin(rot_degree), np.cos(rot_degree), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]

    feat_prev = torch.zeros(1, 1, grid_size[1], grid_size[0])
    feat_prev[:, :, ori_x - 10: ori_x + 10, ori_y - 30: ori_y + 30] = 255
    img_feat_prev = feat_prev[0, 0, :, :].numpy().astype(np.uint8)
    # bev_coord -> img_coord
    cv2.imwrite("feat_prev.png", img_feat_prev[::-1, ::-1])

    meta = dict(
        ego2global=[[
            ego2global_curr,
            ego2global_prev
        ]]
    )
    feats = torch.cat([feat_cur, feat_prev], dim=0)
    print(feats.size())
    export_reference_points = fusion_model.export_reference_points(feats, meta)
    print(export_reference_points["prev_points"].size())
    new_feat = fusion_model._transform(feats[1:, ...], export_reference_points["prev_points"])
    img_feat_prv2cur = new_feat[0, 0, :, :].numpy().astype(np.uint8)

    # bev_coord -> img_coord
    cv2.imwrite("feat_prev2cur.png", img_feat_prv2cur[::-1, ::-1])



