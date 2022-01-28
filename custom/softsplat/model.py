# PIP
import torch
import torch.nn as nn
from torch.nn.functional import interpolate, l1_loss

# Custom
from .grid_net import GridNet
from .if_net import IFNet
from .modules import ContextExtractor, MatricUNet
from .pwc_net import PWCNet
from .softmax_splatting import softmax_splatting


class SoftSplat(nn.Module):
    backwarp_cache = {}

    def __init__(self, model_option):
        super().__init__()
        self.flow_net_name = model_option.flow_extractor
        self.height = model_option.height
        self.width = model_option.width

        self.feature_extractor_1 = ContextExtractor()
        self.feature_extractor_2 = ContextExtractor()
        self.beta1 = nn.Parameter(torch.Tensor([-1]))
        self.beta2 = nn.Parameter(torch.Tensor([-1]))
        self.matric_unet = MatricUNet()
        self.grid_net = GridNet()

        if self.flow_net_name == 'pwcnet':
            self.flow_extractor = PWCNet()
        elif self.flow_net_name == 'ifnet':
            self.flow_extractor = IFNet()

    def backwarp(self, input_tensor, flow):
        key = flow.shape
        if key in SoftSplat.backwarp_cache:
            backwarped = SoftSplat.backwarp_cache[key]
        else:
            horizontal = torch.linspace(
                start=-1.0 + (1.0 / flow.shape[3]),
                end=1.0 - (1.0 / flow.shape[3]),
                steps=flow.shape[3],
            )
            horizontal = horizontal.view(1, 1, 1, -1)
            horizontal = horizontal.expand(-1, -1, flow.shape[2], -1)
            vertical = torch.linspace(
                start=-1.0 + (1.0 / flow.shape[2]),
                end=1.0 - (1.0 / flow.shape[2]),
                steps=flow.shape[2],
            )
            vertical = vertical.view(1, 1, -1, 1)
            vertical = vertical.expand(-1, -1, -1, flow.shape[3])
            backwarped = torch.cat([horizontal, vertical], dim=1)

            SoftSplat.backwarp_cache[key] = backwarped
        backwarped = backwarped.type_as(flow)

        flow = torch.cat(
            tensors=[
                flow[:, 0:1, :, :] / ((input_tensor.shape[3] - 1.0) / 2.0),
                flow[:, 1:2, :, :] / ((input_tensor.shape[2] - 1.0) / 2.0)
            ],
            dim=1,
        )

        grid = backwarped + flow
        grid = grid.permute(0, 2, 3, 1)

        return F.grid_sample(
            input=input_tensor,
            grid=grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )

    def scale_flow_zero(self, flow):
        SCALE = 20.0

        raw_scaled = (SCALE / 1) * interpolate(
            input=flow,
            size=(self.height, self.width),
            mode='bilinear',
            align_corners=False,
        )
        return raw_scaled

    def scale_flow(self, flow):
        # https://github.com/sniklaus/softmax-splatting/issues/12

        if self.flow_net_name == 'pwcnet':
            SCALE = 20.0
        elif self.flow_net_name == 'ifnet':
            SCALE = 1.0

        raw_scaled = (SCALE / 1) * interpolate(
            input=flow,
            size=(self.height, self.width),
            mode='bilinear',
            align_corners=False,
        )
        half_scaled = (SCALE / 2) * interpolate(
            input=flow,
            size=(self.height // 2, self.width // 2),
            mode='bilinear',
            align_corners=False,
        )
        quarter_scaled = (SCALE / 4) * interpolate(
            input=flow,
            size=(self.height // 4, self.width // 4),
            mode='bilinear',
            align_corners=False,
        )

        return [raw_scaled, half_scaled, quarter_scaled]

    def scale_tenMetric(self, tenMetric):

        raw_scaled = tenMetric
        half_scaled = interpolate(
            input=tenMetric,
            size=(self.height // 2, self.width // 2),
            mode='bilinear',
            align_corners=False,
        )
        quarter_scaled = interpolate(
            input=tenMetric,
            size=(self.height // 4, self.width // 4),
            mode='bilinear',
            align_corners=False,
        )
        return [raw_scaled, half_scaled, quarter_scaled]

    def forward(self, img1, img2):
        # img1, img2: (num_batches, 3, height, width)

        # ↓ Optional Information Provider
        feature_pyramid1 = self.feature_extractor_1(img1)
        feature_pyramid2 = self.feature_extractor_2(img2)

        # feature_pyramid1, feature_pyramid2: [layer1, layer2, layer3]
        # layer1: (num_batches, 32, height, width)
        # layer2: (num_batches, 64, height / 2, width / 2)
        # layer3: (num_batches, 96, height / 4, width / 4)

        # ↓ Optical Flow Estimator
        if self.flow_net_name == 'pwcnet':
            flow_1to2 = self.flow_extractor(img1, img2)
            flow_2to1 = self.flow_extractor(img2, img1)
            # flow_1to2, flow_2to1: (num_batches, 2, height / 4, width / 4)

            flow_1to2_zero = self.scale_flow_zero(flow_1to2)
            flow_2to1_zero = self.scale_flow_zero(flow_2to1)

            flow_1tot = flow_1to2 * 0.5
            flow_2tot = flow_2to1 * 0.5

        elif self.flow_net_name == 'ifnet':
            flow_all = self.flow_extractor(img1, img2)
            flow_1tot = flow_all[:, :2]
            flow_1to2_zero = flow_1tot
            flow_2tot = flow_all[:, 2:]
            flow_2to1_zero = flow_2tot

        flow_1to2_pyramid = self.scale_flow(flow_1tot)
        flow_2to1_pyramid = self.scale_flow(flow_2tot)

        target_1to2 = backwarp(img2, flow_1to2_zero)
        target_2to1 = backwarp(img1, flow_2to1_zero)

        # flow_1to2_pyramid, flow_2to1_pyramid: [raw_scaled, half_scaled, quarter_scaled]
        # raw_scaled: (num_batches, 2, height, width)
        # half_scaled: (num_batches, 2, height / 2, width / 2)
        # quarter_scaled: (num_batches, 2, height / 4, width / 4)

        # ↓ Softmax Splatting
        tenMetric_1to2 = l1_loss(
            input=img1,
            target=target_1to2,
            reduction='none',
        )
        # tenMetric_1to2: (num_batches, 3, height, width)

        tenMetric_1to2 = tenMetric_1to2.mean(1, True)
        # tenMetric_1to2: (num_batches, 1, height, width)

        tenMetric_1to2 = self.matric_unet(tenMetric_1to2, img1)
        # tenMetric_1to2: (num_batches, 1, height, width)

        tenMetric_ls_1to2 = self.scale_tenMetric(tenMetric_1to2)
        # tenMetric_ls_1to2: [raw_scaled, half_scaled, quarter_scaled]
        # raw_scaled: (num_batches, 1, height, width)
        # half_scaled: (num_batches, 1, height / 2, width / 2)
        # quarter_scaled: (num_batches, 1, height / 4, width / 4)

        warped_img1 = softmax_splatting(
            tenInput=img1,
            tenFlow=flow_1to2_pyramid[0],
            tenMetric=self.beta1 * tenMetric_ls_1to2[0],
            _type='softmax',
        )
        warped_pyramid1_1 = softmax_splatting(
            tenInput=feature_pyramid1[0],
            tenFlow=flow_1to2_pyramid[0],
            tenMetric=self.beta1 * tenMetric_ls_1to2[0],
            _type='softmax',
        )
        warped_pyramid1_2 = softmax_splatting(
            tenInput=feature_pyramid1[1],
            tenFlow=flow_1to2_pyramid[1],
            tenMetric=self.beta1 * tenMetric_ls_1to2[1],
            _type='softmax',
        )
        warped_pyramid1_3 = softmax_splatting(
            tenInput=feature_pyramid1[2],
            tenFlow=flow_1to2_pyramid[2],
            tenMetric=self.beta1 * tenMetric_ls_1to2[2],
            _type='softmax',
        )
        # warped_img1: (num_batches, 3, height, width)
        # warped_pyramid1_1: (num_batches, 32, height, width)
        # warped_pyramid1_2: (num_batches, 64, height / 2, width / 2)
        # warped_pyramid1_3: (num_batches, 96, height / 4, width / 4)

        tenMetric_2to1 = l1_loss(
            input=img2,
            target=target_2to1,
            reduction='none',
        )
        tenMetric_2to1 = tenMetric_2to1.mean(1, True)
        tenMetric_2to1 = self.matric_unet(tenMetric_2to1, img2)
        tenMetric_ls_2to1 = self.scale_tenMetric(tenMetric_2to1)

        warped_img2 = softmax_splatting(
            tenInput=img2,
            tenFlow=flow_2to1_pyramid[0],
            tenMetric=self.beta2 * tenMetric_ls_2to1[0],
            _type='softmax',
        )
        warped_pyramid2_1 = softmax_splatting(
            tenInput=feature_pyramid2[0],
            tenFlow=flow_2to1_pyramid[0],
            tenMetric=self.beta2 * tenMetric_ls_2to1[0],
            _type='softmax',
        )
        warped_pyramid2_2 = softmax_splatting(
            tenInput=feature_pyramid2[1],
            tenFlow=flow_2to1_pyramid[1],
            tenMetric=self.beta2 * tenMetric_ls_2to1[1],
            _type='softmax',
        )
        warped_pyramid2_3 = softmax_splatting(
            tenInput=feature_pyramid2[2],
            tenFlow=flow_2to1_pyramid[2],
            tenMetric=self.beta2 * tenMetric_ls_2to1[2],
            _type='softmax',
        )
        # warped_img2: (num_batches, 3, height, width)
        # warped_pyramid2_1: (num_batches, 32, height, width)
        # warped_pyramid2_2: (num_batches, 64, height / 2, width / 2)
        # warped_pyramid2_3: (num_batches, 96, height / 4, width / 4)

        # ↓ Image Synthesis Network
        grid_input_l1 = torch.cat(
            [warped_img1, warped_pyramid1_1, warped_img2, warped_pyramid2_1],
            dim=1,
        )
        grid_input_l2 = torch.cat(
            [warped_pyramid1_2, warped_pyramid2_2],
            dim=1,
        )
        grid_input_l3 = torch.cat(
            [warped_pyramid1_3, warped_pyramid2_3],
            dim=1,
        )
        # grid_input_l1: (num_batches, 70, height, width)
        # grid_input_l2: (num_batches, 128, height / 2, width / 2)
        # grid_input_l3: (num_batches, 192, height / 4, width / 4)

        out = self.grid_net(grid_input_l1, grid_input_l2, grid_input_l3)
        # out: (num_batches, 3, height, width)

        return out
