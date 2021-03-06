# PIP
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom
from .if_net import IFNet
from .modules import ContextExtractor, MetricUNet
from .pwc_net import PWCNet
from .softmax_splatting import softmax_splatting
from .synthesis_net import SynthesisNet
from .raft.model import RAFT
from helper.flow_visualization import flow_visualization


class SoftSplat(nn.Module):
    backwarp_cache = {}

    def __init__(self, model_option):
        super().__init__()
        self.flow_net_name = model_option.flow_extractor

        self.feature_extractor = ContextExtractor()
        self.alpha = nn.Parameter(torch.ones(1))
        self.metric_unet = MetricUNet()
        self.synthesis_net = SynthesisNet(
            channel_list=[32, 64, 96],
            num_column=6,
        )
        self.l1_loss = nn.L1Loss(reduction="none")

        if self.flow_net_name == "pwcnet":
            self.flow_extractor = PWCNet()
        elif self.flow_net_name == "ifnet":
            self.flow_extractor = IFNet()
        elif self.flow_net_name in ["raft", "raft_s"]:
            self.flow_extractor = RAFT(model_option.raft)

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
            horizontal = horizontal.repeat(1, 1, flow.shape[2], 1)
            vertical = torch.linspace(
                start=-1.0 + (1.0 / flow.shape[2]),
                end=1.0 - (1.0 / flow.shape[2]),
                steps=flow.shape[2],
            )
            vertical = vertical.view(1, 1, -1, 1)
            vertical = vertical.repeat(1, 1, 1, flow.shape[3])
            backwarped = torch.cat([horizontal, vertical], 1)

            SoftSplat.backwarp_cache[key] = backwarped
        backwarped = backwarped.type_as(flow)

        flow = torch.cat(
            tensors=[
                flow[:, 0:1, :, :] / ((input_tensor.shape[3] - 1.0) / 2.0),
                flow[:, 1:2, :, :] / ((input_tensor.shape[2] - 1.0) / 2.0),
            ],
            dim=1,
        )

        grid = backwarped + flow
        grid = grid.permute(0, 2, 3, 1)

        return F.grid_sample(
            input=input_tensor,
            grid=grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    def scale_flow_zero(self, flow):
        raw_scaled = F.interpolate(
            input=flow,
            scale_factor=4,
            mode="bilinear",
            align_corners=True,
        )
        return raw_scaled

    def scale_flow(self, flow):
        # https://github.com/sniklaus/softmax-splatting/issues/12

        raw_scaled = F.interpolate(
            input=flow,
            scale_factor=4,
            mode="bilinear",
            align_corners=True,
        )
        half_scaled = (
            F.interpolate(
                input=flow,
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
            / 2.0
        )
        quarter_scaled = flow / 4.0
        return [raw_scaled, half_scaled, quarter_scaled]

    def scale_tenMetric(self, metric):
        half_scaled = F.interpolate(
            input=metric,
            scale_factor=0.5,
            mode="bilinear",
            align_corners=True,
        )
        quarter_scaled = F.interpolate(
            input=metric,
            scale_factor=0.25,
            mode="bilinear",
            align_corners=True,
        )
        return [metric, half_scaled, quarter_scaled]

    def forward(self, img1, img2=None):
        # For flops calculation
        if img2 is None:
            img2 = img1
        # img1, img2: (num_batches, 3, height, width)

        # ??? Optical Flow Estimator
        if self.flow_net_name == "pwcnet":
            with torch.cuda.amp.autocast(enabled=False):
                flow_1to2 = self.flow_extractor(img1, img2)
                flow_2to1 = self.flow_extractor(img2, img1)
                # flow_1to2, flow_2to1: (num_batches, 2, height / 4, width / 4)
        elif self.flow_net_name in ["raft", "raft_s"]:
            flow_1to2 = self.flow_extractor(img1, img2)
            flow_2to1 = self.flow_extractor(img2, img1)
            # flow_1to2, flow_2to1: (num_batches, 2, height, width)
        elif self.flow_net_name == "ifnet":
            flow_1to2, flow_2to1 = self.flow_extractor(img1, img2)
            # flow_1to2, flow_2to1: (num_batches, 2, height, width)

        flow_1to2_zero = self.scale_flow_zero(flow_1to2)
        flow_2to1_zero = self.scale_flow_zero(flow_2to1)
        target_1to2 = self.backwarp(img2, flow_1to2_zero)
        target_2to1 = self.backwarp(img1, flow_2to1_zero)

        flow_1tot = flow_1to2 * 0.5
        flow_2tot = flow_2to1 * 0.5
        flow_1tot_pyramid = self.scale_flow(flow_1tot)
        flow_2tot_pyramid = self.scale_flow(flow_2tot)
        # flow_1tot_pyramid, flow_2tot_pyramid: [raw_scaled, half_scaled, quarter_scaled]
        # raw_scaled: (num_batches, 2, height, width)
        # half_scaled: (num_batches, 2, height / 2, width / 2)
        # quarter_scaled: (num_batches, 2, height / 4, width / 4)

        # ??? Optional Information Provider
        feature_pyramid1 = self.feature_extractor(img1)
        feature_pyramid2 = self.feature_extractor(img2)
        # feature_pyramid1, feature_pyramid2: [layer1, layer2, layer3]
        # layer1: (num_batches, 32, height, width)
        # layer2: (num_batches, 64, height / 2, width / 2)
        # layer3: (num_batches, 96, height / 4, width / 4)

        # ??? Softmax Splatting
        tenMetric_1to2 = self.l1_loss(img1, target_1to2)
        # tenMetric_1to2: (num_batches, 3, height, width)

        tenMetric_1to2 = tenMetric_1to2.mean(1, True)
        # tenMetric_1to2: (num_batches, 1, height, width)

        tenMetric_1to2 = self.metric_unet(img1, tenMetric_1to2)
        # tenMetric_1to2: (num_batches, 1, height, width)

        tenMetric_ls_1to2 = self.scale_tenMetric(tenMetric_1to2)
        # tenMetric_ls_1to2: [raw_scaled, half_scaled, quarter_scaled]
        # raw_scaled: (num_batches, 1, height, width)
        # half_scaled: (num_batches, 1, height / 2, width / 2)
        # quarter_scaled: (num_batches, 1, height / 4, width / 4)

        with torch.cuda.amp.autocast(enabled=False):
            warped_img1 = softmax_splatting(
                input=img1,
                flow=flow_1tot_pyramid[0],
                metric=self.alpha * tenMetric_ls_1to2[0],
            )
            warped_pyramid1_1 = softmax_splatting(
                input=feature_pyramid1[0],
                flow=flow_1tot_pyramid[0],
                metric=self.alpha * tenMetric_ls_1to2[0],
            )
            warped_pyramid1_2 = softmax_splatting(
                input=feature_pyramid1[1],
                flow=flow_1tot_pyramid[1],
                metric=self.alpha * tenMetric_ls_1to2[1],
            )
            warped_pyramid1_3 = softmax_splatting(
                input=feature_pyramid1[2],
                flow=flow_1tot_pyramid[2],
                metric=self.alpha * tenMetric_ls_1to2[2],
            )
            # warped_img1: (num_batches, 3, height, width)
            # warped_pyramid1_1: (num_batches, 32, height, width)
            # warped_pyramid1_2: (num_batches, 64, height / 2, width / 2)
            # warped_pyramid1_3: (num_batches, 96, height / 4, width / 4)

        tenMetric_2to1 = self.l1_loss(img2, target_2to1)
        tenMetric_2to1 = tenMetric_2to1.mean(1, True)
        tenMetric_2to1 = self.metric_unet(img2, tenMetric_2to1)
        tenMetric_ls_2to1 = self.scale_tenMetric(tenMetric_2to1)

        with torch.cuda.amp.autocast(enabled=False):
            warped_img2 = softmax_splatting(
                input=img2,
                flow=flow_2tot_pyramid[0],
                metric=self.alpha * tenMetric_ls_2to1[0],
            )
            warped_pyramid2_1 = softmax_splatting(
                input=feature_pyramid2[0],
                flow=flow_2tot_pyramid[0],
                metric=self.alpha * tenMetric_ls_2to1[0],
            )
            warped_pyramid2_2 = softmax_splatting(
                input=feature_pyramid2[1],
                flow=flow_2tot_pyramid[1],
                metric=self.alpha * tenMetric_ls_2to1[1],
            )
            warped_pyramid2_3 = softmax_splatting(
                input=feature_pyramid2[2],
                flow=flow_2tot_pyramid[2],
                metric=self.alpha * tenMetric_ls_2to1[2],
            )
            # warped_img2: (num_batches, 3, height, width)
            # warped_pyramid2_1: (num_batches, 32, height, width)
            # warped_pyramid2_2: (num_batches, 64, height / 2, width / 2)
            # warped_pyramid2_3: (num_batches, 96, height / 4, width / 4)

        # ??? Image Synthesis Network
        synthesis_1 = torch.cat(
            [warped_img1, warped_pyramid1_1, warped_img2, warped_pyramid2_1],
            dim=1,
        )
        synthesis_2 = torch.cat(
            [warped_pyramid1_2, warped_pyramid2_2],
            dim=1,
        )
        synthesis_3 = torch.cat(
            [warped_pyramid1_3, warped_pyramid2_3],
            dim=1,
        )
        # synthesis_1: (num_batches, 70, height, width)
        # synthesis_2: (num_batches, 128, height / 2, width / 2)
        # synthesis_3: (num_batches, 192, height / 4, width / 4)

        out = self.synthesis_net(synthesis_1, synthesis_2, synthesis_3)
        # out: (num_batches, 3, height, width)

        out = torch.clamp(out, min=0.0, max=1.0)

        return out
