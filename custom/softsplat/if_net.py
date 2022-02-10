# PIP
import torch
import torch.nn as nn
import torch.nn.functional as F


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_planes, c // 2, 1, bias=False),
            # nn.GroupNorm(c // 16, c // 2),
            nn.ReLU(),
            nn.Conv2d(c // 2, c // 2, 3, 2, 2, 2, bias=False),
            # nn.GroupNorm(c // 16, c // 2),
            nn.ReLU(),
            nn.Conv2d(c // 2, c, 1, bias=False),
            # nn.GroupNorm(c // 16, c),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 2, 2, 2, bias=False),
            # nn.GroupNorm(c // 16, c),
            nn.ReLU(),
        )
        self.convblock0 = nn.Sequential(
            self.conv(c, c, 3, 2, 2),
            self.conv(c, c, 3, 2, 2),
        )
        self.convblock1 = nn.Sequential(
            self.conv(c, c, 3, 2, 2),
            self.conv(c, c, 3, 2, 2),
        )
        self.convblock2 = nn.Sequential(
            self.conv(c, c, 3, 2, 2),
            self.conv(c, c, 3, 2, 2),
        )
        self.flow_conv = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 1, bias=False),
            # nn.GroupNorm(c // 16, c // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(c // 2, c // 2, 4, 2, 1, bias=False),
            # nn.GroupNorm(c // 16, c // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(c // 2, 4, 1, bias=False),
            # nn.GroupNorm(1, 4),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 4, 4, 2, 1, bias=False),
        )
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 1, bias=False),
            # nn.GroupNorm(c // 16, c // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(c // 2, c // 2, 4, 2, 1, bias=False),
            # nn.GroupNorm(c // 16, c // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(c // 2, 1, 1, bias=False),
            # nn.GroupNorm(1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False),
        )

    @staticmethod
    def conv(in_planes, out_planes, kernel_size, padding, dilation):
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                bias=False,
            ),
            # nn.GroupNorm(out_planes // 16, out_planes),
            nn.ReLU(),
            nn.Conv2d(
                out_planes,
                out_planes,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            # nn.GroupNorm(out_planes // 16, out_planes),
            nn.ReLU(),
        )

    def forward(self, x, flow, scale):
        if 1 < scale:
            x = F.interpolate(
                x,
                scale_factor=1.0 / scale,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
            flow = (
                F.interpolate(
                    flow,
                    scale_factor=1.0 / scale,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=False,
                )
                / scale
            )
        feature = torch.cat((x, flow), 1)

        feature = self.conv0(feature)
        feature = self.convblock0(feature) + feature
        feature = self.convblock1(feature) + feature
        feature = self.convblock2(feature) + feature

        flow = self.flow_conv(feature)
        mask = self.mask_conv(feature)
        flow = (
            F.interpolate(
                flow,
                scale_factor=scale,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
            * scale
        )
        mask = F.interpolate(
            mask,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )

        return flow, mask


class IFNet(nn.Module):
    grid_cache = {}

    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(3 + 3 + 1 + 4, c=128)
        self.block1 = IFBlock(3 + 3 + 1 + 4, c=128)
        self.block2 = IFBlock(3 + 3 + 1 + 4, c=128)

    def warp(self, input_tensor, flow):
        key = flow.shape
        if key in IFNet.grid_cache:
            grid = IFNet.grid_cache[key]
        else:
            horizontal = torch.linspace(-1.0, 1.0, flow.shape[3])
            horizontal = horizontal.view(1, 1, 1, flow.shape[3])
            horizontal = horizontal.expand(flow.shape[0], -1, flow.shape[2], -1)
            vertical = torch.linspace(-1.0, 1.0, flow.shape[2])
            vertical = vertical.view(1, 1, flow.shape[2], 1)
            vertical = vertical.expand(flow.shape[0], -1, -1, flow.shape[3])
            grid = torch.cat([horizontal, vertical], 1)

            IFNet.grid_cache[key] = grid
        grid = grid.type_as(flow)

        flow = torch.cat(
            tensors=[
                flow[:, 0:1, :, :] / ((input_tensor.shape[3] - 1.0) / 2.0),
                flow[:, 1:2, :, :] / ((input_tensor.shape[2] - 1.0) / 2.0),
            ],
            dim=1,
        )

        grid = (grid + flow).permute(0, 2, 3, 1)
        grid = F.grid_sample(
            input=input_tensor,
            grid=grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return grid

    def forward(self, img1, img2, scale_list=[4, 2, 1]):
        # img1, img2: (num_batches, 3, height, width)
        warped_img1 = img1
        warped_img2 = img2

        [batch_size, channel, height, width] = img1.size()

        flow = torch.zeros([batch_size, 4, height, width]).type_as(img1)
        mask = torch.zeros([batch_size, 1, height, width]).type_as(img1)
        # flow: (num_batches, 4, height, width)
        # mask: (num_batches, 1, height, width)

        block = [self.block0, self.block1, self.block2]

        for i in range(len(block)):
            flow0, mask0 = block[i](
                x=torch.cat((warped_img1, warped_img2, mask), 1),
                flow=flow,
                scale=scale_list[i],
            )
            flow1, mask1 = block[i](
                x=torch.cat((warped_img2, warped_img1, -mask), 1),
                flow=torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                scale=scale_list[i],
            )
            flow1 = torch.cat((flow1[:, 2:4], flow1[:, :2]), 1)

            # flow
            flow = flow + (flow0 + flow1) / 2

            # mask
            mask = mask + (mask0 + (-mask1)) / 2
            mask = F.hardsigmoid(mask)

            # warp
            warped_img1 = self.warp(img1, flow[:, :2]) * mask
            warped_img2 = self.warp(img2, flow[:, 2:4]) * (1 - mask)

        return flow
