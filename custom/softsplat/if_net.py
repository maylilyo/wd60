# PIP
import torch
import torch.nn as nn
import torch.nn.functional as F


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_planes, c // 2, 3, 2, 2, 2, bias=False),
            # nn.GroupNorm(c // 16, c // 2),
            nn.PReLU(c // 2),
            nn.Conv2d(c // 2, c, 3, 2, 2, 2, bias=False),
            # nn.GroupNorm(c // 16, c),
            nn.PReLU(c),
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
        self.convblock3 = nn.Sequential(
            self.conv(c, c, 3, 2, 2),
            self.conv(c, c, 3, 2, 2),
        )
        self.flow_conv = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1, bias=False),
            # nn.GroupNorm(c // 16, c // 2),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 4, 4, 2, 1, bias=False),
        )

    @staticmethod
    def conv(in_planes, out_planes, kernel_size, padding, dilation):
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            # nn.GroupNorm(out_planes // 16, out_planes),
            nn.PReLU(out_planes),
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
            if flow is not None:
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
        if flow is not None:
            x = torch.cat((x, flow), 1)

        feature = self.conv0(x)
        feature = self.convblock0(feature) + feature
        feature = self.convblock1(feature) + feature
        feature = self.convblock2(feature) + feature
        feature = self.convblock3(feature) + feature

        flow = self.flow_conv(feature)
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

        return flow


class IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(3 + 3 + 4, c=90)
        self.block1 = IFBlock(3 + 3 + 4, c=90)
        self.block2 = IFBlock(3 + 3 + 4, c=90)

    def forward(self, img1, img2, scale_list=[4, 2, 1]):
        # img1, img2: (batch_size, 3, height, width)

        [batch_size, channel, height, width] = img1.size()

        flow = torch.zeros([batch_size, 4, height, width]).type_as(img1)
        # flow: (batch_size, 4, height, width)

        block = [self.block0, self.block1, self.block2]

        for i in range(len(block)):
            new_flow1 = block[i](
                x=torch.cat((img1, img2), 1),
                flow=flow,
                scale=scale_list[i],
            )
            new_flow2 = block[i](
                x=torch.cat((img2, img1), 1),
                flow=torch.cat((flow[:, 2:], flow[:, :2]), 1),
                scale=scale_list[i],
            )
            new_flow = (new_flow1 + new_flow2) / 2
            flow = flow + new_flow

        flow_1to2 = flow[:, :2]
        flow_2to1 = flow[:, 2:]

        return flow_1to2, flow_2to1
