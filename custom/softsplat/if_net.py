# PIP
import torch
import torch.nn as nn


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(
            self.conv(in_planes, c // 2, 3, 2, 1),
            self.conv(c // 2, c, 3, 2, 1),
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 4, 4, 2, 1),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 1, 4, 2, 1),
        )
        self.convblock0 = nn.Sequential(
            self.conv(c, c),
            self.conv(c, c)
        )
        self.convblock1 = nn.Sequential(
            self.conv(c, c),
            self.conv(c, c)
        )
        self.convblock2 = nn.Sequential(
            self.conv(c, c),
            self.conv(c, c)
        )
        self.convblock3 = nn.Sequential(
            self.conv(c, c),
            self.conv(c, c)
        )

    @staticmethod
    def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.PReLU(out_planes)
        )

    def forward(self, x, flow, scale):
        x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 1. / scale
        flow = torch.cat((x, flow), dim=1)
        feat = self.conv0(flow)
        feat = self.convblock0(feat) + feat
        feat = self.convblock1(feat) + feat
        feat = self.convblock2(feat) + feat
        feat = self.convblock3(feat) + feat
        flow = self.conv1(feat)
        mask = self.conv2(feat)
        flow = F.interpolate(flow, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False) * scale
        mask = F.interpolate(mask, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        return flow, mask


class IFNet(nn.Module):
    grid_cache = {}

    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 4, c=90)
        self.block1 = IFBlock(7 + 4, c=90)
        self.block2 = IFBlock(7 + 4, c=90)
        self.block_tea = IFBlock(10 + 4, c=90)

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
            grid = torch.cat([horizontal, vertical], dim=1)

            IFNet.grid_cache[key] = grid
        grid = grid.type_as(flow)

        flow = torch.cat(
            tensors=[flow[:, 0:1, :, :] / ((input_tensor.shape[3] - 1.0) / 2.0),
                     flow[:, 1:2, :, :] / ((input_tensor.shape[2] - 1.0) / 2.0)],
            dim=1,
        )

        grid = (grid + flow).permute(0, 2, 3, 1)
        return F.grid_sample(input=input_tensor, grid=grid, mode='bilinear', padding_mode='border', align_corners=True)

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

        for i in range(3):
            f0, m0 = block[i](
                x=torch.cat((warped_img1, warped_img2, mask), 1),
                flow=flow,
                scale=scale_list[i],
            )
            f1, m1 = block[i](
                x=torch.cat((warped_img2, warped_img1, -mask), 1),
                flow=torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                scale=scale_list[i],
            )

            # flow
            flow = flow + (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2

            # mask
            mask = mask + (m0 + (-m1)) / 2
            mask = torch.sigmoid(mask)

            # warp
            warped_img1 = warp(img1, flow[:, :2]) * mask
            warped_img2 = warp(img2, flow[:, 2:4]) * (1 - mask)

        return flow