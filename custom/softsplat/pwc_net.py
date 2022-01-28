# Standard
from pathlib import Path

# PIP
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom
from .correlation import correlation

PROJECT_DIR = Path(__file__).absolute().parent.parent.parent
WEIGHT_DIR = PROJECT_DIR / 'weights'


class Extractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.netOne = self.get_CNN(3, 16)
        self.netTwo = self.get_CNN(16, 32)
        self.netThr = self.get_CNN(32, 64)
        self.netFou = self.get_CNN(64, 96)
        self.netFiv = self.get_CNN(96, 128)
        self.netSix = self.get_CNN(128, 196)

    def get_CNN(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(
                inplace=False,
                negative_slope=0.1,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(
                inplace=False,
                negative_slope=0.1,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(
                inplace=False,
                negative_slope=0.1,
            )
        )

    def forward(self, input):
        tenOne = self.netOne(input)
        tenTwo = self.netTwo(tenOne)
        tenThr = self.netThr(tenTwo)
        tenFou = self.netFou(tenThr)
        tenFiv = self.netFiv(tenFou)
        tenSix = self.netSix(tenFiv)

        return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]


class Decoder(nn.Module):
    tmp_list = [None, None, 81 + 32 + 2 + 2, 81 + 64 +
                2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None]

    grid_cache = {}
    partial_cache = {}

    def __init__(self, intLevel):
        super().__init__()

        intPrevious = self.tmp_list[intLevel + 1]
        intCurrent = self.tmp_list[intLevel]

        if intLevel < 6:
            self.netUpflow = nn.ConvTranspose2d(
                in_channels=2,
                out_channels=2,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        if intLevel < 6:
            self.netUpfeat = nn.ConvTranspose2d(
                in_channels=intPrevious + 128 + 128 + 96 + 64 + 32,
                out_channels=2,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        if intLevel < 6:
            self.fltBackwarp = [None, None, None, 5.0,
                                2.5, 1.25, 0.625, None][intLevel + 1]

        self.netOne = self.get_CNN(intCurrent, 128)
        self.netTwo = self.get_CNN(intCurrent + 128, 128)
        self.netThr = self.get_CNN(intCurrent + 128 + 128, 96)
        self.netFou = self.get_CNN(intCurrent + 128 + 128 + 96, 64)
        self.netFiv = self.get_CNN(intCurrent + 128 + 128 + 96 + 64, 32)
        self.netSix = nn.Sequential(
            nn.Conv2d(
                in_channels=intCurrent + 128 + 128 + 96 + 64 + 32,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

    def get_CNN(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(
                inplace=False,
                negative_slope=0.1,
            )
        )

    def backwarp(self, input, flow):
        key = str(flow.shape)

        if key in Decoder.grid_cache:
            backwarped = Decoder.grid_cache[key]
        else:
            horizontal = torch.linspace(
                start=-1.0 + (1.0 / flow.shape[3]),
                end=1.0 - (1.0 / flow.shape[3]),
                steps=flow.shape[3],
            )
            horizontal = horizontal.view(1, 1, 1, -1)
            horizontal = horizontal.expand(
                -1, -1, flow.shape[2], -1)

            vertical = torch.linspace(
                start=-1.0 + (1.0 / flow.shape[2]),
                end=1.0 - (1.0 / flow.shape[2]),
                steps=flow.shape[2],
            )
            vertical = vertical.view(1, 1, -1, 1)
            vertical = vertical.expand(
                -1, -1, -1, flow.shape[3])

            backwarped = torch.cat(
                tensors=[horizontal, vertical],
                dim=1,
            )
            backwarped = backwarped.type_as(flow)

        partial = flow.new_ones([flow.shape[0], 1, flow.shape[2], flow.shape[3]])
        partial = partial.type_as(flow)

        flow = torch.cat(
            tensors=[
                flow[:, 0:1, :, :] /
                ((input.shape[3] - 1.0) / 2.0),
                flow[:, 1:2, :, :] /
                ((input.shape[2] - 1.0) / 2.0)
            ],
            dim=1,
        )

        input = torch.cat([input, partial], dim=1)

        grid = backwarped + flow
        grid = grid.permute(0, 2, 3, 1)

        output = F.grid_sample(
            input=input,
            grid=grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )

        mask = output[:, -1:, :, :]
        mask[mask > 0.999] = 1.0
        mask[mask < 1.0] = 0.0

        return output[:, :-1, :, :] * mask

    def forward(self, first, second, prev_object):
        flow = None
        feature = None

        if prev_object is None:
            flow = None
            feature = None

            input = correlation(
                first=first,
                second=second,
            )
            volume = F.leaky_relu(
                input=input,
                negative_slope=0.1,
                inplace=False,
            )
            feature = torch.cat(
                tensors=[volume],
                dim=1,
            )
        else:
            flow = self.netUpflow(prev_object['flow'])
            feature = self.netUpfeat(prev_object['feature'])

            second = self.backwarp(
                input=second,
                flow=flow * self.fltBackwarp,
            )
            input = correlation(
                first=first,
                second=second,
            )
            volume = F.leaky_relu(
                input=input,
                negative_slope=0.1,
                inplace=False,
            )
            feature = torch.cat(
                tensors=[volume, first, flow, feature],
                dim=1,
            )

        feature = torch.cat([self.netOne(feature), feature], dim=1)
        feature = torch.cat([self.netTwo(feature), feature], dim=1)
        feature = torch.cat([self.netThr(feature), feature], dim=1)
        feature = torch.cat([self.netFou(feature), feature], dim=1)
        feature = torch.cat([self.netFiv(feature), feature], dim=1)

        flow = self.netSix(feature)

        return {
            'flow': flow,
            'feature': feature
        }


class Refiner(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )
        self.conv_2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2)
        self.conv_3 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=4,
            dilation=4,
        )
        self.conv_4 = nn.Conv2d(
            in_channels=128,
            out_channels=96,
            kernel_size=3,
            stride=1,
            padding=8,
            dilation=8,
        )
        self.conv_5 = nn.Conv2d(
            in_channels=96,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=16,
            dilation=16,
        )
        self.conv_6 = nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )
        self.conv_7 = nn.Conv2d(
            in_channels=32,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )
        self.leaky_relu = nn.LeakyReLU(
            inplace=False,
            negative_slope=0.1,
        )
        # self.refiner = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=in_ch,
        #         out_channels=128,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         dilation=1,
        #     ),
        #     nn.LeakyReLU(
        #         inplace=False,
        #         negative_slope=0.1,
        #     ),
        #     nn.Conv2d(
        #         in_channels=128,
        #         out_channels=128,
        #         kernel_size=3,
        #         stride=1,
        #         padding=2,
        #         dilation=2),
        #     nn.LeakyReLU(
        #         inplace=False,
        #         negative_slope=0.1,
        #     ),
        #     nn.Conv2d(
        #         in_channels=128,
        #         out_channels=128,
        #         kernel_size=3,
        #         stride=1,
        #         padding=4,
        #         dilation=4,
        #     ),
        #     nn.LeakyReLU(
        #         inplace=False,
        #         negative_slope=0.1,
        #     ),
        #     nn.Conv2d(
        #         in_channels=128,
        #         out_channels=96,
        #         kernel_size=3,
        #         stride=1,
        #         padding=8,
        #         dilation=8,
        #     ),
        #     nn.LeakyReLU(
        #         inplace=False,
        #         negative_slope=0.1,
        #     ),
        #     nn.Conv2d(
        #         in_channels=96,
        #         out_channels=64,
        #         kernel_size=3,
        #         stride=1,
        #         padding=16,
        #         dilation=16,
        #     ),
        #     nn.LeakyReLU(
        #         inplace=False,
        #         negative_slope=0.1,
        #     ),
        #     nn.Conv2d(
        #         in_channels=64,
        #         out_channels=32,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         dilation=1,
        #     ),
        #     nn.LeakyReLU(
        #         inplace=False,
        #         negative_slope=0.1,
        #     ),
        #     nn.Conv2d(
        #         in_channels=32,
        #         out_channels=2,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         dilation=1,
        #     )
        # )

    def forward(self, input):
        out = self.conv_1(input)
        out = self.leaky_relu(out)
        out = self.conv_2(out)
        out = self.leaky_relu(out)
        out = self.conv_3(out)
        out = self.leaky_relu(out)
        out = self.conv_4(out)
        out = self.leaky_relu(out)
        out = self.conv_5(out)
        out = self.leaky_relu(out)
        out = self.conv_6(out)
        out = self.leaky_relu(out)
        out = self.conv_7(out)
        return out


class PWCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.netExtractor = Extractor()
        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)
        self.refiner = Refiner(565)

        # state_dict = torch.load(WEIGHT_DIR / 'pwc.pt')
        # self.load_state_dict(state_dict)

    def forward(self, first, second):
        first = self.netExtractor(first)
        second = self.netExtractor(second)

        out = self.netSix(first[-1], second[-1], None)
        out = self.netFiv(first[-2], second[-2], out)
        out = self.netFou(first[-3], second[-3], out)
        out = self.netThr(first[-4], second[-4], out)
        out = self.netTwo(first[-5], second[-5], out)

        out = out['flow'] + self.refiner(out['feature'])
        return out
