# PIP
import torch
import torch.nn as nn


class ContextExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(2, 32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(2, 32),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=96,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(6, 96),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(6, 96),
            nn.ReLU(),
        )

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        return [layer1, layer2, layer3]


class Decoder(nn.Module):
    def __init__(self, num_layers):
        super().__init__()

        self.conv_relu = nn.Sequential(
            nn.Conv2d(
                in_channels=num_layers*32*2,
                out_channels=num_layers*32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(num_layers*2, num_layers*32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_layers*32,
                out_channels=num_layers*32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(num_layers*2, num_layers*32),
            nn.ReLU(),
        )

    def forward(self, x1, x2):
        x1 = torch.cat((x1, x2), 1)
        x1 = self.conv_relu(x1)
        return x1


class MatricUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # conv_img: Color를 유지하면서 3 channels를 12 channels로 변경
        self.conv_img = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=12,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(3, 12),
            nn.ReLU(),
        )
        # conv_metric: 양쪽 사진의 loss를 1 channel에서 4 channels로 바꿈
        # 동시에 background의 중요도를 계산
        self.conv_metric = nn.Sequential(
            nn.GroupNorm(1, 1),
            nn.Conv2d(
                in_channels=1,
                out_channels=4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(1, 4),
            nn.ReLU(),
        )
        self.down_l1 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(2, 32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(2, 32),
            nn.ReLU(),
        )
        self.down_l2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
        )
        self.down_l3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=96,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(6, 96),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(6, 96),
            nn.ReLU(),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(6, 96),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(6, 96),
            nn.ReLU(),
        )
        self.up_l3 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(
                in_channels=96,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
        )
        self.up_l2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(2, 32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(2, 32),
            nn.ReLU(),
        )
        self.up_l1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(1, 16),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(1, 16),
            nn.ReLU(),
        )
        self.decoder3 = Decoder(3)
        self.decoder2 = Decoder(2)
        self.decoder1 = Decoder(1)
        self.out_seq = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(1, 1),
            nn.ReLU(),
        )

    def forward(self, metric, img):
        conv_metric = self.conv_metric(metric)
        conv_img = self.conv_img(img)

        input_l0 = torch.cat([conv_metric, conv_img], 1)
        down_l1 = self.down_l1(input_l0)
        down_l2 = self.down_l2(down_l1)
        down_l3 = self.down_l3(down_l2)
        middle = self.middle(down_l3)
        up_l3 = self.decoder3(down_l3, middle)
        up_l2 = self.up_l3(up_l3)
        up_l2 = self.decoder2(down_l2, up_l2)
        up_l1 = self.up_l2(up_l2)
        up_l1 = self.decoder1(down_l1, up_l1)
        out = self.up_l1(up_l1)

        out = self.out_seq(out)
        return out
