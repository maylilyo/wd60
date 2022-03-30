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
                padding=1,
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.PReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.PReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=96,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                padding=1,
            ),
            nn.PReLU(),
        )

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        return [layer1, layer2, layer3]

class MetricUNet(nn.Module):
    # https://github.com/sniklaus/softmax-splatting/issues/17
    # lateral_img: Color를 유지하면서 3 channels를 12 channels로 변경
    # lateral_metric: 사진의 loss를 1 channel에서 4 channels로 바꿈. 동시에 background의 중요도를 계산
    def __init__(self):
        super().__init__()
        self.lateral_img = self.first_lateral(3, 12)
        self.lateral_metric = self.first_lateral(1, 4)
        self.downsampling_1 = self.downsampling(16, 32)
        self.downsampling_2 = self.downsampling(32, 64)
        self.downsampling_3 = self.downsampling(64, 96)
        self.lateral_0 = self.lateral(16, 16)
        self.lateral_1 = self.lateral(32, 32)
        self.lateral_2 = self.lateral(64, 64)
        self.lateral_3 = self.lateral(96, 96)
        self.upsampling_1 = self.upsampling(32, 16)
        self.upsampling_2 = self.upsampling(64, 32)
        self.upsampling_3 = self.upsampling(96, 64)
        self.lateral_out = self.lateral(16, 1)

    def first_lateral(self, ch_in, ch_out):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=3,
                padding=1,
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=ch_out,
                out_channels=ch_out,
                kernel_size=3,
                padding=1,
            ),
        )

    def lateral(self, ch_in, ch_out):
        return nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=3,
                padding=1,
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=ch_out,
                out_channels=ch_out,
                kernel_size=3,
                padding=1,
            ),
        )

    def downsampling(self, ch_in, ch_out):
        return nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=ch_out,
                out_channels=ch_out,
                kernel_size=3,
                padding=1,
            ),
        )

    def upsampling(self, ch_in, ch_out):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=3,
                padding=1,
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=ch_out,
                out_channels=ch_out,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, img, metric):
        img = self.lateral_img(img)
        metric = self.lateral_metric(metric)
        x0 = torch.cat([img, metric], 1)

        x1 = self.downsampling_1(x0)  # w/2, h/2, 32
        x2 = self.downsampling_2(x1)  # w/4, h/4, 64
        x3 = self.downsampling_3(x2)  # w/8, h/8, 96

        x0 = self.lateral_0(x0)  # w/1, h/1, 16
        x1 = self.lateral_1(x1)  # w/2, h/2, 32
        x2 = self.lateral_2(x2)  # w/4, h/4, 64
        x3 = self.lateral_3(x3)  # w/8, h/8, 96

        x2 = self.upsampling_3(x3) + x2  # w/4, h/4, 64
        x1 = self.upsampling_2(x2) + x1  # w/2, h/2, 32
        x0 = self.upsampling_1(x1) + x0  # w/1, h/1, 16

        x0 = self.lateral_out(x0)
        return x0
