# PIP
import torch.nn as nn


class SynthesisNet(nn.Module):
    # Frame Synthesis Network
    # https://github.com/sniklaus/softmax-splatting/issues/5
    # https://github.com/sniklaus/softmax-splatting/issues/17
    def __init__(
        self,
        channel_list=[32, 64, 96],
        num_column=6,
    ):
        super().__init__()

        self.lateral_0_0 = self.first_lateral(2 * (channel_list[0] + 3), channel_list[0])
        self.lateral_1_0 = self.first_lateral(2 * channel_list[1], channel_list[1])
        self.lateral_2_0 = self.first_lateral(2 * channel_list[2], channel_list[2])

        for row, channel in enumerate(channel_list):
            for col in range(1, num_column):
                setattr(self, f"lateral_{row}_{col}", self.lateral(channel, channel))

        for row in range(len(channel_list) - 1):
            for col in range(num_column):
                if col < num_column // 2:
                    setattr(self, f"downsampling_{row}_{col}", self.downsampling(channel_list[row], channel_list[row + 1]))
                else:
                    setattr(self, f"upsampling_{row}_{col}", self.upsampling(channel_list[row + 1], channel_list[row]))

        self.lateral_0_6 = self.lateral(channel_list[0], 3)

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

    def forward(self, x_0, x_1, x_2):
        # x_0: [num_batches, 70, height, width] warped frames + warped first-level features
        # x_1: [num_batches, 128, height / 2, width / 2] warped second-level features
        # x_2: [num_batches, 192, height / 4, width / 4] warped third-level features

        x_0 = self.lateral_0_0(x_0)
        x_1 = self.lateral_1_0(x_1)
        x_2 = self.lateral_2_0(x_2)

        x_1 = self.downsampling_0_0(x_0) + x_1
        x_2 = self.downsampling_1_0(x_1) + x_2

        x_0 = self.lateral_0_1(x_0)
        x_1 = self.lateral_1_1(x_1)
        x_2 = self.lateral_2_1(x_2)

        x_1 = self.downsampling_0_1(x_0) + x_1
        x_2 = self.downsampling_1_1(x_1) + x_2

        x_0 = self.lateral_0_2(x_0)
        x_1 = self.lateral_1_2(x_1)
        x_2 = self.lateral_2_2(x_2)

        x_1 = self.downsampling_0_2(x_0) + x_1
        x_2 = self.downsampling_1_2(x_1) + x_2

        x_0 = self.lateral_0_3(x_0)
        x_1 = self.lateral_1_3(x_1)
        x_2 = self.lateral_2_3(x_2)

        x_1 = self.upsampling_1_3(x_2) + x_1
        x_0 = self.upsampling_0_3(x_1) + x_0

        x_0 = self.lateral_0_4(x_0)
        x_1 = self.lateral_1_4(x_1)
        x_2 = self.lateral_2_4(x_2)

        x_1 = self.upsampling_1_4(x_2) + x_1
        x_0 = self.upsampling_0_4(x_1) + x_0

        x_0 = self.lateral_0_5(x_0)
        x_1 = self.lateral_1_5(x_1)
        x_2 = self.lateral_2_5(x_2)

        x_1 = self.upsampling_1_5(x_2) + x_1
        x_0 = self.upsampling_0_5(x_1) + x_0

        x_0 = self.lateral_0_6(x_0)
        return x_0
