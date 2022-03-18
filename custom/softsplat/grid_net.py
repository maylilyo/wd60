# PIP
import torch.nn as nn


class LateralBlock(nn.Module):
    def __init__(self, ch_in, ch_out, is_last=False):
        super().__init__()
        self.is_diff_ch = ch_in != ch_out
        self.is_last = is_last

        self.conv_1 = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size=3,
            padding=1,
            dilation=1,
            bias=True,
        )
        self.conv_2 = nn.Conv2d(
            ch_out,
            ch_out,
            kernel_size=3,
            padding=1,
            dilation=1,
            bias=True,
        )
        self.group_norm = (nn.GroupNorm(ch_out // 2, ch_out),)
        self.leaky_relu = nn.LeakyReLU(0.2)

        if self.is_diff_ch:
            self.conv = nn.Conv2d(
                ch_in,
                ch_out,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=True,
            )

    def forward(self, x):
        x1 = self.conv_1(x)
        # x1 = self.group_norm(x1)
        x1 = self.leaky_relu(x1)
        x1 = self.conv_2(x1)
        if not self.is_last:
            # x1 = self.group_norm(x1)
            x1 = self.leaky_relu(x1)

        if self.is_diff_ch:
            x = self.conv(x)

        return x1 + x


class DownSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv_1 = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            bias=True,
        )
        self.conv_2 = nn.Conv2d(
            ch_out,
            ch_out,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=True,
        )
        self.group_norm = nn.GroupNorm(ch_out // 2, ch_out)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv_1(x)
        # x = self.group_norm(x)
        x = self.leaky_relu(x)
        x = self.conv_2(x)
        # x = self.group_norm(x)
        x = self.leaky_relu(x)
        return x


class UpSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_1 = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size=3,
            padding=1,
            dilation=1,
            bias=True,
        )
        self.conv_2 = nn.Conv2d(
            ch_out,
            ch_out,
            kernel_size=3,
            padding=1,
            dilation=1,
            bias=True,
        )
        self.group_norm = nn.GroupNorm(ch_out // 2, ch_out)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv_1(x)
        # x = self.group_norm(x)
        x = self.leaky_relu(x)
        x = self.conv_2(x)
        # x = self.group_norm(x)
        x = self.leaky_relu(x)
        return x


class GridNet(nn.Module):
    # Image Synthesis Network
    def __init__(
        self,
        num_out_channel=3,
        grid_channel_list=[32, 64, 96],
    ):
        super().__init__()

        num_column = 6

        self.lateral_init = LateralBlock(70, grid_channel_list[0])

        for r, num_channel in enumerate(grid_channel_list):
            for c in range(num_column - 1):
                setattr(self, f"lateral_{r}_{c}", LateralBlock(num_channel, num_channel))

        for r, (input_channel, output_channel) in enumerate(zip(grid_channel_list[:-1], grid_channel_list[1:])):
            for c in range(num_column // 2):
                # 00, 10일 때 예외처리
                if r == 0 and c == 0:
                    setattr(self, f"down_{r}_{c}", LateralBlock(128, output_channel))
                elif r == 1 and c == 0:
                    setattr(self, f"down_{r}_{c}", LateralBlock(192, output_channel))
                else:
                    setattr(
                        self,
                        f"down_{r}_{c}",
                        DownSamplingBlock(input_channel, output_channel),
                    )

        for r, (input_channel, output_channel) in enumerate(zip(grid_channel_list[1:], grid_channel_list[:-1])):
            for c in range(num_column // 2):
                setattr(self, f"up_{r}_{c}", UpSamplingBlock(input_channel, output_channel))

        self.lateral_final = LateralBlock(
            grid_channel_list[0],
            num_out_channel,
            is_last=True,
        )

    def forward(self, x_l1, x_l2, x_l3):
        # x_l1: [num_batches, 70, 256, 448]
        # x_l2: [num_batches, 128, 128, 224]
        # x_l3: [num_batches, 192, 64, 112]

        state_00 = self.lateral_init(x_l1)
        state_10 = self.down_0_0(x_l2)
        state_20 = self.down_1_0(x_l3)

        # 01의 입력을 warp로 바꾼 결과
        # Down Block Stage
        state_01 = self.lateral_0_0(state_00)
        state_11 = self.down_0_1(state_01) + self.lateral_1_0(state_10)
        state_21 = self.down_1_1(state_11) + self.lateral_2_0(state_20)

        state_02 = self.lateral_0_1(state_01)
        state_12 = self.down_0_2(state_02) + self.lateral_1_1(state_11)
        state_22 = self.down_1_2(state_12) + self.lateral_2_1(state_21)
        # state_00, 01, 02 : [num_batches, 32, 256, 448]
        # state_10, 11, 12 : [num_batches, 64, 128, 224]
        # state_20, 21, 22 : [num_batches, 96, 64, 112]

        # Up Block Stage
        state_23 = self.lateral_2_2(state_22)
        state_13 = self.up_1_0(state_23) + self.lateral_1_2(state_12)
        state_03 = self.up_0_0(state_13) + self.lateral_0_2(state_02)

        state_24 = self.lateral_2_3(state_23)
        state_14 = self.up_1_1(state_24) + self.lateral_1_3(state_13)
        state_04 = self.up_0_1(state_14) + self.lateral_0_3(state_03)

        state_25 = self.lateral_2_4(state_24)
        state_15 = self.up_1_2(state_25) + self.lateral_1_4(state_14)
        state_05 = self.up_0_2(state_15) + self.lateral_0_4(state_04)
        # state_23, 24, 25 : [num_batches, 96, 64, 112]
        # state_13, 14, 15 : [num_batches, 64, 128, 224]
        # state_03, 04, 05 : [num_batches, 32, 256, 448]

        out = self.lateral_final(state_05)
        # out : [num_batches, 3, 256, 448]
        return out
