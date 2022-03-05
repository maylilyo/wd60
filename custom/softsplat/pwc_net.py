# Standard
from pathlib import Path

# PIP
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom
from .correlation import FunctionCorrelation

backwarp_tenGrid = {}
backwarp_tenPartial = {}


def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = (
            torch.linspace(
                -1.0 + (1.0 / tenFlow.shape[3]),
                1.0 - (1.0 / tenFlow.shape[3]),
                tenFlow.shape[3],
            )
            .view(1, 1, 1, -1)
            .repeat(1, 1, tenFlow.shape[2], 1)
        )
        tenVer = (
            torch.linspace(
                -1.0 + (1.0 / tenFlow.shape[2]),
                1.0 - (1.0 / tenFlow.shape[2]),
                tenFlow.shape[2],
            )
            .view(1, 1, -1, 1)
            .repeat(1, 1, 1, tenFlow.shape[3])
        )

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).cuda()
    # end

    if str(tenFlow.shape) not in backwarp_tenPartial:
        backwarp_tenPartial[str(tenFlow.shape)] = tenFlow.new_ones(
            [tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3]]
        )
    # end

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    )
    tmp = backwarp_tenPartial[str(tenFlow.shape)]
    tmp = tmp.type_as(tenInput)
    tenInput = torch.cat([tenInput, tmp], 1)

    grid = backwarp_tenGrid[str(tenFlow.shape)]
    grid = grid.type_as(tenFlow)
    tenOutput = F.grid_sample(
        input=tenInput,
        grid=(grid + tenFlow).permute(0, 2, 3, 1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    tenMask = tenOutput[:, -1:, :, :]
    tenMask[tenMask > 0.999] = 1.0
    tenMask[tenMask < 1.0] = 0.0

    tenOutput = tenOutput[:, :-1, :, :] * tenMask
    tenOutput = tenOutput.type_as(tenInput)
    return tenOutput


# end
class PWCNet(nn.Module):
    def __init__(self):
        super().__init__()

        class Extractor(nn.Module):
            def __init__(self):
                super().__init__()

                self.netOne = nn.Sequential(
                    nn.Conv2d(
                        in_channels=3,
                        out_channels=16,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(
                        in_channels=16,
                        out_channels=16,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(
                        in_channels=16,
                        out_channels=16,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netTwo = nn.Sequential(
                    nn.Conv2d(
                        in_channels=16,
                        out_channels=32,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netThr = nn.Sequential(
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netFou = nn.Sequential(
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=96,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(
                        in_channels=96,
                        out_channels=96,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(
                        in_channels=96,
                        out_channels=96,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netFiv = nn.Sequential(
                    nn.Conv2d(
                        in_channels=96,
                        out_channels=128,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netSix = nn.Sequential(
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=196,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(
                        in_channels=196,
                        out_channels=196,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(
                        in_channels=196,
                        out_channels=196,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

            # end

            def forward(self, tenInput):
                tenOne = self.netOne(tenInput)
                tenTwo = self.netTwo(tenOne)
                tenThr = self.netThr(tenTwo)
                tenFou = self.netFou(tenThr)
                tenFiv = self.netFiv(tenFou)
                tenSix = self.netSix(tenFiv)

                return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]

            # end

        # end

        class Decoder(nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                intPrevious = [
                    None,
                    None,
                    81 + 32 + 2 + 2,
                    81 + 64 + 2 + 2,
                    81 + 96 + 2 + 2,
                    81 + 128 + 2 + 2,
                    81,
                    None,
                ][intLevel + 1]
                intCurrent = [
                    None,
                    None,
                    81 + 32 + 2 + 2,
                    81 + 64 + 2 + 2,
                    81 + 96 + 2 + 2,
                    81 + 128 + 2 + 2,
                    81,
                    None,
                ][intLevel + 0]

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
                    self.fltBackwarp = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][
                        intLevel + 1
                    ]

                self.netOne = nn.Sequential(
                    nn.Conv2d(
                        in_channels=intCurrent,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netTwo = nn.Sequential(
                    nn.Conv2d(
                        in_channels=intCurrent + 128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netThr = nn.Sequential(
                    nn.Conv2d(
                        in_channels=intCurrent + 128 + 128,
                        out_channels=96,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netFou = nn.Sequential(
                    nn.Conv2d(
                        in_channels=intCurrent + 128 + 128 + 96,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netFiv = nn.Sequential(
                    nn.Conv2d(
                        in_channels=intCurrent + 128 + 128 + 96 + 64,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netSix = nn.Sequential(
                    nn.Conv2d(
                        in_channels=intCurrent + 128 + 128 + 96 + 64 + 32,
                        out_channels=2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )

            # end

            def forward(self, tenOne, tenTwo, objPrevious):
                tenFlow = None
                tenFeat = None

                if objPrevious is None:
                    tenFlow = None
                    tenFeat = None

                    tenVolume = F.leaky_relu(
                        input=FunctionCorrelation(tenOne=tenOne, tenTwo=tenTwo),
                        negative_slope=0.1,
                        inplace=False,
                    )

                    tenFeat = torch.cat([tenVolume], 1)

                elif objPrevious is not None:
                    tenFlow = self.netUpflow(objPrevious["tenFlow"])
                    tenFeat = self.netUpfeat(objPrevious["tenFeat"])

                    tenVolume = F.leaky_relu(
                        input=FunctionCorrelation(
                            tenOne=tenOne,
                            tenTwo=backwarp(
                                tenInput=tenTwo, tenFlow=tenFlow * self.fltBackwarp
                            ),
                        ),
                        negative_slope=0.1,
                        inplace=False,
                    )

                    tenFeat = torch.cat([tenVolume, tenOne, tenFlow, tenFeat], 1)

                # end

                tenFeat = torch.cat([self.netOne(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netTwo(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netThr(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netFou(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netFiv(tenFeat), tenFeat], 1)

                tenFlow = self.netSix(tenFeat)

                return {"tenFlow": tenFlow, "tenFeat": tenFeat}

            # end

        # end

        class Refiner(nn.Module):
            def __init__(self):
                super().__init__()

                self.netMain = nn.Sequential(
                    nn.Conv2d(
                        in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=2,
                        dilation=2,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=4,
                        dilation=4,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=96,
                        kernel_size=3,
                        stride=1,
                        padding=8,
                        dilation=8,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(
                        in_channels=96,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=16,
                        dilation=16,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                    ),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                    ),
                )

            # end

            def forward(self, tenInput):
                return self.netMain(tenInput)

            # end

        # end

        self.netExtractor = Extractor()

        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)

        self.netRefiner = Refiner()

    # end

    def forward(self, tenOne, tenTwo):
        tenOne = self.netExtractor(tenOne)
        tenTwo = self.netExtractor(tenTwo)

        objEstimate = self.netSix(tenOne[-1], tenTwo[-1], None)
        objEstimate = self.netFiv(tenOne[-2], tenTwo[-2], objEstimate)
        objEstimate = self.netFou(tenOne[-3], tenTwo[-3], objEstimate)
        objEstimate = self.netThr(tenOne[-4], tenTwo[-4], objEstimate)
        objEstimate = self.netTwo(tenOne[-5], tenTwo[-5], objEstimate)

        return (objEstimate["tenFlow"] + self.netRefiner(objEstimate["tenFeat"])) * 20.0
