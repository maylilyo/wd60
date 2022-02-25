# Standard
from math import exp

# PIP
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class SSIMLoss(nn.Module):
    def __init__(
        self,
        window_size=10,
        size_average=True,
    ):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(self.channel)

    @staticmethod
    def gaussian(window_size, sigma):
        gauss = torch.Tensor(
            [
                exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                for x in range(window_size)
            ]
        )
        return gauss / gauss.sum()

    def create_window(self, channel):
        _1D_window = self.gaussian(self.window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(
            _2D_window.expand(
                channel, 1, self.window_size, self.window_size
            ).contiguous()
        )
        return window

    @staticmethod
    def _ssim(img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(channel)
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return self._ssim(
            img1, img2, window, self.window_size, channel, self.size_average
        )


class LapLoss(nn.Module):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super().__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None

    def forward(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = self.build_gauss_kernel(
                size=self.k_size,
                sigma=self.sigma,
                n_channels=input.shape[1],
            )
        pyr_input = self.laplacian_pyramid(input)
        pyr_target = self.laplacian_pyramid(target)

        weights = [1, 2, 4, 8, 16, 32]

        return sum(
            weights[i] * F.l1_loss(a, b)
            for i, (a, b) in enumerate(zip(pyr_input, pyr_target))
        ).mean()

    @staticmethod
    def build_gauss_kernel(size=5, sigma=1.0, n_channels=1):
        if size % 2 != 1:
            raise ValueError("kernel size must be uneven")
        grid = np.float32(np.mgrid[0:size, 0:size].T)
        gaussian = lambda x: np.exp((x - size // 2) ** 2 / (-2 * sigma**2)) ** 2
        kernel = np.sum(gaussian(grid), axis=2)
        kernel /= np.sum(kernel)
        # repeat same kernel across depth dimension
        kernel = np.tile(kernel, (n_channels, 1, 1))
        # conv weight should be (out_channels, groups/in_channels, h, w),
        # and since we have depth-separable convolution we want the groups dimension to be 1
        kernel = torch.FloatTensor(kernel[:, None, :, :])
        return kernel

    @staticmethod
    def conv_gauss(img, kernel):
        """convolve img with a gaussian kernel that has been built with build_gauss_kernel"""
        n_channels, _, kw, kh = kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode="replicate")
        kernel = kernel.type_as(img)
        return F.conv2d(img, kernel, groups=n_channels)

    def laplacian_pyramid(self, img):
        current = img
        pyr = []

        for level in range(self.max_levels):
            filtered = self.conv_gauss(current, self._gauss_kernel)
            diff = current - filtered
            pyr.append(diff)
            current = F.avg_pool2d(filtered, 2)

        pyr.append(current)
        return pyr
