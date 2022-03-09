# Standard
from math import exp

# PIP
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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

        return sum(weights[i] * F.l1_loss(a, b) for i, (a, b) in enumerate(zip(pyr_input, pyr_target))).mean()

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
