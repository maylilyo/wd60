# PIP
import torch
import torch.nn.functional as F


def psnr(input, target):
    score = F.mse_loss(input, target)
    score = 1 / score
    score = torch.log10(score)
    score *= 10
    return score
