import torch
import torch.nn as nn
import torch.nn.functional as F

from .corr import CorrBlock
from .extractor import BasicEncoder, SmallEncoder
from .update import BasicUpdateBlock, SmallUpdateBlock
from .utils import coords_grid, upflow2


class RAFT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.small:
            self.hidden_dim = 96
            self.context_dim = 64
        else:
            self.hidden_dim = 128
            self.context_dim = 128

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(
                output_dim=128,
                norm_fn="instance",
                dropout=args.dropout,
                is_list=True,
            )
            self.cnet = SmallEncoder(
                output_dim=self.hidden_dim + self.context_dim,
                norm_fn="none",
                dropout=args.dropout,
            )
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=self.hidden_dim)
        else:
            self.fnet = BasicEncoder(
                output_dim=256,
                norm_fn="instance",
                dropout=args.dropout,
                is_list=True,
            )
            self.cnet = BasicEncoder(
                output_dim=self.hidden_dim + self.context_dim,
                norm_fn="batch",
                dropout=args.dropout,
            )
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=self.hidden_dim)

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8)
        coords0 = coords0.type_as(img)
        coords1 = coords_grid(N, H // 8, W // 8)
        coords1 = coords0.type_as(img)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12):
        """Estimate optical flow between pair of frames"""
        image1 = 2 * image1 - 1.0
        image2 = 2 * image2 - 1.0

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        for itr in range(iters):
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

        # upsample predictions
        if self.args.small:
            return upflow2(coords1 - coords0)
        return self.upsample_flow(coords1 - coords0, up_mask)
