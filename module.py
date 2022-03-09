# Standard

# PIP
from ignite.metrics import PSNR, SSIM
import lpips
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    OneCycleLR,
    CosineAnnealingWarmRestarts,
    StepLR,
)

# Custom
from custom.softsplat.model import SoftSplat
import helper.loss as c_loss


class CustomModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.model = SoftSplat(cfg.model)

        self.criterion = self.get_loss_function()
        self.freeze_module(self.criterion)
        self.metric_psnr = PSNR(data_range=1.0, device=self.device)
        self.metric_ssim = SSIM(data_range=1.0, device=self.device)

        self.load_pretrained_model()

    def load_pretrained_model(self):
        # Load SoftSplat
        weight_dir = f"{self.cfg.common.work_dir}/weights"
        softsplat_path = f"{weight_dir}/softsplat.pt"
        state_dict = torch.load(softsplat_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)

        # Load estimator
        if self.cfg.model.flow_extractor == "pwcnet":
            # https://github.com/sniklaus/pytorch-pwc/blob/master/run.py#L259
            self.model.flow_extractor.load_state_dict(
                {
                    strKey.replace("module", "net"): tenWeight
                    for strKey, tenWeight in torch.hub.load_state_dict_from_url(
                        url="http://content.sniklaus.com/github/pytorch-pwc/network-chairs-things.pytorch",
                        file_name="pwc-chairs-things",
                    ).items()
                }
            )
        else:
            estimator_path = f"{weight_dir}/{self.cfg.model.flow_extractor}.pt"
            state_dict = torch.load(estimator_path, map_location=self.device)
            if self.cfg.model.flow_extractor in ["raft", "raft_s"]:
                for key in list(state_dict.keys()):
                    new_key = key.replace("module.", "")
                    state_dict[new_key] = state_dict.pop(key)
            self.model.flow_extractor.load_state_dict(state_dict, strict=True)

    def freeze_module(self, module):
        for name, child in module.named_children():
            for param in child.parameters():
                param.requires_grad = False
            self.freeze_module(child)

    def unfreeze_module(self, module):
        for name, child in module.named_children():
            for param in child.parameters():
                param.requires_grad = True
            self.unfreeze_module(child)

    def get_loss_function(self):
        name = self.cfg.module.criterion.lower()

        if name == "L1".lower():
            return nn.L1Loss()
        elif name == "MSE".lower():
            return nn.MSELoss()
        elif name == "MAE".lower():
            return nn.L1Loss()
        elif name == "CrossEntropy".lower():
            return nn.CrossEntropyLoss()
        elif name == "BCE".lower():
            return nn.BCEWithLogitsLoss()
        elif name == "LAP".lower():
            return c_loss.LapLoss()
        elif name == "LPIPS".lower():
            return lpips.LPIPS(net="vgg", verbose=False)

        raise ValueError(f"{name} is not on the custom criterion list!")

    def get_optimizer(self):
        name = self.cfg.optimizer.name.lower()

        if name == "SGD".lower():
            return torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.cfg.optimizer.lr,
                momentum=self.cfg.optimizer.momentum,
            )
        elif name == "Adam".lower():
            return torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.cfg.optimizer.lr,
            )
        elif name == "AdamW".lower():
            return torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.cfg.optimizer.lr,
                weight_decay=1e-5,
            )

        raise ValueError(f"{name} is not on the custom optimizer list!")

    def get_lr_scheduler(self, optimizer):
        name = self.cfg.lr_scheduler.name.lower()

        if name == "None".lower():
            return None
        if name == "OneCycleLR".lower():
            return OneCycleLR(
                optimizer=optimizer,
                max_lr=self.cfg.optimizer.lr,
                total_steps=self.cfg.trainer.max_epochs * 3268,  # batch_size=16: 3268
                anneal_strategy=self.cfg.lr_scheduler.anneal_strategy,
            )
        elif name == "CosineAnnealingWarmRestarts".lower():
            return CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=self.cfg.lr_scheduler.T_0,
                T_mult=self.cfg.lr_scheduler.T_mult,
                eta_min=self.cfg.lr_scheduler.eta_min,
            )
        elif name == "StepLR".lower():
            return StepLR(
                optimizer=optimizer,
                step_size=self.cfg.lr_scheduler.step_size,
                gamma=self.cfg.lr_scheduler.gamma,
            )

        raise ValueError(f"{name} is not on the custom scheduler list!")

    def forward(self, img1, img2):
        # img1: (batch_size, channel, width, height)
        # img2: (batch_size, channel, width, height)

        out = self.model(img1, img2)
        # out : (batch_size, channel, width, height)

        return out

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        self.lr_scheduler = self.get_lr_scheduler(optimizer)

        if self.lr_scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        img1, img2, y = batch
        # img1: (batch_size, channel, width, height)
        # img2: (batch_size, channel, width, height)
        # y: (batch_size, channel, width, height)

        y_hat = self(img1, img2)
        # y_hat: (batch_size, channel, width, height)

        loss = self.criterion(y_hat, y)
        self.metric_psnr.update((y_hat, y))
        psnr_score = self.metric_psnr.compute()
        self.metric_psnr.reset()
        self.metric_ssim.update((y_hat, y))
        ssim_score = self.metric_ssim.compute()
        self.metric_ssim.reset()

        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True)

        self.log("train_psnr", psnr_score, prog_bar=True)
        self.log("train_ssim", ssim_score, prog_bar=True)
        return loss

    def training_step_end(self, batch_parts):
        # losses from each GPU on DP strategy
        loss = batch_parts.mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img1, img2, y = batch
        # img1: (batch_size, channel, width, height)
        # img2: (batch_size, channel, width, height)
        # y: (batch_size, channel, width, height)

        y_hat = self(img1, img2)
        # y_hat: (batch_size, channel, width, height)

        loss = self.criterion(y_hat, y)
        self.metric_psnr.update((y_hat, y))
        psnr_score = self.metric_psnr.compute()
        self.metric_psnr.reset()
        self.metric_ssim.update((y_hat, y))
        ssim_score = self.metric_ssim.compute()
        self.metric_ssim.reset()

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_psnr", psnr_score, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_ssim", ssim_score, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        img1, img2, y = batch
        # img1: (batch_size, channel, width, height)
        # img2: (batch_size, channel, width, height)
        # y: (batch_size, channel, width, height)

        y_hat = self(img1, img2)
        # y_hat: (batch_size, channel, width, height)

        loss = self.criterion(y_hat, y)
        self.metric_psnr.update((y_hat, y))
        psnr_score = self.metric_psnr.compute()
        self.metric_psnr.reset()
        self.metric_ssim.update((y_hat, y))
        ssim_score = self.metric_ssim.compute()
        self.metric_ssim.reset()

        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_psnr", psnr_score, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_ssim", ssim_score, on_step=False, on_epoch=True, sync_dist=True)
