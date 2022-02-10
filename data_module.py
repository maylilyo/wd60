# Standard
from pathlib import Path

# PIP
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

# Custom
from custom.vimeo.dataset import Vimeo


class CustomDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        work_dir = Path(cfg.common.work_dir).absolute()
        self.data_dir = work_dir / cfg.data_module.data_dir

    def setup(self, stage):
        self.train_dataset = Vimeo(data_dir=self.data_dir, state="train", is_pt=True)
        self.test_dataset = Vimeo(data_dir=self.data_dir, state="test", is_pt=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data_module.batch_size,
            shuffle=True,
            num_workers=self.cfg.data_module.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.data_module.batch_size,
            shuffle=False,
            num_workers=self.cfg.data_module.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.data_module.batch_size,
            shuffle=False,
            num_workers=self.cfg.data_module.num_workers,
            pin_memory=True,
        )
