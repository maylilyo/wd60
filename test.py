# PIP
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

# Custom
from evaluation import test


@hydra.main(config_path="conf", config_name="test")
def main(cfg: DictConfig) -> None:
    test(cfg)


if __name__ == "__main__":
    main()
