# Standard
import gc
from pathlib import Path
import time

# PIP
from lpips import LPIPS
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ptflops import get_model_complexity_info

# Custom
from custom.softsplat.model import SoftSplat
from custom.vimeo.dataset import Vimeo
from helper.loss import SSIMLoss
from helper.metric import psnr


# Timing utilities
start_time = None


def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()


def end_timer_and_print():
    global start_time
    torch.cuda.synchronize()
    end_time = time.time()
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    memory = torch.cuda.max_memory_allocated() // 1024 // 1024
    print(f"Max memory used by tensors = {memory}MB")


def test(cfg):
    print(f"[ {cfg.model.flow_extractor} ]")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    work_dir = Path(cfg.work_dir).absolute()
    data_dir = work_dir / cfg.data_dir
    weight_dir = work_dir / cfg.weight_dir

    # Load data
    test_dataset = Vimeo(
        data_dir=data_dir,
        state="test",
        is_pt=False,
        is_aug=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Init model
    model = SoftSplat(cfg.model).to(device)
    model.eval()

    if cfg.flops:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                macs, params = get_model_complexity_info(model, (3, cfg.model.height, cfg.model.width))
                print("{:<30}  {:<8}".format("Computational complexity: ", macs))
                print("{:<30}  {:<8}".format("Number of parameters: ", params))
                return

    # Load model
    if cfg.name != "none":
        weight_path = weight_dir / f"{cfg.name}.pt"
        print(f"Load {cfg.name} model from {weight_path}")
        state_dict = torch.load(weight_path)
        model.load_state_dict(state_dict)

    # Set metrics
    if cfg.psnr:
        calculate_psnr = psnr
    if cfg.ssim:
        calculate_ssim = SSIMLoss().to(device)
    if cfg.lpips:
        calculate_lpips = LPIPS(net="alex", verbose=False).to(device)

    # Inference
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            start_timer()
            for batch in tqdm(test_dataloader):
                img1, img2, y = batch
                img1 = img1.to(device)
                img2 = img2.to(device)
                if cfg.psnr or cfg.ssim or cfg.lpips:
                    y = y.to(device)

                y_hat = model(img1, img2)
                if cfg.psnr:
                    total_psnr += calculate_psnr(y_hat, y)
                if cfg.ssim:
                    total_ssim += calculate_ssim(y_hat, y)
                if cfg.lpips:
                    total_lpips += calculate_lpips(y_hat, y).mean()

    end_timer_and_print()

    if cfg.psnr:
        average_psnr = total_psnr / len(test_dataloader)
        print(f"PSNR: {average_psnr:.2f}")
    if cfg.ssim:
        average_ssim = total_ssim / len(test_dataloader)
        print(f"SSIM: {average_ssim:.2f}")
    if cfg.lpips:
        average_lpips = total_lpips / len(test_dataloader)
        print(f"LPIPS: {average_lpips:.2f}")
