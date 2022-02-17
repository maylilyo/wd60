# Standard
from pathlib import Path
import time

# PIP
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

# Custom
from dataset import Vimeo


def get_triplet_list(data_dir, state):
    triplet_list = []
    with open(data_dir / f"tri_{state}list.txt", "r") as info_file:
        sequence_list = info_file.readlines()
        if "" in sequence_list:
            sequence_list.remove("")
        for sequence in sequence_list:
            sequence = sequence.strip("\n")
            path_list = []
            for file_name in ["im1.png", "im2.png", "im3.png"]:
                path_list.append(data_dir / "sequences" / sequence / file_name)
            triplet_list.append(path_list)
    return triplet_list


def augmentation(img, function_name):
    if function_name == "none":
        return img
    if function_name == "vflip":
        return TF.vflip(img)
    if function_name == "hflip":
        return TF.hflip(img)


def convert(data_dir, pt_dir, state, is_aug=True):
    triplet_list = get_triplet_list(data_dir, state)

    count = 0
    for path_list in tqdm(triplet_list):
        img_list = []
        for img_path in path_list:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            img = img.transpose(2, 0, 1)
            img = img.astype(np.float32)
            img /= 255
            img = torch.from_numpy(img)
            img_list.append(img)

        [img1, target, img2] = img_list
        img_tensor = torch.stack((img1, img2, target), 0)

        if is_aug:
            function_list = ["none", "hflip"]
        else:
            function_list = ["none"]

        for function_name in function_list:
            img_tensor = augmentation(img_tensor, function_name)

            torch.save(img_tensor, pt_dir / state / f"{count}.pt")
            count += 1


def convert_to_pt():
    project_dir = Path(__file__).absolute().parent.parent.parent
    data_dir = project_dir / "data" / "vimeo_triplet"
    pt_dir = project_dir / "data" / "vimeo_pt"

    convert(data_dir, pt_dir, state="train")
    convert(data_dir, pt_dir, state="test", is_aug=False)


if __name__ == "__main__":
    convert_to_pt()
