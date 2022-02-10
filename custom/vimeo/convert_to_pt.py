# Standard
from pathlib import Path
import time

# PIP
import cv2
import numpy as np
import torch
from tqdm import tqdm

# Custom
from dataset import Vimeo


def convert_to_pt():
    project_dir = Path(__file__).absolute().parent.parent.parent
    data_dir = project_dir / "data" / "vimeo_triplet"
    tri_img = ["im1.png", "im2.png", "im3.png"]
    state_list = ["train", "test"]

    for state in state_list:
        total_res_ls = []
        with open(data_dir / f"tri_{state}list.txt", "r") as info_file:
            sequence_list = info_file.readlines()
            if "" in sequence_list:
                sequence_list.remove("")
            for sequence in sequence_list:
                sequence = sequence.strip("\n")
                tmp_p_ls = []
                for frame in tri_img:
                    tmp_p_ls.append(data_dir / "sequences" / sequence / frame)
                total_res_ls.append(tmp_p_ls)

        for res_ls in tqdm(total_res_ls):
            img_list = []
            for img_path in res_ls:
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                img = img.transpose(2, 0, 1)
                img = img.astype(np.float32)
                img /= 255
                img = torch.from_numpy(img)
                img_list.append(img)

            [img1, target, img2] = img_list
            img_tensor = torch.stack((img1, img2, target), 0)

            parent_dir = res_ls[0].parent
            torch.save(img_tensor, parent_dir / "tensor.pt")


if __name__ == "__main__":
    convert_to_pt()
