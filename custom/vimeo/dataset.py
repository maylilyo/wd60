# PIP
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class Vimeo(Dataset):
    def __init__(
        self,
        data_dir,
        state,  # train, test
        is_pt=False,
    ):
        self.data_dir = data_dir / "vimeo_triplet"
        self.is_pt = is_pt
        if is_pt:
            self.path_list = self.get_pt_list(state)
        else:
            self.path_list = self.get_png_list(state)

    def get_pt_list(self, state):
        total_res_ls = []
        with open(self.data_dir / f"tri_{state}list.txt", "r") as info_file:
            sequence_list = info_file.readlines()
            if "" in sequence_list:
                sequence_list.remove("")
            for sequence in sequence_list:
                sequence = sequence.strip("\n")
                total_res_ls.append(
                    str(self.data_dir / "sequences" / sequence / "tensor.pt")
                )
        return total_res_ls

    def get_png_list(self, state):
        tri_img = ["im1.png", "im2.png", "im3.png"]
        total_res_ls = []
        with open(self.data_dir / f"tri_{state}list.txt", "r") as info_file:
            sequence_list = info_file.readlines()
            if "" in sequence_list:
                sequence_list.remove("")
            for sequence in sequence_list:
                sequence = sequence.strip("\n")
                tmp_p_ls = []
                for frame in tri_img:
                    tmp_p_ls.append(str(self.data_dir / "sequences" / sequence / frame))
                total_res_ls.append(tmp_p_ls)
        return total_res_ls

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if self.is_pt:
            img_tensor = torch.load(self.path_list[idx])
            return [img_tensor[0], img_tensor[1], img_tensor[2]]

        img_list = []
        for img_path in self.path_list[idx]:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = img.transpose(2, 0, 1)
            img = img.astype(np.float32)
            img /= 255
            img = torch.from_numpy(img)
            img_list.append(img)

        [img1, target, img2] = img_list
        return [img1, img2, target]
