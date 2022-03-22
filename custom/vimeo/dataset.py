# PIP
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as TF


class Vimeo(Dataset):
    def __init__(
        self,
        data_dir,
        state,  # train, test
        is_pt=False,
        is_aug=True,
        is_crop=False,
    ):
        self.is_pt = is_pt
        self.is_aug = is_aug
        self.is_crop = is_crop
        if is_pt:
            data_dir = data_dir / "vimeo_pt" / state
            self.path_list = self.get_pt_list(data_dir, state)
        else:
            data_dir = data_dir / "vimeo_triplet"
            self.path_list = self.get_png_list(data_dir, state)

    def get_pt_list(self, data_dir):
        pt_list = data_dir.glob("**/*.pt")
        pt_list = list(pt_list)
        pt_list.sort()
        return pt_list

    def get_png_list(self, data_dir, state):
        triplet_list = []
        with open(data_dir / f"tri_{state}list.txt", "r") as info_file:
            sequence_list = info_file.readlines()
            if "" in sequence_list:
                sequence_list.remove("")
            for sequence in sequence_list:
                sequence = sequence.strip("\n")
                path_list = []
                for file_name in ["im1.png", "im2.png", "im3.png"]:
                    path_list.append(str(data_dir / "sequences" / sequence / file_name))
                triplet_list.append(path_list)
        return triplet_list

    def __len__(self):
        if self.is_aug:
            return len(self.path_list) * 4
        return len(self.path_list)

    def __getitem__(self, idx):
        if self.is_pt:
            img_tensor = torch.load(self.path_list[idx])
            return [img_tensor[0], img_tensor[1], img_tensor[2]]

        if self.is_aug:
            aug_idx = idx % 4
            idx = idx // 4

        img_list = []
        crop_params = None
        for img_path in self.path_list[idx]:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1)
            img = img.astype(np.float32)
            img /= 255.0
            img = torch.from_numpy(img)
            if self.is_crop:
                if crop_params is None:
                    crop_params = RandomCrop.get_params(img, output_size=(256, 256))
                img = TF.crop(img, *crop_params)
            if self.is_aug:
                if aug_idx == 1:
                    img = TF.hflip(img)
                elif aug_idx == 2:
                    img = TF.vflip(img)
                elif aug_idx == 3:
                    img = TF.rotate(img, 90)
            img_list.append(img)

        [img1, target, img2] = img_list
        return [img1, img2, target]
