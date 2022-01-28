# PIP
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset


class Vimeo(Dataset):
    def __init__(
        self,
        data_dir,
        state,  # train, test
    ):
        self.data_dir = data_dir / 'vimeo_triplet'
        self.width = 448
        self.height = 256
        self.path_list = self.get_path_list(state)

    def get_path_list(self, state):
        tri_img = ['im1.png', 'im2.png', 'im3.png']
        total_res_ls = []
        with open(self.data_dir / f'tri_{state}list.txt', 'r') as info_file:
            sequence_list = info_file.readlines()
            if '' in sequence_list:
                sequence_list.remove('')
            for sequence in sequence_list:
                sequence = sequence.strip('\n')
                tmp_p_ls = []
                for frame in tri_img:
                    tmp_p_ls.append(self.data_dir / 'sequences' / sequence / frame)
                total_res_ls.append(tmp_p_ls)
        return total_res_ls

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        img_list = []
        for img_path in self.path_list[idx]:
            img = imageio.imread(img_path)
            H, W = img.shape[:2]
            img = np.pad(
                array=img,
                pad_width=((0, (64 - H % 64) if H % 64 else 0), (0, (64 - W % 64) if W % 64 else 0), (0, 0)),
                mode='constant',
            )
            img = img.transpose(2, 0, 1)
            img = img.astype(np.float32)
            img /= 255
            img = torch.from_numpy(img)
            img_list.append(img)

        [img1, target, img2] = img_list
        return [img1, img2, target]
