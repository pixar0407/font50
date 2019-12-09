import numpy as np
# import cv2
import torch
import os
import glob
from torchvision import transforms


class FontDataset():
    def __init__(self, npy_dir, max_dataset_size=float("inf")):
        self.dir_path = npy_dir
        # self.to_tensor = transforms.ToTensor()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (-0.5,))
        ])

        entry = []
        files = glob.glob1(npy_dir, '*npy')
        for f in files:
            f = os.path.join(npy_dir, f)
            entry.append(f)

        self.npy_entry = entry[:min(max_dataset_size, len(entry))]

    # def __getitem__(self, index):
    #     npy_entry = self.npy_entry
    #     single_npy_path = npy_entry[index]
    #
    #     single_npy = np.load(single_npy_path, allow_pickle=True)[0]
    #     single_npy_tensor = self.to_tensor(single_npy)
    #
    #     single_npy_label = np.load(single_npy_path, allow_pickle=True)[1]
    #
    #     return (single_npy_tensor, single_npy_label)
    def __getitem__(self, index):
        npy_entry = self.npy_entry
        single_npy_path = npy_entry[index]
        # print(single_npy_path)

        single_npy = np.load(single_npy_path, allow_pickle=True)[0][:, :, 0]
        single_npy = 1.- single_npy
        single_npy = single_npy.astype(np.float32)

        # single_npy = cv2.Laplacian(single_npy, cv2.CV_32F, ksize = 3)
        single_npy_tensor = self.to_tensor(single_npy)


        single_npy_label = np.load(single_npy_path, allow_pickle=True)[1]
        # print(single_npy_label)

        return (single_npy_tensor, single_npy_label)

    def __len__(self):
        return len(self.npy_entry)


# if __name__ == '__main__':
#     train_dir = '../data/npy_train'
#     val_dir = '../data/npy_val'

    # ================================================================== #
    #                        1. Load Data
    # ================================================================== #
    # train_dataset = FontDataset(train_dir)
    # val_dataset = FontDataset(val_dir)

    # ================================================================== #
    #                        2. Define Dataloader
    # ================================================================== #
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                            batch_size=1, shuffle=True)
    #
    # val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
    #                                          batch_size=1)

    # 이하 양곤 작성