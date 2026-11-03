
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import os

def read_img(path_file):
    img = cv2.imread(path_file, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def compute_mean_std(dataset):
    # mean and std for all channels of the given image (scaled to 0-1)
    sum = np.zeros(3)
    sq_sum = np.zeros(3)

    for path in dataset["image_path"]:
        img = read_img(path) / 255.0
        sum += img.sum(axis=(0,1))
        sq_sum += (img ** 2).sum(axis=(0,1))

    n_pixels = len(dataset) * img.shape[0] * img.shape[1]
    mean = sum / n_pixels
    std = np.sqrt(sq_sum / n_pixels - mean ** 2)

    return mean, std

class bloodmnist_dataset(Dataset):
    def __init__(self, dataset, mean, std):
        self.dataset = dataset
        # shape (3, 1, 1), so every channel is normalized with its own value
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return (len(self.dataset))

    def __getitem__(self, index):
        img = read_img(self.dataset["image_path"].iloc[index])
        # (H, W, C) -> (C, H, W), PyTorch expects the channels first
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255
        img = (img - self.mean) / self.std
        name = (self.dataset["image_path"].iloc[index]).split("/")[-1]
        target = self.dataset["label"].iloc[index]

        res = {
            "img": img,
            "name": name,
            "target": target
        }

        return res
        

