
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch

def read_img(path_file):
    xray = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE)

    return xray

def compute_mean_std(dataset):
    # mean and std over all pixels of the given images (scaled to 0-1)
    imgs = np.stack([read_img(path) for path in dataset["image_path"]]) / 255.0

    return imgs.mean(), imgs.std()

class knee_xray_dataset(Dataset):
    def __init__(self, dataset, mean, std):
        self.dataset = dataset
        self.mean = mean
        self.std = std

    def __len__(self):
        return (len(self.dataset))

    def __getitem__(self, index):
        img = read_img(self.dataset["image_path"].iloc[index])
        img = torch.from_numpy(img).float() / 255.0
        img = (img - self.mean) / self.std
        img = img.unsqueeze(0).repeat(3, 1, 1)
        name = (self.dataset["image_path"].iloc[index]).split("/")[-1]
        target = self.dataset["label"].iloc[index]

        res = {
            "xray": img,
            "name": name,
            "target": target
        }

        return res
        

