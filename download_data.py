import os
import argparse
import cv2
import numpy as np
from medmnist import BloodMNIST


def download_data(raw_dir="data/raw", dataset_path="data/Labeled", size=28):
    os.makedirs(raw_dir, exist_ok=True)
    # own folder for each image size, e.g. data/Labeled/128
    dataset_path = os.path.join(dataset_path, str(size))

    # merge the official splits to create own splits in create_folds.py
    images, labels = [], []
    for split in ["train", "val", "test"]:
        data = BloodMNIST(split=split, download=True, size=size, root=raw_dir)
        images.append(data.imgs)
        labels.append(data.labels.squeeze(1))

    images = np.concatenate(images)
    labels = np.concatenate(labels)

    # save every image as png in a folder named after its label
    for label in np.unique(labels):
        os.makedirs(os.path.join(dataset_path, str(label)), exist_ok=True)

    for idx, (img, label) in enumerate(zip(images, labels)):
        image_path = os.path.join(dataset_path, str(label), f"{idx:05d}.png")
        cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"Saved {len(images)} images to {dataset_path}")
    print("Images per class:", dict(zip(*np.unique(labels, return_counts=True))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BloodMNIST and save it as png images")
    parser.add_argument("-img_size", type=int, default=28, choices=[28, 64, 128, 224])
    args = parser.parse_args()

    download_data(size=args.size)
