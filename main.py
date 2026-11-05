from args import *
import pandas as pd
import os
import random
import numpy as np
import torch
from dataset import bloodmnist_dataset, compute_mean_std
from torch.utils.data import DataLoader
from model import MyModel
from trainer import train_model

# 1. We need the arguments
args = get_args()
# CSVs of the chosen image size, e.g. data/CSVs/128
csv_dir = os.path.join(args.csv_dir, str(args.img_size))

fold_metrics = []

# 2. Iterate over each fold
for fold in range(5):
    print(f"Fold {fold}")

    # same random start in every run (initial weights, shuffling)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_set = pd.read_csv(os.path.join(csv_dir, f"train_fold_{fold}.csv"))
    val_set = pd.read_csv(os.path.join(csv_dir, f"val_fold_{fold}.csv"))

    # 3. Preparing Dataset
    mean, std = compute_mean_std(train_set)
    print(f"Mean: {mean.round(4)}, Std: {std.round(4)}")

    train_dataset = bloodmnist_dataset(train_set, mean, std)
    val_dataset = bloodmnist_dataset(val_set, mean, std)

    # 4. Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs)

    # 5. Initializing the model
    model = MyModel(backbone=args.backbone)

    # 6. Train the model
    fold_metrics.append(train_model(model, train_loader, val_loader, fold))

# 7. Save validation metrics of the best epoch per fold (+ mean and std)
results = pd.DataFrame(fold_metrics)
metric_cols = ["bal_acc", "acc", "roc_auc", "avg_prec"]

summary = pd.DataFrame([
    {"fold": "mean", **results[metric_cols].mean().to_dict()},
    {"fold": "std", **results[metric_cols].std().to_dict()},
])
results = pd.concat([results, summary], ignore_index=True)

results.to_csv(os.path.join(args.output_dir, "val_metrics.csv"), index=False)
print(results.round(4))