from args import *
import pandas as pd
import os
from datetime import datetime
from dataset import bloodmnist_dataset, compute_mean_std
from torch.utils.data import DataLoader
from model import MyModel
from trainer import train_model

# 1. We need the arguments
args = get_args()

# own folder for each run (timestamp), so old runs are not overwritten
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = os.path.join(args.output_dir, timestamp)
print(f"Saving run to: {run_dir}")

# 2. Iterate over each fold
for fold in range(5):
    print(f"Fold {fold}")

    train_set = pd.read_csv(os.path.join(args.csv_dir, f"train_fold_{fold}.csv"))
    val_set = pd.read_csv(os.path.join(args.csv_dir, f"val_fold_{fold}.csv"))

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
    train_model(model, train_loader, val_loader, fold, run_dir)