import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


def load_image_data(dataset_path="data/Labeled", labels=("0", "1", "2", "3", "4")):
    # save data in pandas dataframe
    image_data = []

    for label in labels:
        label_path = os.path.join(dataset_path, label)

        for image in os.listdir(label_path):
            image_path = os.path.join(label_path, image)
            image_data.append((image_path, label))

    return pd.DataFrame(image_data, columns=["image_path", "label"])


def create_folds(csv_dir="data/CSVs", dataset_path="data/Labeled", n_splits=5, seed=42):
    os.makedirs(csv_dir, exist_ok=True)

    df = load_image_data(dataset_path)

    # split in train and test data
    train_df, test_df = train_test_split(df, train_size=0.8, stratify=df["label"], random_state=seed)
    test_df.to_csv(os.path.join(csv_dir, "test.csv"), index=False)

    # split train with kfold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df["image_path"], train_df["label"])):
        fold_train_df = train_df.iloc[train_idx]
        fold_val_df = train_df.iloc[val_idx]

        fold_train_df.to_csv(os.path.join(csv_dir, f"train_fold_{fold}.csv"), index=False)
        fold_val_df.to_csv(os.path.join(csv_dir, f"val_fold_{fold}.csv"), index=False)


if __name__ == "__main__":
    from args import get_args

    args = get_args()
    create_folds(csv_dir=args.csv_dir)
