import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from args import get_args
from dataset import read_img

# BloodMNIST label order
CLASS_NAMES = ["basophil", "eosinophil", "erythroblast", "immature gran.",
               "lymphocyte", "monocyte", "neutrophil", "platelet"]


def plot_distribution(df, title, save_path):
    # number of images per class (all 8 classes, sorted by label)
    counts = df["label"].value_counts().reindex(range(len(CLASS_NAMES)), fill_value=0)
    percent = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(CLASS_NAMES, counts.values, color="steelblue")

    # count and percentage above every bar
    for bar, n, p in zip(bars, counts.values, percent.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{n}\n({p:.1f}%)", ha="center", va="bottom", fontsize=9)

    ax.set_title(f"{title} (n = {counts.sum()})")
    ax.set_ylabel("Number of images")
    ax.set_ylim(0, counts.max() * 1.15)  # space for the text above the bars
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_examples(df, save_path, n_examples=5, seed=42):
    # one row per class, n_examples random images per row
    fig, axes = plt.subplots(len(CLASS_NAMES), n_examples, figsize=(2 * n_examples, 2 * len(CLASS_NAMES)))

    for label, class_name in enumerate(CLASS_NAMES):
        # fixed random_state so the same images are chosen every run
        examples = df[df["label"] == label].sample(n=n_examples, random_state=seed)

        for col, path in enumerate(examples["image_path"]):
            ax = axes[label, col]
            ax.imshow(read_img(path))
            ax.set_xticks([])
            ax.set_yticks([])

        # class name on the left of every row
        axes[label, 0].set_ylabel(class_name, fontsize=11, rotation=0, ha="right", va="center")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    args = get_args()
    csv_dir = os.path.join(args.csv_dir, str(args.img_size))
    output_dir = "data/overview"
    os.makedirs(output_dir, exist_ok=True)

    # all training data = train + val of one fold (every fold splits the same 80%)
    train_df = pd.concat([
        pd.read_csv(os.path.join(csv_dir, "train_fold_0.csv")),
        pd.read_csv(os.path.join(csv_dir, "val_fold_0.csv")),
    ], ignore_index=True)
    test_df = pd.read_csv(os.path.join(csv_dir, "test.csv"))

    plot_distribution(train_df, "Training data", os.path.join(output_dir, "train_distribution.png"))
    plot_distribution(test_df, "Test data", os.path.join(output_dir, "test_distribution.png"))
    plot_examples(train_df, os.path.join(output_dir, "class_examples.png"))

    print(f"Saved plots to {output_dir}")
