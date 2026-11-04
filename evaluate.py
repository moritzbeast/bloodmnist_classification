from args import * 
import pandas as pd
import os
from dataset import compute_mean_std, bloodmnist_dataset
from torch.utils.data import DataLoader
from model import MyModel
import torch
import numpy as np
from trainer import compute_metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dataset import read_img

# BloodMNIST label order
CLASS_NAMES = ["basophil", "eosinophil", "erythroblast", "immature gran.",
               "lymphocyte", "monocyte", "neutrophil", "platelet"]

# load test data (test.csv) and args
args = get_args()
# CSVs of the chosen image size
csv_dir = os.path.join(args.csv_dir, str(args.img_size))
test_set = pd.read_csv(os.path.join(csv_dir, "test.csv"))

print(f"Evaluating run: {args.output_dir}")

# use mac gpu
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

all_fold_probs = []
fold_metrics = []

# for each fold:
for fold in range(5):
    # compute mean/std from train_fold_k (same normalization as in training)
    train_set = pd.read_csv(os.path.join(csv_dir, f"train_fold_{fold}.csv"))
    mean, std = compute_mean_std(train_set)
    # create test dataset + dataloader with these stats
    test_dataset = bloodmnist_dataset(test_set, mean, std)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    # load model, load best_model_fold_k.pth, model.eval()
    model = MyModel(backbone=args.backbone)
    weight_path = os.path.join(args.output_dir, f"best_model_fold_{fold}.pth")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    # predict test data, softmax probabilities
    probs = []
    labels = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["img"].to(device)
            targets = batch["target"]

            outputs = model(inputs)
            probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
            labels.append(targets.numpy())

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    # compute metrics for this folds
    bal_acc, roc_auc, avg_prec = compute_metrics(labels, probs)
    print(f"Fold {fold} | Balanced Acc: {bal_acc:.4f} | ROC-AUC: {roc_auc:.4f} | Avg Precision: {avg_prec:.4f}")

    all_fold_probs.append(probs)
    fold_metrics.append({"fold": fold, "bal_acc": bal_acc, "roc_auc": roc_auc, "avg_prec": avg_prec})

# ensemble: average the probabilities of all 5 models
ensemble_probs = np.mean(all_fold_probs, axis=0)   # (5, 3419, 8) -> (3419, 8)
ensemble_preds = ensemble_probs.argmax(axis=1)     # one label for each image

# compute metrics for the ensemble 
bal_acc, roc_auc, avg_prec = compute_metrics(labels, ensemble_probs)
print(f"Ensemble | Balanced Acc: {bal_acc:.4f} | ROC-AUC: {roc_auc:.4f} | Avg Precision: {avg_prec:.4f}")
ensemble_metrics = {"fold": "ensemble", "bal_acc": bal_acc, "roc_auc": roc_auc, "avg_prec": avg_prec}

# save metrics (per fold + ensemble) to the run folder
results = pd.DataFrame(fold_metrics)
metric_cols = ["bal_acc", "roc_auc", "avg_prec"]

summary = pd.DataFrame([
    {"fold": "mean", **results[metric_cols].mean().to_dict()},
    {"fold": "std", **results[metric_cols].std().to_dict()},
    ensemble_metrics,
])
results = pd.concat([results, summary], ignore_index=True)

results.to_csv(os.path.join(args.output_dir, "test_metrics.csv"), index=False)
print(results.round(4))

# plot confusion matrix 
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, normalize, title in zip(axes, [None, "true"], ["Counts", "Normalized per true class"]):
    cm = confusion_matrix(labels, ensemble_preds, normalize=normalize)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False,
              values_format="d" if normalize is None else ".2f")
    ax.set_title(title)

fig.tight_layout()
fig.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=150)
plt.close(fig)

# plot examples
wrong_idx = np.where(ensemble_preds != labels)[0]
print(f"Misclassified: {len(wrong_idx)} of {len(labels)}")

rng = np.random.default_rng(42)
chosen = rng.choice(wrong_idx, size=min(5, len(wrong_idx)), replace=False)

fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for ax in axes.flat:
    ax.axis("off")

for ax, idx in zip(axes.flat, chosen):
    img = read_img(test_set["image_path"].iloc[idx])
    pred = ensemble_preds[idx]
    ax.imshow(img)
    ax.set_title(f"true: {CLASS_NAMES[labels[idx]]}\n"
                 f"pred: {CLASS_NAMES[pred]} ({ensemble_probs[idx, pred]:.2f})", fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(args.output_dir, "misclassified_examples.png"), dpi=150)
plt.close(fig)