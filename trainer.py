import os
from args import *
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

args = get_args()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def compute_metrics(labels, probs):
    classes = list(range(probs.shape[1]))
    preds = probs.argmax(axis=1)

    bal_acc = balanced_accuracy_score(labels, preds)
    roc_auc = roc_auc_score(labels, probs, multi_class="ovr", labels=classes)
    avg_prec = average_precision_score(label_binarize(labels, classes=classes), probs, average="macro")

    return bal_acc, roc_auc, avg_prec

def train_model(model, train_loader, val_loader, fold):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    os.makedirs(args.output_dir, exist_ok=True)
    best_bal_acc = -1

    history = {key: [] for key in ["train_loss", "train_bal_acc", "train_roc_auc", "train_avg_prec",
                                   "val_loss", "val_bal_acc", "val_roc_auc", "val_avg_prec"]}

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0

        all_labels = []
        all_probs = []

        for batch in train_loader:
            inputs = batch["img"].to(device)
            labels = batch["target"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            probs = torch.softmax(outputs.detach(), dim=1)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        train_loss = running_loss/len(train_loader)
        train_bal_acc, train_roc_auc, train_avg_prec = compute_metrics(np.concatenate(all_labels), np.concatenate(all_probs))

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Training Loss: {train_loss:.4f} | Balanced Acc: {train_bal_acc:.4f} | "
              f"ROC-AUC: {train_roc_auc:.4f} | Avg Precision: {train_avg_prec:.4f}")
        
        val_loss, val_bal_acc, val_roc_auc, val_avg_prec = validate_model(model, val_loader, criterion)

        for key, value in [("train_loss", train_loss), ("train_bal_acc", train_bal_acc),
                           ("train_roc_auc", train_roc_auc), ("train_avg_prec", train_avg_prec),
                           ("val_loss", val_loss), ("val_bal_acc", val_bal_acc),
                           ("val_roc_auc", val_roc_auc), ("val_avg_prec", val_avg_prec)]:
            history[key].append(value)
        
        if val_bal_acc > best_bal_acc:
            best_bal_acc = val_bal_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"best_model_fold_{fold}.pth"))
            print(f"New best model saved (Balanced Acc: {best_bal_acc:.4f})")

    plot_metrics(history, fold)

def validate_model(model, val_loader, criterion):
    model.eval()

    running_loss = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["img"].to(device)
            labels = batch["target"].to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    validation_loss = running_loss/len(val_loader)
    bal_acc, roc_auc, avg_prec = compute_metrics(all_labels, all_probs)

    print(f"Validation Loss: {validation_loss:.4f} | Balanced Acc: {bal_acc:.4f} | "
          f"ROC-AUC: {roc_auc:.4f} | Avg Precision: {avg_prec:.4f}")

    return validation_loss, bal_acc, roc_auc, avg_prec

def plot_metrics(history, fold):
    metrics = [("loss", "Loss"),
               ("bal_acc", "Balanced Accuracy"),
               ("roc_auc", "ROC-AUC"),
               ("avg_prec", "Average Precision")]

    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = int(np.argmax(history["val_bal_acc"])) + 1

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, (key, title) in zip(axes.flat, metrics):
        ax.plot(epochs, history[f"train_{key}"], color="blue", linewidth=2, label="Training")
        ax.plot(epochs, history[f"val_{key}"], color="orange", linewidth=2, linestyle="--", label="Validation")
        ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1, label=f"Best model (epoch {best_epoch})")
        ax.set_title(f"{title} over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(f"Fold {fold}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, f"metrics_fold_{fold}.png"), dpi=150)
    plt.close(fig)





    

