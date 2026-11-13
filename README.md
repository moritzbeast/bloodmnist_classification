# BloodMNIST: Blood Cell Classification

Classification of 8 blood cell types in microscope images (BloodMNIST) with a ResNet-18.

## Data

BloodMNIST from [MedMNIST v2](https://medmnist.com/), downloaded with the `medmnist` package. 17,092 RGB images, 8 classes:
basophil, eosinophil, erythroblast, immature granulocyte, lymphocyte, monocyte, neutrophil, platelet.
The classes are imbalanced: from 7.1 % (basophil, lymphocyte) to 19.5 % (neutrophil).

The official MedMNIST splits are **not** used. All images are merged and split again:

- stratified 80/20 split into train (13,673 images) and test (3,419 images)
- 5-fold stratified cross-validation on the train part
- seed 42 for the splits (training uses seed 13, see Method)

The data is not part of the repository; `download_data.py` and `create_folds.py` create it.

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.12 on a Mac (MPS). On the GPU server, torch 2.8.0 and torchvision 0.23.0 were used instead, see `setup_server.sh`.

## Usage

```bash
# 1. download and save images as png (data/Labeled/28/<label>/)
python download_data.py -size 28

# 2. create test split and 5 folds (data/CSVs/28/)
python create_folds.py -img_size 28

# 3. optional: class distribution and example images (data/overview/)
python data_overview.py -img_size 28

# 4. train one model per fold (final configuration)
python main.py -img_size 28 -backbone resnet18_modified -epochs 50

# 5. evaluate the 5 models on the test set
python evaluate.py -img_size 28 -backbone resnet18_modified
```

`evaluate.py` needs the same `-img_size` and `-backbone` as the training. With a wrong `-img_size` it still runs, but the results are wrong.

Outputs are written to `sessions/` and are overwritten by the next run:

- training: `best_model_fold_<k>.pth`, `metrics_fold_<k>.png` (loss and metrics per epoch), `val_metrics.csv`
- evaluation: `test_metrics.csv`, `confusion_matrix.png`, `misclassified_examples.png`

## Method

| | Final configuration |
|---|---|
| Model | ResNet-18, ImageNet weights. Modified stem for 28 px: `conv1` 3×3 with stride 1, no max pooling |
| Input | 28×28 RGB, normalized per channel with mean/std of the training fold |
| Augmentation (train only) | random horizontal and vertical flip, ColorJitter (brightness, contrast, saturation 0.1) |
| Loss | cross-entropy |
| Optimizer | AdamW, lr 2e-4, weight decay 0.01, batch size 64 |
| Scheduler | cosine annealing over 50 epochs |
| Seed | 13 for training (set at the start of each fold), 42 for the splits |
| Model selection | best epoch per fold by validation balanced accuracy |
| Test | each of the 5 fold models, and an ensemble (mean of the softmax probabilities) |

Main metric is **balanced accuracy** because of the class imbalance. Accuracy, ROC-AUC and average precision are reported as well.
Decisions were made on the validation results only; the test set was only used for reporting.

## Results

Balanced accuracy, 28 px. Validation and single-model test values are mean ± std over the 5 folds.

| # | Run | Val | Test (single) | Test (ensemble) |
|---|---|---|---|---|
| 0 | Baseline: `resnet18`, from scratch, Adam, 15 epochs, no augmentation | 0.882 ± 0.006 | 0.888 ± 0.006 | 0.922 |
| 1 | + modified stem (`resnet18_modified`) | 0.916 ± 0.009 | 0.918 ± 0.010 | 0.941 |
| 2 | + ImageNet weights | 0.949 ± 0.005 | 0.948 ± 0.004 | 0.965 |
| 3a | + flips | 0.960 ± 0.004 | 0.960 ± 0.001 | 0.973 |
| 3b | + ColorJitter | 0.956 ± 0.006 | 0.959 ± 0.002 | 0.973 |
| 4 | + AdamW, cosine schedule, 50 epochs (**final**) | **0.972 ± 0.004** | **0.973 ± 0.002** | **0.978** |

Baseline settings: Adam, lr 2e-4, weight decay 1e-4, batch size 64. The current code contains the final configuration; the earlier runs were trained with earlier versions of the code.

**Findings** (change in validation balanced accuracy)

- Modified stem: +3.4 pp. A 28×28 image ends at 4×4 instead of 1×1 after the last block.
- ImageNet weights: +3.3 pp.
- Flips: +1.1 pp. ColorJitter did not help, but was kept in the final run.
- AdamW, cosine schedule, 50 epochs: +1.6 pp.
- The ensemble of the 5 fold models is better than the single models in every run.
- The weakest class of the final model is immature granulocyte (recall 0.94), mostly confused with neutrophils and monocytes.

For comparison, MedMNIST+ ([Doerrich et al.](https://arxiv.org/abs/2404.15786)) reports an accuracy of 91.9 % (28 px) and 98.3 % (128 px) for ResNet-18 on BloodMNIST. The final model here reaches 97.2 % (single, mean) and 97.8 % (ensemble) accuracy at 28 px. This is not directly comparable: different splits and a different training setup.

## Files

| File | Content |
|---|---|
| `download_data.py` | downloads BloodMNIST, merges the official splits, saves the images as png |
| `create_folds.py` | stratified train/test split and 5 folds, saved as CSVs |
| `data_overview.py` | class distribution of train and test, 5 example images per class |
| `args.py` | command line arguments |
| `dataset.py` | PyTorch dataset, mean/std of the training fold |
| `model.py` | ResNet variants with 8 outputs, modified stem |
| `trainer.py` | training and validation loop, metrics, plots |
| `main.py` | trains one model per fold, saves `val_metrics.csv` |
| `evaluate.py` | test metrics per fold and for the ensemble, confusion matrix, misclassified examples |
| `setup_server.sh` | environment setup on the GPU server |
