#!/usr/bin/env bash
# Setup on the SAMK GPU server. Run once from the project folder:
#   sudo apt update && sudo apt install -y python3-venv tmux
#   upload the project folder (code and data/ for the image size) from the Mac, then cd into it
#   bash setup_server.sh        (optional: bash setup_server.sh 128)
# Afterwards: source ~/venv/bin/activate
set -e
SIZE=${1:-28}

python3 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip
# torch 2.8 instead of 2.7: the RTX PRO 4500 (Blackwell) needs a CUDA 12.8 build,
# which is the default package on PyPI from torch 2.8 on. Other versions as on the Mac.
pip install torch==2.8.0 torchvision==0.23.0 numpy==1.26.4 pandas==2.2.1 \
    opencv-python-headless==4.9.0.80 scikit-learn==1.4.1.post1 matplotlib==3.8.3 medmnist==3.0.2

# real GPU test, not only is_available (catches missing kernels for this card)
python -c "
import torch
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.randn(256, 256, device='cuda')
print('matmul ok:', (x @ x).sum().item() == (x @ x).sum().item())
"

# only create the data if it was not uploaded already
if [ ! -f "data/CSVs/$SIZE/test.csv" ]; then
    python download_data.py -size "$SIZE"
    python create_folds.py -img_size "$SIZE"
fi
echo "Done. Example training: tmux new -s train, then python main.py -img_size $SIZE -backbone resnet18_modified"
