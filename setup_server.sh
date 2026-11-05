#!/usr/bin/env bash
# Einrichtung auf dem SAMK GPU Server, einmal aus dem Projektordner starten:
#   sudo apt update && sudo apt install -y python3.11-venv tmux
#   Projektordner (Code und data/ fuer die Groesse) vom Mac hochladen, dann hineinwechseln
#   bash setup_server.sh        (optional: bash setup_server.sh 128)
# Danach: source ~/venv/bin/activate
set -e
SIZE=${1:-28}

python3 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip
# gleiche Versionen wie auf dem Mac
pip install torch==2.7.0 torchvision==0.22.0 numpy==1.26.4 pandas==2.2.1 \
    opencv-python-headless==4.9.0.80 scikit-learn==1.4.1 matplotlib==3.8.3 medmnist==3.0.2

python -c "import torch; print('cuda verfuegbar:', torch.cuda.is_available())"

# Daten nur erzeugen, wenn sie nicht schon hochgeladen wurden
if [ ! -f "data/CSVs/$SIZE/test.csv" ]; then
    python download_data.py -size "$SIZE"
    python create_folds.py -img_size "$SIZE"
fi
echo "Fertig. Training zum Beispiel: tmux new -s train, dann python main.py -img_size $SIZE -backbone resnet18_modified"
