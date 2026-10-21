import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Parsed model training options")

    parser.add_argument("-backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50"])

    parser.add_argument("-output_dir", type=str, default="sessions")

    parser.add_argument("-csv_dir", type=str, default="data/CSVs")

    parser.add_argument("-epochs", type=int, default=20)

    parser.add_argument("-bs", type=int, default=16,
                        choices=[16, 32, 64, 128])

    parser.add_argument("-lr", type=float, default=1e-4,
                        choices=[1e-4, 1e-3])

    parser.add_argument("-wd", type=float, default=1e-4)

    args = parser.parse_args()

    return args