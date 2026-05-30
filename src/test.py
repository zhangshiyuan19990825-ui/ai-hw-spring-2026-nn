import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import get_model


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="Test trained MNIST models.")
    parser.add_argument("--models", nargs="+", default=["mlp", "cnn", "transformer"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--out", type=str, default="results/test_results.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    rows = []
    for model_name in args.models:
        ckpt_path = Path(args.model_dir) / f"{model_name}.pt"
        if not ckpt_path.exists():
            print(f"Skipping {model_name}: missing {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device)
        model = get_model(model_name).to(device)
        model.load_state_dict(ckpt["state_dict"])
        acc = evaluate(model, test_loader, device)
        rows.append({"model": model_name, "test_accuracy_percent": round(acc * 100, 2)})
        print(f"{model_name}: {acc*100:.2f}%")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "test_accuracy_percent"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
