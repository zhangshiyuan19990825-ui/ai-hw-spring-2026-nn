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
    parser = argparse.ArgumentParser(description="Test MNIST model under simple image augmentations.")
    parser.add_argument("--model", default="cnn", choices=["mlp", "cnn", "transformer"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", default="results/augmentation_results.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(Path(args.model_dir) / f"{args.model}.pt", map_location=device)
    model = get_model(args.model).to(device)
    model.load_state_dict(ckpt["state_dict"])

    tests = {
        "clean": transforms.ToTensor(),
        "rotate_10deg": transforms.Compose([transforms.RandomRotation((10, 10)), transforms.ToTensor()]),
        "rotate_minus10deg": transforms.Compose([transforms.RandomRotation((-10, -10)), transforms.ToTensor()]),
        "translate_2px": transforms.Compose([transforms.RandomAffine(degrees=0, translate=(2/28, 2/28)), transforms.ToTensor()]),
    }
    rows = []
    for name, transform in tests.items():
        ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        acc = evaluate(model, loader, device)
        rows.append({"test_condition": name, "accuracy_percent": round(acc * 100, 2)})
        print(f"{name}: {acc*100:.2f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["test_condition", "accuracy_percent"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved augmentation results to {args.out}")


if __name__ == "__main__":
    main()
