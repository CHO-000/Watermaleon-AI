from __future__ import annotations

import argparse
import io
import json
import random
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import CLASSES, IMAGE_SIZE, build_model


class LeafDataset(Dataset):
    def __init__(self, rows: list[tuple[Path, int]], training: bool, strong_augment: bool = False):
        self.rows = rows
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        ] if training else []
        if training and strong_augment:
            aug = [
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.85, 1.15)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.25),
            ]
        self.transform = transforms.Compose([
            *([] if training and strong_augment else [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]),
            *aug,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, label = self.rows[index]
        with Image.open(path) as im:
            image = im.convert("RGB")
        return self.transform(image), label


def prepare_originals(archive: Path, cache_dir: Path) -> list[tuple[Path, int]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.infolist() if "/Original Image/Watermelon/" in m.filename and m.filename.lower().endswith((".jpg", ".jpeg", ".png"))]
        for i, member in enumerate(members, 1):
            parts = member.filename.split("/")
            label = parts[-2]
            if label not in CLASSES:
                continue
            dest = cache_dir / label / (Path(parts[-1]).stem + ".jpg")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                try:
                    with Image.open(io.BytesIO(zf.read(member))) as im:
                        rgb = ImageOps.exif_transpose(im).convert("RGB")
                        rgb.thumbnail((512, 512))
                        rgb.save(dest, "JPEG", quality=92)
                except Exception as exc:
                    print(f"Skipping corrupt image {member.filename}: {exc}", flush=True)
                    continue
            rows.append((dest, CLASSES.index(label)))
            if i % 200 == 0:
                print(f"Prepared {i}/{len(members)} original images", flush=True)
    return rows


def split_rows(rows: list[tuple[Path, int]], seed: int):
    # File numbers follow capture order. Keep adjacent captures in the same split.
    # A random image split gives an optimistic score when the same leaf was photographed repeatedly.
    groups = {i: [] for i in range(len(CLASSES))}
    for row in rows:
        groups[row[1]].append(row)
    train, val, test = [], [], []
    for group in groups.values():
        group.sort(key=lambda row: int("".join(c for c in row[0].stem if c.isdigit())))
        n_test = max(1, round(len(group) * 0.15))
        n_val = max(1, round(len(group) * 0.15))
        train.extend(group[:-(n_test + n_val)])
        val.extend(group[-(n_test + n_val):-n_test])
        test.extend(group[-n_test:])
    return train, val, test


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    truth, prediction = [], []
    loss_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * len(labels)
        truth.extend(labels.cpu().tolist())
        prediction.extend(logits.argmax(1).cpu().tolist())
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for actual, predicted in zip(truth, prediction):
        confusion[actual, predicted] += 1
    per_class = {}
    for i, name in enumerate(CLASSES):
        tp = int(confusion[i, i])
        support = int(confusion[i].sum())
        predicted_count = int(confusion[:, i].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {
        "loss": loss_sum / max(1, len(truth)),
        "accuracy": float(np.trace(confusion) / max(1, len(truth))),
        "macro_f1": float(np.mean([x["f1"] for x in per_class.values()])),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/leaf-cache"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--architecture", choices=["mobilenet_v3_small", "efficientnet_b0"], default="mobilenet_v3_small")
    parser.add_argument("--strong-augment", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    rows = prepare_originals(args.archive, args.cache_dir)
    train, val, test = split_rows(rows, args.seed)
    print("Split:", {k: dict(Counter(label for _, label in x)) for k, x in [("train", train), ("val", val), ("test", test)]}, flush=True)
    loaders = {
        "train": DataLoader(LeafDataset(train, True, args.strong_augment), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda"),
        "val": DataLoader(LeafDataset(val, False), batch_size=args.batch_size, num_workers=0, pin_memory=device.type == "cuda"),
        "test": DataLoader(LeafDataset(test, False), batch_size=args.batch_size, num_workers=0, pin_memory=device.type == "cuda"),
    }
    model = build_model(pretrained=True, architecture=args.architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    best_f1 = -1.0
    best_epoch = 0
    checkpoint = Path(__file__).parent / "model.pt"
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
        metrics = evaluate(model, loaders["val"], device)
        print(f"Epoch {epoch}: train_loss={loss_sum/len(train):.4f} val_accuracy={metrics['accuracy']:.4f} val_macro_f1={metrics['macro_f1']:.4f}", flush=True)
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = metrics["macro_f1"], epoch
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "image_size": IMAGE_SIZE, "epoch": epoch, "architecture": args.architecture}, checkpoint)
        if epoch - best_epoch >= 4:
            print("Early stopping", flush=True)
            break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    val_metrics = evaluate(model, loaders["val"], device)
    test_metrics = evaluate(model, loaders["test"], device)
    report = {
        "dataset": args.archive.name, "source": "Original Image only", "seed": args.seed,
        "split_method": "class-stratified chronological image-number split",
        "architecture": args.architecture, "strong_augment": args.strong_augment,
        "split_counts": {"train": len(train), "validation": len(val), "test": len(test)},
        "classes": CLASSES, "best_epoch": best_epoch, "training_seconds": round(time.time()-start, 1),
        "validation": val_metrics, "test": test_metrics,
    }
    (Path(__file__).parent / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("TEST", json.dumps(test_metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
