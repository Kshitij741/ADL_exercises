"""ADL Sheet 1: CIFAR-10 training for ResNet18, ViT, and (when completed) CrossViT.

Exercise 1.1: training augmentation and best-validation checkpoint.
Exercise 1.2: train the ViT implemented in my_models.py.

Example: python ex1_main.py --model vit --epochs 100 --batch-size 64
"""

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from my_models import ViT, CrossViT


BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train on CIFAR-10 (ADL Exercise 1)")
    parser.add_argument("--model", choices=["r18", "vit", "cvit"], default="vit")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Train one batch, then stop")
    parser.add_argument("--resume", action="store_true", help="Resume from the last epoch")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data")
    parser.add_argument("--checkpoint-dir", type=Path, default=BASE_DIR / "checkpoints")
    return parser.parse_args()


def get_loaders(args, device):
    # Augmentation is applied only to the training split, not validation/test.
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.CIFAR10(
        root=str(args.data_dir), train=True, download=True, transform=train_transform
    )
    validation_dataset = datasets.CIFAR10(
        root=str(args.data_dir), train=True, download=False, transform=eval_transform
    )
    test_dataset = datasets.CIFAR10(
        root=str(args.data_dir), train=False, download=True, transform=eval_transform
    )

    # Use the same deterministic indices for both train and validation datasets.
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    train_count = int(len(indices) * 0.9)
    train_subset = Subset(train_dataset, indices[:train_count])
    validation_subset = Subset(validation_dataset, indices[train_count:])

    settings = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    trainloader = DataLoader(train_subset, shuffle=True, **settings)
    valloader = DataLoader(validation_subset, shuffle=False, **settings)
    testloader = DataLoader(test_dataset, shuffle=False, **settings)
    print(f"Dataset: {len(train_subset)} train, {len(validation_subset)} validation, "
          f"{len(test_dataset)} test images")
    return trainloader, valloader, testloader


def create_model(model_name):
    if model_name == "r18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif model_name == "vit":
        # Keep the original architecture and hyperparameters from the starter.
        model = ViT(
            image_size=32, patch_size=8, num_classes=10, dim=64,
            depth=2, heads=8, mlp_dim=128, dropout=0.1, emb_dropout=0.1,
        )
    elif model_name == "cvit":
        # This option becomes usable when Exercise 1.3 is implemented.
        model = CrossViT(
            image_size=32, num_classes=10, sm_dim=64, lg_dim=128,
            sm_patch_size=8, sm_enc_depth=2, sm_enc_heads=8,
            sm_enc_mlp_dim=128, sm_enc_dim_head=64,
            lg_patch_size=16, lg_enc_depth=2, lg_enc_heads=8,
            lg_enc_mlp_dim=128, lg_enc_dim_head=64,
            cross_attn_depth=2, cross_attn_heads=8,
            cross_attn_dim_head=64, depth=3, dropout=0.1, emb_dropout=0.1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, log_interval, dry_run):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(loader, start=1):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        batch_size = target.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == target).sum().item()
        total_samples += batch_size

        if batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == len(loader):
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | "
                  f"Loss {total_loss / total_samples:.4f} | "
                  f"Accuracy {100.0 * total_correct / total_samples:.2f}%",
                  flush=True)

        if dry_run:
            break

    return total_loss / total_samples, 100.0 * total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, name):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(data)
        batch_size = target.size(0)
        total_loss += criterion(logits, target).item() * batch_size
        total_correct += (logits.argmax(dim=1) == target).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    print(f"{name}: Loss {average_loss:.4f} | Accuracy {accuracy:.2f}%")
    return average_loss, accuracy


def run(args):
    if args.epochs < 1 or args.batch_size < 1 or args.log_interval < 1:
        raise ValueError("epochs, batch-size and log-interval must be positive")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA is not in use. A 100-epoch run will be much slower on CPU.")

    print(f"Model: {args.model}, Epochs: {args.epochs}, Batch size: {args.batch_size}")
    trainloader, valloader, testloader = get_loaders(args, device)
    model = create_model(args.model).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.checkpoint_dir / f"{args.model}_best.pt"
    last_path = args.checkpoint_dir / f"{args.model}_last.pt"
    csv_path = args.checkpoint_dir / f"{args.model}_history.csv"
    best_accuracy = -1.0
    best_epoch = 0
    start_epoch = 1

    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"No checkpoint found to resume: {last_path}")
        checkpoint = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_accuracy = checkpoint["best_validation_accuracy"]
        best_epoch = checkpoint["best_epoch"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch} of {args.epochs}")
    else:
        if not args.dry_run:
            # A fresh run resets its history. Existing checkpoint names are reused.
            with csv_path.open("w", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow([
                    "epoch", "train_loss", "train_accuracy", "val_loss",
                    "val_accuracy", "seconds"
                ])

    if args.dry_run:
        train_one_epoch(model, trainloader, optimizer, criterion, device,
                        1, args.log_interval, dry_run=True)
        print("Dry run passed. No checkpoints were saved.")
        return

    if start_epoch > args.epochs:
        print("Requested epoch count already reached. Evaluating saved best model.")

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(
            model, trainloader, optimizer, criterion, device,
            epoch, args.log_interval, dry_run=False
        )
        val_loss, val_accuracy = evaluate(model, valloader, criterion,
                                          device, name="Validation")
        elapsed = time.perf_counter() - start_time

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "validation_accuracy": val_accuracy,
            }, best_path)
            print(f"New best model saved: {best_path}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_validation_accuracy": best_accuracy,
            "best_epoch": best_epoch,
        }, last_path)

        with csv_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                epoch, train_loss, train_accuracy,
                val_loss, val_accuracy, round(elapsed, 1)
            ])
        print(f"Epoch {epoch}/{args.epochs} completed in {elapsed:.1f}s. "
              f"Best validation accuracy: {best_accuracy:.2f}% (epoch {best_epoch})\n",
              flush=True)

    if not best_path.is_file():
        raise FileNotFoundError(f"Best checkpoint is missing: {best_path}")
    best = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    print(f"Testing the best validation model from epoch {best['epoch']}")
    evaluate(model, testloader, criterion, device, name="Test")
    print(f"Best model: {best_path}")
    print(f"Last-epoch checkpoint: {last_path}")
    print(f"Training history: {csv_path}")


if __name__ == "__main__":
    run(parse_args())
