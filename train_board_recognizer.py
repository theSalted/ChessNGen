"""
Train a CNN to classify each of the 64 squares on a 256x256 chess board image.

Architecture: 5× stride-2 conv layers (256→8 spatial) + 1×1 head → (B, 13, 8, 8)
Loss: cross-entropy per square
Metrics: per-square accuracy, full-board accuracy (all 64 squares correct)

Usage:
  python generate_board_dataset.py          # create dataset first
  python train_board_recognizer.py --data datasets/board_recognition
"""

import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pyarrow.parquet as pq
from aim import Run as AimRun

NUM_CLASSES = 13  # empty + 6 white + 6 black

# Inverse mapping: class index → FEN char
CLASS_TO_CHAR = ".PNBRQKpnbrqk"


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
class BoardRecognitionDataset(Dataset):
    def __init__(self, parquet_path: str, images_root: str):
        table = pq.read_table(parquet_path)
        self.filenames = table.column("filename").to_pylist()
        self.labels = table.column("labels").to_pylist()
        self.images_root = Path(images_root)
        self.tf = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        img = Image.open(self.images_root / self.filenames[i]).convert("RGB")
        x = self.tf(img)
        # labels: flat list of 64 ints → (8, 8) tensor, rank-major (rank 0 = bottom)
        # flip so row 0 = rank 8 (top of image), matching CNN spatial output
        y = torch.tensor(self.labels[i], dtype=torch.long).view(8, 8).flip(0)
        return x, y


# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------
class BoardRecognizer(nn.Module):
    """Small CNN: 256×256 → 8×8×13 per-square classification."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 256×256
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # → 128
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # → 64
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # → 32
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # → 16
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # → 8
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 1×1 head → 13 classes per spatial location
            nn.Conv2d(256, NUM_CLASSES, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 256, 256) → (B, 13, 8, 8)"""
        return self.net(x)


# -------------------------------------------------------------------
# FEN conversion
# -------------------------------------------------------------------
def grid_to_fen(grid) -> str:
    """Convert an 8×8 class-index grid to FEN piece-placement string.

    grid: (8, 8) tensor or nested list, row 0 = rank 8 (top of board/image).
    """
    rows = []
    for row in range(8):  # row 0 = rank 8, already FEN order (top to bottom)
        row_str = ""
        empty = 0
        for file in range(8):
            cls = int(grid[row][file])
            if cls == 0:
                empty += 1
            else:
                if empty > 0:
                    row_str += str(empty)
                    empty = 0
                row_str += CLASS_TO_CHAR[cls]
        if empty > 0:
            row_str += str(empty)
        rows.append(row_str)
    return "/".join(rows)


# -------------------------------------------------------------------
# LR schedule (same pattern as train_tokenizer.py)
# -------------------------------------------------------------------
def lr_at_step(step: int, total_steps: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total_steps - warmup)
    min_lr = base_lr * 0.1
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
def train(args):
    device = torch.device(args.device)
    use_amp = (device.type == "cuda") and not args.no_amp
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data)
    images_root = data_root / "images"

    train_ds = BoardRecognitionDataset(str(data_root / "train.parquet"), str(images_root))
    val_ds = BoardRecognitionDataset(str(data_root / "val.parquet"), str(images_root))
    print(f"Train: {len(train_ds)} positions | Val: {len(val_ds)} positions")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(args.batch_size, 64), shuffle=False,
        num_workers=2, pin_memory=True,
    )

    model = BoardRecognizer().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.2f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Aim experiment tracking — separate experiment name from dynamics/tokenizer
    aim_run = AimRun(repo="runs", experiment="board_recognizer")
    aim_run["hparams"] = {
        "model": "BoardRecognizer",
        "lr": args.lr,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "weight_decay": args.wd,
        "n_params": n_params,
    }

    step = 0
    best_board_acc = 0.0

    while step < args.steps:
        for x, y in train_loader:
            if step >= args.steps:
                break

            model.train()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)  # (B, 8, 8)

            lr = lr_at_step(step, args.steps, args.warmup, args.lr)
            for pg in opt.param_groups:
                pg["lr"] = lr

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)  # (B, 13, 8, 8)
                # reshape for cross-entropy: (B*64, 13) vs (B*64,)
                loss = F.cross_entropy(
                    logits.permute(0, 2, 3, 1).reshape(-1, NUM_CLASSES),
                    y.reshape(-1),
                )

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if step % args.log_every == 0:
                with torch.no_grad():
                    preds = logits.argmax(dim=1)  # (B, 8, 8)
                    sq_acc = (preds == y).float().mean().item()
                    board_acc = (preds == y).all(dim=-1).all(dim=-1).float().mean().item()
                print(
                    f"step {step:6d} | loss {loss.item():.4f} | "
                    f"sq_acc {sq_acc:.4f} | board_acc {board_acc:.4f} | lr {lr:.2e}"
                )
                aim_run.track(loss.item(), name="loss", step=step, context={"subset": "train"})
                aim_run.track(sq_acc, name="sq_acc", step=step, context={"subset": "train"})
                aim_run.track(board_acc, name="board_acc", step=step, context={"subset": "train"})
                aim_run.track(lr, name="lr", step=step)

            # validation + checkpoint
            if step > 0 and step % args.save_every == 0:
                model.eval()
                val_sq_correct = 0
                val_board_correct = 0
                val_total = 0
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                    for vx, vy in val_loader:
                        vx = vx.to(device, non_blocking=True)
                        vy = vy.to(device, non_blocking=True)
                        vlogits = model(vx)
                        vpreds = vlogits.argmax(dim=1)
                        val_sq_correct += (vpreds == vy).sum().item()
                        val_board_correct += (vpreds == vy).all(dim=-1).all(dim=-1).sum().item()
                        val_total += vx.size(0)

                val_sq_acc = val_sq_correct / (val_total * 64)
                val_board_acc = val_board_correct / val_total
                print(f"  [val] sq_acc {val_sq_acc:.4f} | board_acc {val_board_acc:.4f}")
                aim_run.track(val_sq_acc, name="sq_acc", step=step, context={"subset": "val"})
                aim_run.track(val_board_acc, name="board_acc", step=step, context={"subset": "val"})

                ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "val_sq_acc": val_sq_acc,
                    "val_board_acc": val_board_acc,
                }
                torch.save(ckpt, out / f"ckpt_{step:06d}.pt")
                if val_board_acc > best_board_acc:
                    best_board_acc = val_board_acc
                    torch.save(ckpt, out / "ckpt_best.pt")
                    print(f"  [best] board_acc {val_board_acc:.4f}")

            step += 1

    # final save
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
    }, out / "ckpt_final.pt")
    aim_run.close()
    print(f"Done. {step} steps, output in {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="datasets/board_recognition")
    p.add_argument("--out-dir", default="runs/board_recognizer")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=2000)
    args = p.parse_args()
    train(args)
