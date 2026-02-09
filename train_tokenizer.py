"""
FSQ-VAE tokenizer training for 256x256 chess board frames.

Architecture:
  - Encoder: 4x stride-2 downsample → 16x16 latent grid (256 tokens/frame)
  - FSQ: [8,5,5,5] = 1000 implicit codes, dim=4
  - Decoder: 4x stride-2 upsample → 256x256 reconstruction
  - base_ch=64 (lightweight, sufficient for structured chess images)

Loss: L1 + 0.5 * Sobel-edge-L1 (no commitment loss needed with FSQ)

Usage:
  python split_dataset.py                          # generate train/val/test.txt
  python train_tokenizer.py --data datasets/lichess_1k
"""

import os
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image
from vector_quantize_pytorch import FSQ


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
class ChessFrameDataset(Dataset):
    """All PNG frames from game directories listed in a split file."""

    def __init__(self, split_file: str, frames_root: str):
        root = Path(frames_root)
        self.paths = []
        with open(split_file) as f:
            for line in f:
                game = line.strip()
                if not game:
                    continue
                self.paths.extend(sorted((root / game).glob("*.png")))
        if not self.paths:
            raise RuntimeError(f"No PNGs found via {split_file}")
        self.tf = transforms.ToTensor()  # [0,1] float32

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img)


# -------------------------------------------------------------------
# Sobel edge operator (for edge-aware loss)
# -------------------------------------------------------------------
class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        gx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        gy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("gx", gx.view(1, 1, 3, 3))
        self.register_buffer("gy", gy.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        gx = self.gx.expand(c, -1, -1, -1)
        gy = self.gy.expand(c, -1, -1, -1)
        x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
        dx = F.conv2d(x_pad, gx, groups=c)
        dy = F.conv2d(x_pad, gy, groups=c)
        return torch.sqrt(dx * dx + dy * dy + 1e-8)


# -------------------------------------------------------------------
# Model building blocks
# -------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            ResBlock(out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            ResBlock(out_ch),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------------------------
# FSQ-VAE
# -------------------------------------------------------------------
class FSQVAE(nn.Module):
    def __init__(self, levels=(8, 5, 5, 5), base_ch=64, quantize=True):
        super().__init__()
        latent_ch = base_ch * 4  # 256 — encoder/decoder work at full width
        self.quantize = quantize

        # Encoder: 256 → 128 → 64 → 32 → 16
        self.enc = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, padding=1),
            Down(base_ch, base_ch * 2),          # 256→128
            Down(base_ch * 2, latent_ch),        # 128→64
            Down(latent_ch, latent_ch),          # 64→32
            Down(latent_ch, latent_ch),          # 32→16
            ResBlock(latent_ch),
            nn.GroupNorm(32, latent_ch),
            nn.SiLU(),
        )

        # FSQ handles projection: latent_ch → len(levels) → quantize → latent_ch
        self.fsq = FSQ(list(levels), dim=latent_ch, channel_first=True)

        # Decoder: 16 → 32 → 64 → 128 → 256
        self.dec = nn.Sequential(
            Up(latent_ch, latent_ch),            # 16→32
            Up(latent_ch, latent_ch),            # 32→64
            Up(latent_ch, base_ch * 2),          # 64→128
            Up(base_ch * 2, base_ch),            # 128→256
            ResBlock(base_ch),
            nn.GroupNorm(32, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor):
        z_e = self.enc(x)                                  # (B, latent_ch, 16, 16)
        if self.quantize:
            z_q, indices = self.fsq(z_e)                   # project → quantize → project back
            return z_q, indices
        return z_e, None

    def decode(self, z_q: torch.Tensor):
        return self.dec(z_q)

    def forward(self, x: torch.Tensor):
        z_q, indices = self.encode(x)
        x_hat = self.decode(z_q)
        return x_hat, indices


# -------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay to 10%
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
    frames_root = data_root / "frames"

    train_ds = ChessFrameDataset(str(data_root / "train.txt"), str(frames_root))
    val_ds = ChessFrameDataset(str(data_root / "val.txt"), str(frames_root))
    print(f"Train: {len(train_ds)} frames | Val: {len(val_ds)} frames")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(args.batch_size, 32), shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    levels = tuple(int(x) for x in args.levels.split(","))
    model = FSQVAE(levels=levels, base_ch=args.base_ch, quantize=not args.no_quantize).to(device)
    sobel = Sobel().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    codebook_size = 1
    for l in levels:
        codebook_size *= l
    print(f"Model: {n_params / 1e6:.1f}M params | FSQ levels {levels} = {codebook_size} codes")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # grab a fixed val batch for consistent recon grids
    fixed_val = next(iter(val_loader))[:16].to(device)

    step = 0
    best_val_loss = float("inf")

    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break

            model.train()
            x = batch.to(device, non_blocking=True)

            lr = lr_at_step(step, args.steps, args.warmup, args.lr)
            for pg in opt.param_groups:
                pg["lr"] = lr

            with torch.amp.autocast("cuda", enabled=use_amp):
                x_hat, indices = model(x)
                l_rec = F.l1_loss(x_hat, x)
                l_edge = F.l1_loss(sobel(x_hat), sobel(x))
                loss = l_rec + args.edge_weight * l_edge

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # --- logging ---
            if step % args.log_every == 0:
                code_str = ""
                if indices is not None:
                    with torch.no_grad():
                        unique = indices.unique().numel()
                    code_str = f" | codes {unique}/{codebook_size}"
                print(
                    f"step {step:6d} | loss {loss.item():.4f} | "
                    f"rec {l_rec.item():.4f} | edge {l_edge.item():.4f}"
                    f"{code_str} | lr {lr:.2e}"
                )

            # --- checkpoint + recon grid ---
            if step > 0 and step % args.save_every == 0:
                # val loss
                model.eval()
                val_losses = []
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                    for i, vb in enumerate(val_loader):
                        if i >= 20:  # sample 20 batches
                            break
                        vx = vb.to(device, non_blocking=True)
                        vx_hat, _ = model(vx)
                        val_losses.append(F.l1_loss(vx_hat, vx).item())
                val_loss = sum(val_losses) / len(val_losses)
                print(f"  [val] rec_l1 {val_loss:.4f}")

                # save recon grid (fixed val batch)
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                    recon, _ = model(fixed_val)
                grid = make_grid(torch.cat([fixed_val, recon], dim=0), nrow=8)
                save_image(grid, out / f"recon_{step:06d}.png")

                # save checkpoint
                ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "levels": levels,
                    "base_ch": args.base_ch,
                    "val_loss": val_loss,
                }
                torch.save(ckpt, out / f"ckpt_{step:06d}.pt")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(ckpt, out / "ckpt_best.pt")
                    print(f"  [best] val_loss {val_loss:.4f}")

            step += 1

    # final save
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "levels": levels,
        "base_ch": args.base_ch,
    }, out / "ckpt_final.pt")
    print(f"Done. {step} steps, output in {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="datasets/lichess_1k",
                   help="dataset root (expects frames/, train.txt, val.txt)")
    p.add_argument("--out-dir", default="runs/tokenizer")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--edge-weight", type=float, default=0.5)
    p.add_argument("--levels", default="8,5,5,5", help="FSQ levels, comma-separated")
    p.add_argument("--no-quantize", action="store_true", help="disable FSQ (plain autoencoder sanity check)")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--base-ch", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=5000)
    args = p.parse_args()

    train(args)
