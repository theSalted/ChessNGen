"""
GPT-style dynamics transformer for chess board state prediction.

Given k=4 context frames (4 × 256 = 1024 tokens), the model autoregressively
predicts the next frame (256 tokens) using discrete FSQ-VAE token sequences.

Architecture (~7M params):
  - d_model=256, n_heads=8, n_layers=8, d_ff=1024
  - Factored positional embeddings: frame + row + col
  - Weight-tied output head
  - Pre-norm GPT-2 style via nn.TransformerEncoder

Usage:
  python train_dynamics.py
  python train_dynamics.py --d-model 384 --n-layers 12
  python train_dynamics.py --out-dir runs/dynamics --steps 50000
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from train_tokenizer import FSQVAE, lr_at_step


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
class TransitionsTokenDataset(Dataset):
    """Loads token transitions filtered by train/val split."""

    def __init__(self, parquet_path: str, data_root: str, split_file: str):
        data_root = Path(data_root)
        table = pq.read_table(parquet_path)

        # Read split file → set of game_ids
        with open(split_file) as f:
            game_names = {line.strip() for line in f if line.strip()}
        game_ids = {int(name.split("_")[1]) for name in game_names}

        # Filter rows by game_id
        all_game_ids = table.column("game_id").to_pylist()
        mask = [gid in game_ids for gid in all_game_ids]

        self.token_files = [table.column("token_file")[i].as_py() for i, m in enumerate(mask) if m]
        self.in_starts = [table.column("in_start")[i].as_py() for i, m in enumerate(mask) if m]
        self.out_idxs = [table.column("out_idx")[i].as_py() for i, m in enumerate(mask) if m]
        self.k = table.column("k")[0].as_py()

        # Cache loaded .npy files in memory
        self.data_root = data_root
        self._cache: dict[str, np.ndarray] = {}

        print(f"  {split_file}: {len(self)} transitions from {len(game_ids)} games (k={self.k})")

    def _load_tokens(self, token_file: str) -> np.ndarray:
        if token_file not in self._cache:
            self._cache[token_file] = np.load(self.data_root / token_file)
        return self._cache[token_file]

    def __len__(self):
        return len(self.token_files)

    def __getitem__(self, idx):
        tokens = self._load_tokens(self.token_files[idx])  # (T, 16, 16)
        in_start = self.in_starts[idx]
        out_idx = self.out_idxs[idx]

        # Context: k frames flattened
        context = tokens[in_start : in_start + self.k].reshape(-1).astype(np.int64)
        # Target: 1 frame flattened
        target = tokens[out_idx].reshape(-1).astype(np.int64)

        return torch.from_numpy(context), torch.from_numpy(target)


# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------
class DynamicsTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        n_frames: int = 5,
        grid_size: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.grid_size = grid_size
        self.tokens_per_frame = grid_size * grid_size
        self.seq_len = n_frames * self.tokens_per_frame  # 1280

        # Token embedding
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # Factored positional embeddings
        self.frame_emb = nn.Embedding(n_frames, d_model)
        self.row_emb = nn.Embedding(grid_size, d_model)
        self.col_emb = nn.Embedding(grid_size, d_model)

        # Transformer backbone (pre-norm GPT-2 style)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.ln_f = nn.LayerNorm(d_model)

        # Output head (weight-tied with token embedding)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        # Pre-compute positional indices
        positions = torch.arange(self.seq_len)
        self.register_buffer("frame_ids", positions // self.tokens_per_frame)
        self.register_buffer("row_ids", (positions % self.tokens_per_frame) // grid_size)
        self.register_buffer("col_ids", positions % grid_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len) token indices — positions 0..1278
        Returns: (B, seq_len, vocab_size) logits
        """
        B, T = x.shape

        # Embeddings
        tok = self.tok_emb(x)
        pos = (
            self.frame_emb(self.frame_ids[:T])
            + self.row_emb(self.row_ids[:T])
            + self.col_emb(self.col_ids[:T])
        )
        h = tok + pos

        # Causal transformer
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.transformer(h, mask=mask, is_causal=True)
        h = self.ln_f(h)

        return self.head(h)


# -------------------------------------------------------------------
# Autoregressive generation
# -------------------------------------------------------------------
@torch.no_grad()
def generate(
    model: DynamicsTransformer,
    context: torch.Tensor,
    n_tokens: int = 256,
    temperature: float = 0.0,
    top_k: int = 0,
) -> torch.Tensor:
    """
    Autoregressively generate n_tokens given context.
    context: (B, context_len) token indices
    Returns: (B, n_tokens) generated token indices
    """
    model.eval()
    device = context.device
    generated = []

    seq = context
    for _ in range(n_tokens):
        logits = model(seq)[:, -1, :]  # (B, vocab)

        if temperature <= 0:
            # Greedy
            tok = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                v, _ = logits.topk(top_k, dim=-1)
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            tok = torch.multinomial(probs, 1)

        generated.append(tok)
        seq = torch.cat([seq, tok], dim=1)

    return torch.cat(generated, dim=1)


# -------------------------------------------------------------------
# Visualization helpers
# -------------------------------------------------------------------
def decode_tokens_to_image(
    vae: FSQVAE, tokens: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Decode flat token indices (256,) → image (3, 256, 256)."""
    tok = tokens.view(1, 16, 16).long().to(device)
    z_q = vae.fsq.indices_to_codes(tok)
    img = vae.decode(z_q)
    return img.squeeze(0).cpu()


def save_prediction_grid(
    vae: FSQVAE,
    model: DynamicsTransformer,
    samples: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    out_path: str,
):
    """Generate prediction grid: [last context | predicted | ground truth] per row."""
    images = []
    for context, target in samples:
        context = context.unsqueeze(0).to(device)
        target = target.to(device)

        # Last context frame
        last_ctx = context[0, -256:]
        img_ctx = decode_tokens_to_image(vae, last_ctx, device)

        # Generate prediction
        pred_tokens = generate(model, context, n_tokens=256, temperature=0.0)
        pred = pred_tokens[0]
        img_pred = decode_tokens_to_image(vae, pred, device)

        # Ground truth
        img_gt = decode_tokens_to_image(vae, target, device)

        images.extend([img_ctx, img_pred, img_gt])

    grid = torch.stack(images)
    save_image(grid, out_path, nrow=3)


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
def train(args):
    device = torch.device(args.device)
    use_amp = (device.type == "cuda") and not args.no_amp
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data)
    parquet_path = data_root / "transitions_tokens.parquet"

    # Datasets
    print("Loading datasets...")
    train_ds = TransitionsTokenDataset(
        str(parquet_path), str(data_root), str(data_root / "train.txt")
    )
    val_ds = TransitionsTokenDataset(
        str(parquet_path), str(data_root), str(data_root / "val.txt")
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    model = DynamicsTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M params | d={args.d_model} h={args.n_heads} L={args.n_layers}")

    # Optimizer
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Load VAE for visualization
    vae = None
    if args.vae_ckpt and Path(args.vae_ckpt).exists():
        ckpt = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
        vae = FSQVAE(levels=ckpt["levels"], base_ch=ckpt["base_ch"]).to(device)
        vae.load_state_dict(ckpt["model"])
        vae.eval()
        print(f"Loaded VAE from {args.vae_ckpt}")

    # Fixed val samples for visualization
    fixed_samples = [val_ds[i] for i in range(min(4, len(val_ds)))]

    # Context length
    k = train_ds.k
    ctx_len = k * 256  # 1024
    tgt_len = 256

    step = 0
    best_val_loss = float("inf")

    print(f"\nTraining for {args.steps} steps (batch_size={args.batch_size})")
    print(f"  Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    print(f"  ~{len(train_ds) // args.batch_size} steps/epoch\n")

    while step < args.steps:
        for context, target in train_loader:
            if step >= args.steps:
                break

            model.train()
            context = context.to(device, non_blocking=True)  # (B, 1024)
            target = target.to(device, non_blocking=True)    # (B, 256)

            # LR schedule
            lr = lr_at_step(step, args.steps, args.warmup, args.lr)
            for pg in opt.param_groups:
                pg["lr"] = lr

            # Build input sequence: context + target[:-1] (teacher forcing)
            # Input:  [ctx_0 ... ctx_1023, tgt_0 ... tgt_254]  (1279 tokens)
            # Target: logits at positions 1023..1278 predict tgt_0..tgt_255
            inp = torch.cat([context, target[:, :-1]], dim=1)  # (B, 1279)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(inp)  # (B, 1279, vocab)
                # Loss only on target positions: logits[1023:1279] → target[0:256]
                pred_logits = logits[:, ctx_len - 1 :, :]  # (B, 256, vocab)
                loss = F.cross_entropy(
                    pred_logits.reshape(-1, args.vocab_size),
                    target.reshape(-1),
                )

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # Logging
            if step % args.log_every == 0:
                with torch.no_grad():
                    acc = (pred_logits.argmax(-1) == target).float().mean().item()
                print(
                    f"step {step:6d} | loss {loss.item():.4f} | "
                    f"acc {acc:.4f} | lr {lr:.2e}"
                )

            # Checkpoint + validation
            if step > 0 and step % args.save_every == 0:
                model.eval()
                val_losses = []
                val_accs = []
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                    for i, (vc, vt) in enumerate(val_loader):
                        if i >= 50:
                            break
                        vc = vc.to(device, non_blocking=True)
                        vt = vt.to(device, non_blocking=True)
                        vinp = torch.cat([vc, vt[:, :-1]], dim=1)
                        vlogits = model(vinp)
                        vpred = vlogits[:, ctx_len - 1 :, :]
                        vloss = F.cross_entropy(
                            vpred.reshape(-1, args.vocab_size),
                            vt.reshape(-1),
                        )
                        vacc = (vpred.argmax(-1) == vt).float().mean().item()
                        val_losses.append(vloss.item())
                        val_accs.append(vacc)

                val_loss = sum(val_losses) / len(val_losses)
                val_acc = sum(val_accs) / len(val_accs)
                print(f"  [val] loss {val_loss:.4f} | acc {val_acc:.4f}")

                # Save checkpoint
                ckpt_data = {
                    "step": step,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                }
                torch.save(ckpt_data, out / f"ckpt_{step:06d}.pt")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(ckpt_data, out / "ckpt_best.pt")
                    print(f"  [best] val_loss {val_loss:.4f}")

                # Visualization
                if vae is not None:
                    save_prediction_grid(
                        vae, model, fixed_samples, device,
                        str(out / f"pred_{step:06d}.png"),
                    )

            step += 1

    # Final save
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
        },
        out / "ckpt_final.pt",
    )
    print(f"Done. {step} steps, output in {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train dynamics transformer")
    p.add_argument("--data", default="datasets/lichess_1k")
    p.add_argument("--out-dir", default="runs/dynamics")
    p.add_argument("--vae-ckpt", default="runs/tokenizer_v2/ckpt_best.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--vocab-size", type=int, default=1000)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    train(args)
