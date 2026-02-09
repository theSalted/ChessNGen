"""
Encode all chess board frames to FSQ-VAE discrete token grids.

Loads the trained FSQ-VAE checkpoint, encodes every frame across all 1000
games to token indices (shape [16, 16], values 0-999), and saves:
  - Per-game token arrays: tokens/game_XXXXXX.npy  (shape [T, 16, 16], uint16)
  - transitions_tokens.parquet: same transitions with token_file / indices

Usage:
  python encode_frames.py
  python encode_frames.py --ckpt runs/tokenizer_v2/ckpt_best.pt
  python encode_frames.py --verify        # spot-check decode quality
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# Import the model class from train_tokenizer
from train_tokenizer import FSQVAE


def load_model(ckpt_path: str, device: torch.device) -> FSQVAE:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FSQVAE(levels=ckpt["levels"], base_ch=ckpt["base_ch"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path} (step {ckpt.get('step', '?')})")
    return model


def collect_game_frames(frames_root: Path):
    """Yield (game_dir_name, sorted_list_of_png_paths) for each game."""
    game_dirs = sorted(
        d for d in frames_root.iterdir() if d.is_dir() and d.name.startswith("game_")
    )
    for game_dir in game_dirs:
        pngs = sorted(game_dir.glob("*.png"))
        if pngs:
            yield game_dir.name, pngs


def encode_game(
    model: FSQVAE,
    frame_paths: list[Path],
    device: torch.device,
    batch_size: int,
    tf: transforms.ToTensor,
) -> np.ndarray:
    """Encode all frames of a game, return token array [T, 16, 16] uint16."""
    all_indices = []
    for i in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(tf(img))
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            _, indices = model.encode(batch)  # (B, 16, 16)
        all_indices.append(indices.cpu().numpy())
    tokens = np.concatenate(all_indices, axis=0).astype(np.uint16)
    return tokens


def verify_reconstruction(
    model: FSQVAE,
    tokens_dir: Path,
    frames_root: Path,
    device: torch.device,
    num_samples: int = 4,
    out_path: str = "verify_tokens.png",
):
    """Decode a few token arrays back through the decoder, compare to originals."""
    tf = transforms.ToTensor()
    originals = []
    reconstructions = []

    # Pick first game with enough frames
    game_dirs = sorted(
        d for d in frames_root.iterdir() if d.is_dir() and d.name.startswith("game_")
    )
    for game_dir in game_dirs[:1]:
        token_file = tokens_dir / f"{game_dir.name}.npy"
        if not token_file.exists():
            continue
        tokens = np.load(token_file)
        pngs = sorted(game_dir.glob("*.png"))

        # Pick evenly spaced frames
        n = len(pngs)
        sample_idxs = [int(i * n / num_samples) for i in range(num_samples)]

        for idx in sample_idxs:
            # Original
            orig = tf(Image.open(pngs[idx]).convert("RGB"))
            originals.append(orig)

            # Decode from tokens
            tok = torch.from_numpy(tokens[idx : idx + 1].astype(np.int64)).to(device)
            with torch.no_grad():
                # indices_to_codes handles project_out + channel_first permutation
                z_q = model.fsq.indices_to_codes(tok)  # (1, latent_ch, 16, 16)
                recon = model.decode(z_q)  # (1, 3, 256, 256)
            reconstructions.append(recon.squeeze(0).cpu())

    if not originals:
        print("No games found for verification!")
        return

    # Stack: top row originals, bottom row reconstructions
    grid = torch.stack(originals + reconstructions)
    save_image(grid, out_path, nrow=num_samples)
    print(f"Verification grid saved to {out_path}")

    # Compute L1 error
    orig_t = torch.stack(originals)
    recon_t = torch.stack(reconstructions)
    l1 = (orig_t - recon_t).abs().mean().item()
    print(f"  Mean L1 error (orig vs token-decoded): {l1:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Encode chess frames to FSQ tokens")
    parser.add_argument(
        "--ckpt",
        default="runs/tokenizer_v2/ckpt_best.pt",
        help="path to FSQ-VAE checkpoint",
    )
    parser.add_argument(
        "--data", default="datasets/lichess_1k", help="dataset root directory"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="encoding batch size"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--verify", action="store_true", help="spot-check decode quality after encoding"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    data_root = Path(args.data)
    frames_root = data_root / "frames"
    tokens_dir = data_root / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.ckpt, device)
    tf = transforms.ToTensor()

    # Read original transitions.parquet for context length (k)
    orig_table = pq.read_table(data_root / "transitions.parquet")
    k = orig_table["k"][0].as_py()

    # Encode all games
    total_frames = 0
    game_frame_counts = {}  # game_name -> frame_count

    print(f"\nEncoding frames from {frames_root} → {tokens_dir}")
    for game_name, frame_paths in collect_game_frames(frames_root):
        tokens = encode_game(model, frame_paths, device, args.batch_size, tf)
        out_file = tokens_dir / f"{game_name}.npy"
        np.save(out_file, tokens)
        game_frame_counts[game_name] = len(frame_paths)
        total_frames += len(frame_paths)

        if len(game_frame_counts) % 100 == 0:
            print(f"  {len(game_frame_counts)} games encoded ({total_frames} frames)")

    print(f"\nEncoded {total_frames} frames across {len(game_frame_counts)} games")

    # Compute total size on disk
    total_bytes = sum(f.stat().st_size for f in tokens_dir.glob("*.npy"))
    print(f"Token cache size: {total_bytes / 1024 / 1024:.1f} MB")

    # Verify shapes
    print("\nShape verification (sample):")
    for game_name, frame_paths in list(collect_game_frames(frames_root))[:3]:
        npy = np.load(tokens_dir / f"{game_name}.npy")
        expected_t = len(frame_paths)
        status = "OK" if npy.shape == (expected_t, 16, 16) else "MISMATCH"
        print(f"  {game_name}: {npy.shape} (expected ({expected_t}, 16, 16)) [{status}]")

    # Generate transitions_tokens.parquet
    print(f"\nGenerating transitions_tokens.parquet (k={k})...")
    game_ids = []
    ts = []
    ks = []
    token_files = []
    in_starts = []
    out_idxs = []

    for game_name, n_frames in sorted(game_frame_counts.items()):
        game_id = int(game_name.split("_")[1])
        token_file = f"tokens/{game_name}.npy"

        for t in range(k, n_frames):
            game_ids.append(game_id)
            ts.append(t)
            ks.append(k)
            token_files.append(token_file)
            in_starts.append(t - k)
            out_idxs.append(t)

    table = pa.table(
        {
            "game_id": pa.array(game_ids, type=pa.int64()),
            "t": pa.array(ts, type=pa.int64()),
            "k": pa.array(ks, type=pa.int64()),
            "token_file": pa.array(token_files, type=pa.string()),
            "in_start": pa.array(in_starts, type=pa.int64()),
            "out_idx": pa.array(out_idxs, type=pa.int64()),
        }
    )
    parquet_path = data_root / "transitions_tokens.parquet"
    pq.write_table(table, parquet_path)
    print(f"Saved {table.num_rows} transitions to {parquet_path}")

    # Verify row count matches original
    if table.num_rows == orig_table.num_rows:
        print(f"  Row count matches original transitions.parquet ({table.num_rows})")
    else:
        print(
            f"  WARNING: row count {table.num_rows} != original {orig_table.num_rows}"
        )

    # Optional verification
    if args.verify:
        print("\nRunning decode verification...")
        verify_reconstruction(model, tokens_dir, frames_root, device)


if __name__ == "__main__":
    main()
