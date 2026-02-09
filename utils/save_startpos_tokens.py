"""
Pre-compute the standard chess starting position as FSQ-VAE tokens.

Saves a tiny (4 × 256) uint16 array so that inference can skip rendering
and encoding entirely when bootstrapping from the default start position.

Usage:
  python save_startpos_tokens.py
  python save_startpos_tokens.py --ckpt runs/tokenizer_v2/ckpt_best.pt --out weights/startpos_tokens.npy
"""

import argparse
from pathlib import Path

import chess
import numpy as np
import torch
from torchvision import transforms

from generate_dataset import render_board
from train_tokenizer import FSQVAE


def main():
    parser = argparse.ArgumentParser(description="Save starting-position tokens")
    parser.add_argument(
        "--ckpt",
        default="runs/tokenizer_v2/ckpt_best.pt",
        help="path to FSQ-VAE checkpoint",
    )
    parser.add_argument(
        "--out",
        default="weights/startpos_tokens.npy",
        help="output .npy file",
    )
    parser.add_argument(
        "--k", type=int, default=4, help="context length (number of repeated frames)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load VAE
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = FSQVAE(levels=ckpt["levels"], base_ch=ckpt["base_ch"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded VAE from {args.ckpt} (step {ckpt.get('step', '?')})")

    # Render starting position
    board = chess.Board()
    img = render_board(board)
    print(f"Rendered starting position: {img.size}")

    # Encode to tokens
    to_tensor = transforms.ToTensor()
    t = to_tensor(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        _, indices = model.encode(t)  # (1, 16, 16)
    flat = indices[0].reshape(-1).cpu().numpy().astype(np.uint16)  # (256,)
    print(f"Encoded to {flat.shape} tokens, range [{flat.min()}, {flat.max()}]")

    # Tile k times → (k, 256)
    tokens = np.tile(flat, (args.k, 1))  # (k, 256)
    print(f"Tiled to shape {tokens.shape}, dtype {tokens.dtype}")

    # Save
    np.save(args.out, tokens)
    file_size = Path(args.out).stat().st_size
    print(f"Saved to {args.out} ({file_size} bytes)")


if __name__ == "__main__":
    main()
