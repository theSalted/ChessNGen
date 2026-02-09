"""
Pre-compute 100 opening sequences as FSQ-VAE tokens.

Reads games 1001–1100 from the Lichess PGN (outside the train/val/test set),
extracts the first 4 half-moves of each game, renders and encodes each board
position through the FSQ-VAE, and saves:

  opening_tokens.npy   — (100, 4, 256) uint16  (~200 KB)
  opening_moves.txt    — one line per game with the 4 half-moves in SAN

Usage:
  python save_opening_tokens.py
  python save_opening_tokens.py --pgn lichess/lichess_db_standard_rated_2025-12.pgn
"""

import argparse
import json
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch
from torchvision import transforms

from generate_dataset import render_board
from train_tokenizer import FSQVAE

HALF_MOVES = 4  # positions after each of the first 4 half-moves


def load_vae(ckpt_path: str, device: torch.device) -> FSQVAE:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FSQVAE(levels=ckpt["levels"], base_ch=ckpt["base_ch"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded VAE from {ckpt_path} (step {ckpt.get('step', '?')})")
    return model


def encode_board(vae: FSQVAE, board: chess.Board, device: torch.device,
                 to_tensor: transforms.ToTensor) -> np.ndarray:
    """Render a board and encode to flat token indices (256,) uint16."""
    img = render_board(board)
    t = to_tensor(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        _, indices = vae.encode(t)  # (1, 16, 16)
    return indices[0].reshape(-1).cpu().numpy().astype(np.uint16)


def main():
    parser = argparse.ArgumentParser(description="Save opening tokens for 100 games")
    parser.add_argument(
        "--pgn",
        default="lichess/lichess_db_standard_rated_2025-12.pgn",
        help="path to Lichess PGN file",
    )
    parser.add_argument(
        "--ckpt",
        default="runs/tokenizer_v2/ckpt_best.pt",
        help="path to FSQ-VAE checkpoint",
    )
    parser.add_argument("--skip", type=int, default=1000,
                        help="number of games to skip (already in dataset)")
    parser.add_argument("--num-games", type=int, default=100,
                        help="number of opening sequences to collect")
    parser.add_argument("--out-tokens", default="weights/openings/opening_tokens.npy")
    parser.add_argument("--out-moves", default="weights/openings/opening_moves.txt")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    vae = load_vae(args.ckpt, device)
    to_tensor = transforms.ToTensor()

    # --- Read games from PGN, skipping the first `skip` ---
    pgn_path = Path(args.pgn)
    if not pgn_path.exists():
        raise FileNotFoundError(f"PGN not found: {pgn_path}")

    all_tokens = []   # list of (4, 256) arrays
    all_moves = []     # list of move strings

    print(f"Skipping first {args.skip} games...")
    with open(pgn_path) as f:
        # Skip games already in the dataset
        for i in range(args.skip):
            game = chess.pgn.read_game(f)
            if game is None:
                raise RuntimeError(f"PGN ended after only {i} games (expected {args.skip})")
            if (i + 1) % 200 == 0:
                print(f"  skipped {i + 1}/{args.skip}")

        print(f"Collecting {args.num_games} openings (games {args.skip + 1}–{args.skip + args.num_games})...")
        games_read = 0
        while len(all_tokens) < args.num_games:
            game = chess.pgn.read_game(f)
            if game is None:
                raise RuntimeError(
                    f"PGN exhausted after {games_read} extra games, "
                    f"only collected {len(all_tokens)}/{args.num_games}"
                )
            games_read += 1

            # Extract first HALF_MOVES positions (after each move)
            board = game.board()
            moves_san = []
            positions = []
            for move in game.mainline_moves():
                moves_san.append(board.san(move))
                board.push(move)
                positions.append(board.copy())
                if len(positions) >= HALF_MOVES:
                    break

            if len(positions) < HALF_MOVES:
                # Game too short, skip it
                continue

            # Encode all 4 positions
            tokens = np.stack([
                encode_board(vae, pos, device, to_tensor)
                for pos in positions
            ])  # (4, 256)
            all_tokens.append(tokens)
            all_moves.append(" ".join(moves_san))

            if len(all_tokens) % 10 == 0:
                print(f"  {len(all_tokens)}/{args.num_games} openings encoded")

    # Stack into single array
    tokens_arr = np.stack(all_tokens).astype(np.uint16)  # (100, 4, 256)
    print(f"\nTokens shape: {tokens_arr.shape}, dtype: {tokens_arr.dtype}")
    print(f"Value range: [{tokens_arr.min()}, {tokens_arr.max()}]")

    # Save tokens
    np.save(args.out_tokens, tokens_arr)
    token_size = Path(args.out_tokens).stat().st_size
    print(f"Saved tokens to {args.out_tokens} ({token_size:,} bytes)")

    # Save move annotations
    with open(args.out_moves, "w") as f:
        for i, moves in enumerate(all_moves):
            f.write(f"{i:03d} {moves}\n")
    print(f"Saved move list to {args.out_moves}")

    # Show first 10 openings
    print("\nFirst 10 openings:")
    for i, moves in enumerate(all_moves[:10]):
        print(f"  [{i:03d}] {moves}")


if __name__ == "__main__":
    main()
