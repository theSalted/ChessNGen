"""
Evaluate single-step move legality of the dynamics model.

For each test context (k=4 real game frames), generates one next frame,
recognizes FEN of the last context frame and generated frame, and checks
whether the transition is a legal chess move.

Usage:
  python eval_legality.py
  python eval_legality.py --dynamics weights/dynamics_v2.1.pt --n 200 --temperature 0.8
"""

import argparse
from pathlib import Path

import chess
import numpy as np
import torch
from torchvision import transforms

from train_dynamics import DynamicsTransformer, TransitionsTokenDataset, generate
from train_tokenizer import FSQVAE
from train_board_recognizer import BoardRecognizer, grid_to_fen
from infer import load_vae, load_dynamics, decode_tokens_to_pil


def load_recognizer(weights_path: str, device: torch.device) -> BoardRecognizer:
    model = BoardRecognizer().to(device)
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


_to_tensor = transforms.ToTensor()


def recognize_pil(recognizer: BoardRecognizer, img, device: torch.device) -> str:
    x = _to_tensor(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = recognizer(x)
        preds = logits.argmax(dim=1).squeeze(0)
    return grid_to_fen(preds)


def is_legal_transition(fen_before: str, fen_after: str) -> dict:
    """
    Check if fen_after can be reached from fen_before by a single legal move
    (by either side). Returns a dict with detailed results.
    """
    result = {
        "fen_before": fen_before,
        "fen_after": fen_after,
        "identical": fen_before == fen_after,
        "legal": False,
        "move": None,
        "side": None,
        "piece_count_before": 0,
        "piece_count_after": 0,
        "pieces_vanished": False,
    }

    # Count pieces
    pieces_before = sum(1 for c in fen_before if c.isalpha())
    pieces_after = sum(1 for c in fen_after if c.isalpha())
    result["piece_count_before"] = pieces_before
    result["piece_count_after"] = pieces_after
    # A legal move can remove at most 1 piece (capture). Promotion doesn't
    # change count. If more than 1 piece vanished, something is wrong.
    result["pieces_vanished"] = (pieces_before - pieces_after) > 1

    if result["identical"]:
        return result

    # Try both sides - we don't know whose turn it is from images alone
    for color, side_name in [(chess.WHITE, "white"), (chess.BLACK, "black")]:
        full_fen = f"{fen_before} {'w' if color == chess.WHITE else 'b'} KQkq - 0 1"
        try:
            board = chess.Board(full_fen)
        except ValueError:
            continue

        for move in board.legal_moves:
            board.push(move)
            result_fen = board.fen().split(" ")[0]
            if result_fen == fen_after:
                result["legal"] = True
                result["move"] = move.uci()
                result["side"] = side_name
                return result
            board.pop()

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dynamics", default="weights/dynamics_v2.1.pt")
    p.add_argument("--vae", default="weights/tokenizer_best.pt")
    p.add_argument("--recognizer", default="weights/board_recognizer.pt")
    p.add_argument("--data", default="datasets/lichess_1k")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--n", type=int, default=100, help="number of transitions to evaluate")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {device}")
    print(f"Loading models...")

    dynamics = load_dynamics(args.dynamics, device)
    vae = load_vae(args.vae, device)
    recognizer = load_recognizer(args.recognizer, device)

    print(f"Loading dataset ({args.split} split)...")
    data_root = Path(args.data)
    ds = TransitionsTokenDataset(
        parquet_path=str(data_root / "transitions_tokens.parquet"),
        data_root=str(data_root),
        split_file=str(data_root / f"{args.split}.txt"),
    )

    # Sample random indices
    n = min(args.n, len(ds))
    indices = np.random.choice(len(ds), size=n, replace=False)

    print(f"Evaluating {n} single-step predictions (temp={args.temperature}, top_k={args.top_k})...")
    print("-" * 70)

    stats = {
        "legal": 0,
        "identical": 0,
        "pieces_vanished": 0,
        "illegal": 0,
        "total": 0,
    }

    for i, idx in enumerate(indices):
        context, target_gt = ds[idx]
        context = context.to(device)

        # Generate one next frame
        with torch.no_grad():
            pred_tokens = generate(
                dynamics,
                context.unsqueeze(0),
                n_tokens=256,
                temperature=args.temperature,
                top_k=args.top_k,
            )[0]  # (256,)

        # Decode last context frame and predicted frame to images
        last_context_tokens = context[-256:]
        img_before = decode_tokens_to_pil(vae, last_context_tokens, device)
        img_after = decode_tokens_to_pil(vae, pred_tokens, device)

        # Recognize FENs
        fen_before = recognize_pil(recognizer, img_before, device)
        fen_after = recognize_pil(recognizer, img_after, device)

        # Check legality
        result = is_legal_transition(fen_before, fen_after)

        stats["total"] += 1
        if result["identical"]:
            stats["identical"] += 1
            label = "IDENTICAL"
        elif result["legal"]:
            stats["legal"] += 1
            label = f"LEGAL ({result['side']} {result['move']})"
        else:
            stats["illegal"] += 1
            if result["pieces_vanished"]:
                stats["pieces_vanished"] += 1
            label = "ILLEGAL"
            if result["pieces_vanished"]:
                label += f" (pieces vanished: {result['piece_count_before']}→{result['piece_count_after']})"

        if (i + 1) % 10 == 0 or not result["legal"]:
            print(f"  [{i+1:4d}/{n}] {label:40s}  {fen_before} → {fen_after}")

    # Summary
    print("=" * 70)
    print(f"RESULTS ({n} single-step predictions)")
    print(f"  Legal moves:      {stats['legal']:4d} ({100*stats['legal']/n:.1f}%)")
    print(f"  No move (same):   {stats['identical']:4d} ({100*stats['identical']/n:.1f}%)")
    print(f"  Illegal moves:    {stats['illegal']:4d} ({100*stats['illegal']/n:.1f}%)")
    print(f"    - pieces vanish:{stats['pieces_vanished']:4d} ({100*stats['pieces_vanished']/n:.1f}%)")
    print(f"  Total:            {stats['total']:4d}")


if __name__ == "__main__":
    main()
