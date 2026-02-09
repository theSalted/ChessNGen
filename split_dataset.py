"""Split chess game directories into train/val/test by game (no cross-split leakage)."""

import random
import argparse
from pathlib import Path


def split_dataset(frames_dir: str, output_dir: str, seed: int = 42,
                  train_ratio: float = 0.8, val_ratio: float = 0.1):
    games = sorted(d.name for d in Path(frames_dir).iterdir() if d.is_dir())
    n = len(games)
    assert n > 0, f"No game directories found in {frames_dir}"

    rng = random.Random(seed)
    rng.shuffle(games)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": games[:n_train],
        "val": games[n_train:n_train + n_val],
        "test": games[n_train + n_val:],
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, game_list in splits.items():
        path = out / f"{name}.txt"
        path.write_text("\n".join(game_list) + "\n")
        # count total frames
        total_frames = sum(
            len(list((Path(frames_dir) / g).glob("*.png")))
            for g in game_list
        )
        print(f"{name}: {len(game_list)} games, {total_frames} frames -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="datasets/lichess_1k/frames")
    parser.add_argument("--output", default="datasets/lichess_1k")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_dataset(args.frames, args.output, args.seed)
