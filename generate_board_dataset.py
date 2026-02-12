"""
Generate a board-recognition dataset from Lichess PGN.

For each game, replays all moves and saves each unique board position
as a 256x256 PNG alongside a label grid (64 ints, one per square).

Class mapping:
  0=empty, 1=P, 2=N, 3=B, 4=R, 5=Q, 6=K, 7=p, 8=n, 9=b, 10=r, 11=q, 12=k

Usage:
  python generate_board_dataset.py --pgn lichess/lichess_db_standard_rated_2025-12.pgn
"""

import argparse
from pathlib import Path

import chess
import pyarrow as pa
import pyarrow.parquet as pq

from generate_dataset import render_board, parse_games

# piece → class index  (white uppercase 1-6, black lowercase 7-12, empty 0)
PIECE_TO_CLASS = {
    None: 0,
    chess.Piece(chess.PAWN, chess.WHITE): 1,
    chess.Piece(chess.KNIGHT, chess.WHITE): 2,
    chess.Piece(chess.BISHOP, chess.WHITE): 3,
    chess.Piece(chess.ROOK, chess.WHITE): 4,
    chess.Piece(chess.QUEEN, chess.WHITE): 5,
    chess.Piece(chess.KING, chess.WHITE): 6,
    chess.Piece(chess.PAWN, chess.BLACK): 7,
    chess.Piece(chess.KNIGHT, chess.BLACK): 8,
    chess.Piece(chess.BISHOP, chess.BLACK): 9,
    chess.Piece(chess.ROOK, chess.BLACK): 10,
    chess.Piece(chess.QUEEN, chess.BLACK): 11,
    chess.Piece(chess.KING, chess.BLACK): 12,
}


def board_to_label_grid(board: chess.Board) -> list[int]:
    """Return a flat list of 64 class indices (a1, b1, ..., h8 order → rank 0 file 0..7, rank 1 file 0..7, ...)."""
    labels = []
    for rank in range(8):
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            labels.append(PIECE_TO_CLASS[piece])
    return labels


def piece_placement_fen(board: chess.Board) -> str:
    """Return only the piece-placement portion of the FEN (first field)."""
    return board.board_fen()


def generate(args):
    out = Path(args.output)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    seen_fens: set[str] = set()
    records: list[dict] = []
    game_ids: list[int] = []

    print(f"Parsing {args.games} games from {args.pgn} ...")

    for game_idx, game in enumerate(parse_games(args.pgn, args.games)):
        if game_idx % 200 == 0:
            print(f"  game {game_idx}/{args.games}  ({len(records)} unique positions)")

        board = game.board()
        positions = [board.copy()]
        for move in game.mainline_moves():
            board.push(move)
            positions.append(board.copy())

        for pos in positions:
            fen = piece_placement_fen(pos)
            if fen in seen_fens:
                continue
            seen_fens.add(fen)

            idx = len(records)
            fname = f"{idx:06d}.png"
            img = render_board(pos)
            img.save(img_dir / fname, "PNG")

            records.append({
                "filename": fname,
                "fen": fen,
                "labels": board_to_label_grid(pos),
                "game_id": game_idx,
            })

        game_ids.append(game_idx)

    # split by game: first 80% of games → train, rest → val
    split_point = int(len(game_ids) * 0.8)
    train_game_set = set(game_ids[:split_point])

    train_recs = [r for r in records if r["game_id"] in train_game_set]
    val_recs = [r for r in records if r["game_id"] not in train_game_set]

    def save_parquet(recs, path):
        table = pa.table({
            "filename": [r["filename"] for r in recs],
            "fen": [r["fen"] for r in recs],
            "labels": [r["labels"] for r in recs],
        })
        pq.write_table(table, path)

    save_parquet(train_recs, out / "train.parquet")
    save_parquet(val_recs, out / "val.parquet")

    print(f"\nDone! {len(records)} unique positions")
    print(f"  Train: {len(train_recs)}  |  Val: {len(val_recs)}")
    print(f"  Output: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pgn", default="lichess/lichess_db_standard_rated_2025-12.pgn")
    p.add_argument("--output", default="datasets/board_recognition")
    p.add_argument("--games", type=int, default=2000)
    args = p.parse_args()
    generate(args)
