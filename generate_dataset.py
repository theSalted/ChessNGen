import chess
import chess.pgn
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

BOARD_SIZE = 256
SQUARE_SIZE = BOARD_SIZE // 8

LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
WHITE_PIECE = (255, 255, 255)
BLACK_PIECE = (0, 0, 0)

PIECE_SYMBOLS = {
    chess.KING: '\u265A',
    chess.QUEEN: '\u265B',
    chess.ROOK: '\u265C',
    chess.BISHOP: '\u265D',
    chess.KNIGHT: '\u265E',
    chess.PAWN: '\u265F',
}

_FONT = None

def get_font():
    global _FONT
    if _FONT is not None:
        return _FONT

    font_paths = [
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/noto/NotoSansSymbols2-Regular.ttf",
        "/usr/share/fonts/TTF/NotoSansSymbols2-Regular.ttf",
    ]
    for path in font_paths:
        try:
            _FONT = ImageFont.truetype(path, 26)
            return _FONT
        except (OSError, IOError):
            continue
    _FONT = ImageFont.load_default()
    return _FONT


def render_board(board: chess.Board) -> Image.Image:
    img = Image.new('RGB', (BOARD_SIZE, BOARD_SIZE))
    draw = ImageDraw.Draw(img)
    font = get_font()

    for rank in range(8):
        for file in range(8):
            display_rank = 7 - rank
            x_start = file * SQUARE_SIZE
            y_start = display_rank * SQUARE_SIZE

            is_light = (rank + file) % 2 == 1
            square_color = LIGHT_SQUARE if is_light else DARK_SQUARE

            draw.rectangle(
                [x_start, y_start, x_start + SQUARE_SIZE - 1, y_start + SQUARE_SIZE - 1],
                fill=square_color
            )

            square = chess.square(file, rank)
            piece = board.piece_at(square)

            if piece:
                center_x = x_start + SQUARE_SIZE // 2
                center_y = y_start + SQUARE_SIZE // 2

                symbol = PIECE_SYMBOLS[piece.piece_type]
                piece_color = WHITE_PIECE if piece.color == chess.WHITE else BLACK_PIECE

                bbox = draw.textbbox((0, 0), symbol, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                text_x = center_x - text_w // 2 - bbox[0]
                text_y = center_y - text_h // 2 - bbox[1]
                draw.text((text_x, text_y), symbol, fill=piece_color, font=font)

    return img


def parse_games(pgn_path: str, max_games: int = None):
    with open(pgn_path, 'r') as f:
        game_count = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            yield game
            game_count += 1
            if max_games and game_count >= max_games:
                break


def generate_dataset(pgn_path: str, output_dir: str, num_games: int = 1000, context_length: int = 4):
    output_path = Path(output_dir)
    frames_dir = output_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    transitions = []

    print(f"Processing {num_games} games with K={context_length}...")

    for game_idx, game in enumerate(parse_games(pgn_path, num_games)):
        if game_idx % 100 == 0:
            print(f"  Game {game_idx}/{num_games}")

        game_id = game_idx + 1
        game_dir = frames_dir / f"game_{game_id:06d}"
        game_dir.mkdir(exist_ok=True)

        board = game.board()
        positions = [board.copy()]

        for move in game.mainline_moves():
            board.push(move)
            positions.append(board.copy())

        frame_paths = []
        for frame_idx, pos in enumerate(positions):
            frame_path = game_dir / f"f{frame_idx:04d}.png"
            img = render_board(pos)
            img.save(frame_path, "PNG")
            frame_paths.append(str(frame_path.relative_to(output_path)))

        for t in range(context_length, len(positions)):
            in_paths = frame_paths[t - context_length:t]
            out_path = frame_paths[t]

            transitions.append({
                'game_id': game_id,
                't': t,
                'k': context_length,
                'in_paths': in_paths,
                'out_path': out_path,
            })

    print(f"Saving {len(transitions)} transitions to parquet...")

    table = pa.table({
        'game_id': [t['game_id'] for t in transitions],
        't': [t['t'] for t in transitions],
        'k': [t['k'] for t in transitions],
        'in_paths': [t['in_paths'] for t in transitions],
        'out_path': [t['out_path'] for t in transitions],
    })

    pq.write_table(table, output_path / "transitions.parquet")

    print(f"Done! Output in {output_path}")
    print(f"  Frames: {frames_dir}")
    print(f"  Transitions: {output_path / 'transitions.parquet'}")
    print(f"  Total transitions: {len(transitions)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", default="lichess/lichess_db_standard_rated_2025-12.pgn")
    parser.add_argument("--output", default="datasets/lichess_1k")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--context", type=int, default=4)

    args = parser.parse_args()

    generate_dataset(
        pgn_path=args.pgn,
        output_dir=args.output,
        num_games=args.games,
        context_length=args.context,
    )
