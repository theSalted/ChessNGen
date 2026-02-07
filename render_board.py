import chess
from PIL import Image, ImageDraw, ImageFont

# Parse the first move from the first game: 1. d4
board = chess.Board()
board.push_san("d4")

# Board rendering settings
BOARD_SIZE = 256
SQUARE_SIZE = BOARD_SIZE // 8  # 32 pixels per square

# Colors (RGB)
LIGHT_SQUARE = (240, 217, 181)  # Light tan
DARK_SQUARE = (181, 136, 99)    # Dark brown
WHITE_PIECE = (255, 255, 255)
BLACK_PIECE = (0, 0, 0)

# Unicode chess symbols (use filled symbols for all, color determines side)
PIECE_SYMBOLS = {
    chess.KING: '\u265A',    # ♚
    chess.QUEEN: '\u265B',   # ♛
    chess.ROOK: '\u265C',    # ♜
    chess.BISHOP: '\u265D',  # ♝
    chess.KNIGHT: '\u265E',  # ♞
    chess.PAWN: '\u265F',    # ♟
}

def render_board(board: chess.Board) -> Image.Image:
    """Render chess board as 256x256 RGB image."""
    img = Image.new('RGB', (BOARD_SIZE, BOARD_SIZE))
    draw = ImageDraw.Draw(img)

    # Try to load a font with good Unicode chess symbol support
    font = None
    font_paths = [
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/noto/NotoSansSymbols2-Regular.ttf",
        "/usr/share/fonts/TTF/NotoSansSymbols2-Regular.ttf",
    ]
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, 26)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    for rank in range(8):
        for file in range(8):
            # Calculate pixel coordinates (rank 0 = bottom, but we draw from top)
            display_rank = 7 - rank
            x_start = file * SQUARE_SIZE
            y_start = display_rank * SQUARE_SIZE

            # Determine square color
            is_light = (rank + file) % 2 == 1
            square_color = LIGHT_SQUARE if is_light else DARK_SQUARE

            # Fill the square
            draw.rectangle(
                [x_start, y_start, x_start + SQUARE_SIZE - 1, y_start + SQUARE_SIZE - 1],
                fill=square_color
            )

            # Check for piece on this square
            square = chess.square(file, rank)
            piece = board.piece_at(square)

            if piece:
                center_x = x_start + SQUARE_SIZE // 2
                center_y = y_start + SQUARE_SIZE // 2

                # Use filled symbol, color determines side
                symbol = PIECE_SYMBOLS[piece.piece_type]
                piece_color = WHITE_PIECE if piece.color == chess.WHITE else BLACK_PIECE

                # Draw the piece symbol centered
                bbox = draw.textbbox((0, 0), symbol, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                # Use anchor for proper centering
                text_x = center_x - text_w // 2 - bbox[0]
                text_y = center_y - text_h // 2 - bbox[1]
                draw.text((text_x, text_y), symbol, fill=piece_color, font=font)

    return img


if __name__ == "__main__":
    img = render_board(board)
    img.save("board_move1.png", "PNG")
    print(f"Board after 1. d4:")
    print(board)
    print(f"\nSaved to board_move1.png ({img.size[0]}x{img.size[1]}, mode: {img.mode})")
