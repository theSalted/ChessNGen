"""
Recognize chess board images and output FEN piece-placement strings.

Usage:
  python recognize_board.py --image path/to/board.png
  python recognize_board.py --dir path/to/images/
  python recognize_board.py --image img.png --weights weights/board_recognizer.pt
"""

import argparse
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from train_board_recognizer import BoardRecognizer, grid_to_fen


def load_model(weights_path: str, device: torch.device) -> BoardRecognizer:
    model = BoardRecognizer().to(device)
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def recognize(model: BoardRecognizer, image_path: str, device: torch.device) -> str:
    tf = transforms.ToTensor()
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)  # (1, 13, 8, 8)
        preds = logits.argmax(dim=1).squeeze(0)  # (8, 8)

    return grid_to_fen(preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, help="single image path")
    p.add_argument("--dir", type=str, help="directory of images")
    p.add_argument("--weights", default="weights/board_recognizer.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if not args.image and not args.dir:
        p.error("provide --image or --dir")

    device = torch.device(args.device)
    model = load_model(args.weights, device)

    paths = []
    if args.image:
        paths.append(Path(args.image))
    if args.dir:
        paths.extend(sorted(Path(args.dir).glob("*.png")))

    for path in paths:
        fen = recognize(model, str(path), device)
        print(f"{path.name}\t{fen}")


if __name__ == "__main__":
    main()
