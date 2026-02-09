"""
Core inference module for chess world-model self-play.

Loads the dynamics transformer and FSQ-VAE, bootstraps an initial context
from a starting position or PGN, and runs the autoregressive self-play loop.
"""

import io
import base64

import chess
import chess.pgn
import torch
from PIL import Image
from torchvision import transforms

from generate_dataset import render_board
from train_tokenizer import FSQVAE
from train_dynamics import DynamicsTransformer, generate


def load_vae(ckpt_path: str, device: torch.device) -> FSQVAE:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FSQVAE(levels=ckpt["levels"], base_ch=ckpt["base_ch"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_dynamics(ckpt_path: str, device: torch.device) -> DynamicsTransformer:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = DynamicsTransformer(
        vocab_size=args["vocab_size"],
        d_model=args["d_model"],
        n_heads=args["n_heads"],
        n_layers=args["n_layers"],
        d_ff=args["d_ff"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


_to_tensor = transforms.ToTensor()


def encode_board_image(vae: FSQVAE, img: Image.Image, device: torch.device) -> torch.Tensor:
    """Encode a 256x256 PIL image to flat token indices (256,)."""
    t = _to_tensor(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        _, indices = vae.encode(t)  # (1, 16, 16)
    return indices[0].reshape(-1).long()  # (256,)


def decode_tokens_to_pil(vae: FSQVAE, flat_tokens: torch.Tensor, device: torch.device) -> Image.Image:
    """Decode flat token indices (256,) to a PIL image."""
    tok = flat_tokens.view(1, 16, 16).long().to(device)
    with torch.no_grad():
        z_q = vae.fsq.indices_to_codes(tok)
        img = vae.decode(z_q)  # (1, 3, 256, 256)
    img = img.squeeze(0).clamp(0, 1).cpu()
    return transforms.ToPILImage()(img)


def bootstrap_startpos(vae: FSQVAE, device: torch.device, k: int = 4) -> torch.Tensor:
    """Encode the standard chess starting position, repeated k times → (k*256,)."""
    board = chess.Board()
    img = render_board(board)
    tokens = encode_board_image(vae, img, device)  # (256,)
    return tokens.repeat(k)  # (k*256,)


def bootstrap_pgn(vae: FSQVAE, pgn_str: str, device: torch.device, k: int = 4) -> torch.Tensor:
    """Parse PGN, take first k board positions, encode each → (k*256,)."""
    game = chess.pgn.read_game(io.StringIO(pgn_str))
    if game is None:
        raise ValueError("Could not parse PGN")

    board = game.board()
    positions = [board.copy()]
    for move in game.mainline_moves():
        board.push(move)
        positions.append(board.copy())
        if len(positions) >= k:
            break

    while len(positions) < k:
        positions.append(positions[-1].copy())

    token_list = []
    for pos in positions[:k]:
        img = render_board(pos)
        token_list.append(encode_board_image(vae, img, device))

    return torch.cat(token_list)  # (k*256,)


def self_play(
    model: DynamicsTransformer,
    vae: FSQVAE,
    context: torch.Tensor,
    n_steps: int,
    device: torch.device,
    temperature: float = 0.0,
    top_k: int = 0,
) -> list[Image.Image]:
    """
    Run self-play: repeatedly predict the next frame and slide the context window.

    Returns: list of PIL images (k context frames + n_steps generated frames).
    """
    import time

    k = context.shape[0] // 256

    print(f"Decoding {k} context frames...")
    frames = []
    for i in range(k):
        frame_tokens = context[i * 256 : (i + 1) * 256]
        frames.append(decode_tokens_to_pil(vae, frame_tokens, device))

    print(f"Generating {n_steps} steps (temp={temperature}, top_k={top_k})...")
    t0 = time.time()
    for step in range(n_steps):
        pred_tokens = generate(
            model,
            context.unsqueeze(0),
            n_tokens=256,
            temperature=temperature,
            top_k=top_k,
        )
        pred_tokens = pred_tokens[0]  # (256,)

        frames.append(decode_tokens_to_pil(vae, pred_tokens, device))
        context = torch.cat([context[256:], pred_tokens])

        elapsed = time.time() - t0
        print(f"  step {step + 1}/{n_steps} ({elapsed:.1f}s)")

    print(f"Done in {time.time() - t0:.1f}s")
    return frames


def generate_streaming(
    model: DynamicsTransformer,
    vae: FSQVAE,
    context: torch.Tensor,
    device: torch.device,
    temperature: float = 0.0,
    top_k: int = 0,
    row_size: int = 16,
):
    """
    Generate 256 tokens one at a time, yielding a partial PIL image
    after every completed row (16 tokens).
    """
    import torch.nn.functional as F

    model.eval()
    generated = []
    seq = context.unsqueeze(0)  # (1, 1024)

    for i in range(256):
        with torch.no_grad():
            logits = model(seq)[:, -1, :]

        if temperature <= 0:
            tok = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                v, _ = logits.topk(top_k, dim=-1)
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            tok = torch.multinomial(probs, 1)

        generated.append(tok.item())
        seq = torch.cat([seq, tok], dim=1)

        # After each completed row, decode partial frame
        if (i + 1) % row_size == 0:
            partial = torch.zeros(256, dtype=torch.long, device=device)
            partial[: len(generated)] = torch.tensor(generated, dtype=torch.long, device=device)
            yield decode_tokens_to_pil(vae, partial, device), False

    # Final complete frame
    final_tokens = torch.tensor(generated, dtype=torch.long, device=device)
    yield decode_tokens_to_pil(vae, final_tokens, device), True
    # Return the flat tokens so caller can update context
    yield final_tokens, None


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
