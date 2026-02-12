"""
FastAPI server for chess world-model self-play inference.

Usage:
  pixi run python playground/app.py
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from infer import (
    load_dynamics,
    load_vae,
    load_openings,
    bootstrap_startpos,
    bootstrap_random_opening,
    decode_tokens_to_pil,
    generate_streaming,
    pil_to_base64,
)
from train_board_recognizer import BoardRecognizer, grid_to_fen
from torchvision import transforms

app = FastAPI()


@dataclass
class ModelEntry:
    name: str
    dynamics_path: str
    vae_path: str
    dynamics_model: object = None
    vae_model: object = None


@dataclass
class ServerState:
    device: torch.device | None = None
    default_vae_model: object = None
    models: dict[str, ModelEntry] = field(default_factory=dict)
    opening_moves: list[str] = field(default_factory=list)
    recognizer: BoardRecognizer | None = None
    # Session
    active_model_id: str | None = None
    context: torch.Tensor | None = None


state = ServerState()

_to_tensor = transforms.ToTensor()


def recognize_pil(img) -> str:
    """Run board recognizer on a PIL image, return FEN piece-placement string."""
    x = _to_tensor(img.convert("RGB")).unsqueeze(0).to(state.device)
    with torch.no_grad():
        logits = state.recognizer(x)
        preds = logits.argmax(dim=1).squeeze(0)
    return grid_to_fen(preds)


class InitRequest(BaseModel):
    mode: str = "opening"
    model_id: str | None = None


@app.get("/api/models")
def list_models():
    """Return available dynamics models."""
    models = [
        {"id": model_id, "name": entry.name}
        for model_id, entry in state.models.items()
    ]
    return {"models": models, "active": state.active_model_id}


@app.post("/api/init")
def init_game(req: InitRequest):
    """Bootstrap context and return initial frames + opening info."""
    # Resolve model
    model_id = req.model_id
    if model_id is None:
        model_id = state.active_model_id or next(iter(state.models))
    if model_id not in state.models:
        raise HTTPException(400, f"Unknown model: {model_id}. Available: {list(state.models.keys())}")

    entry = state.models[model_id]
    state.active_model_id = model_id
    vae = entry.vae_model

    if req.mode == "startpos":
        state.context = bootstrap_startpos(state.device)
        moves = ""
    else:
        state.context, idx = bootstrap_random_opening(state.device)
        moves = state.opening_moves[idx]

    k = state.context.shape[0] // 256
    frames = []
    fens = []
    for i in range(k):
        frame_tokens = state.context[i * 256 : (i + 1) * 256]
        pil_img = decode_tokens_to_pil(vae, frame_tokens, state.device)
        frames.append(pil_to_base64(pil_img))
        fens.append(recognize_pil(pil_img))

    return {"frames": frames, "fens": fens, "opening": moves, "model_id": model_id}


@app.get("/api/step")
def step_game(temperature: float = 0.0, top_k: int = 0):
    """Generate one frame with SSE streaming of partial renders."""
    if state.context is None or state.active_model_id is None:
        raise HTTPException(400, "No active session. Call /api/init first.")

    entry = state.models[state.active_model_id]
    dynamics = entry.dynamics_model
    vae = entry.vae_model

    def event_stream():
        final_tokens = None

        for result, is_final in generate_streaming(
            dynamics, vae, state.context, state.device,
            temperature=temperature, top_k=top_k,
        ):
            if is_final is None:
                final_tokens = result
                continue

            b64 = pil_to_base64(result)
            if is_final:
                fen = recognize_pil(result)
                yield f"event: done\ndata: {json.dumps({'frame': b64, 'fen': fen})}\n\n"
            else:
                yield f"event: partial\ndata: {json.dumps({'frame': b64})}\n\n"

        if final_tokens is not None:
            state.context = torch.cat([state.context[256:], final_tokens])

    return StreamingResponse(event_stream(), media_type="text/event-stream")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
W = PROJECT_ROOT / "weights"

# ── Model registry ──────────────────────────────────────────────────
# (id, display name, dynamics checkpoint, vae checkpoint)
MODEL_LIST: list[tuple[str, str, Path, Path]] = [
    ("v1",   "v1",   W / "dynamics_best.pt",  W / "tokenizer_best.pt"),
    ("v2",   "v2",   W / "dynamics_v2.pt",    W / "tokenizer_best.pt"),
    ("v2.1", "v2.1", W / "dynamics_v2.1.pt",  W / "tokenizer_best.pt"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    state.device = torch.device(args.device)
    print(f"Loading models on {state.device}...")

    # Load models, sharing VAEs when paths match
    loaded_vaes: dict[Path, object] = {}
    for model_id, name, dyn_path, vae_path in MODEL_LIST:
        if not dyn_path.exists():
            print(f"  Skipping '{name}': {dyn_path} not found")
            continue
        print(f"  Loading '{name}' from {dyn_path}...")
        if vae_path not in loaded_vaes:
            loaded_vaes[vae_path] = load_vae(str(vae_path), state.device)
        state.models[model_id] = ModelEntry(
            name=name,
            dynamics_path=str(dyn_path),
            vae_path=str(vae_path),
            dynamics_model=load_dynamics(str(dyn_path), state.device),
            vae_model=loaded_vaes[vae_path],
        )

    if not state.models:
        print("ERROR: No models found.")
        sys.exit(1)

    # Load board recognizer
    recognizer_path = W / "board_recognizer.pt"
    if recognizer_path.exists():
        print(f"  Loading board recognizer from {recognizer_path}...")
        state.recognizer = BoardRecognizer().to(state.device)
        ckpt = torch.load(str(recognizer_path), map_location=state.device, weights_only=True)
        state.recognizer.load_state_dict(ckpt["model"])
        state.recognizer.eval()
    else:
        print(f"  WARNING: Board recognizer not found at {recognizer_path}")

    # Load openings
    _, state.opening_moves = load_openings()

    print(f"Loaded {len(state.models)} model(s): {list(state.models.keys())}")
    print(f"{len(state.opening_moves)} openings available. Starting server...")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
