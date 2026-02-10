"""
FastAPI server for chess world-model self-play inference.

Usage:
  pixi run python playground/app.py
"""

import argparse
import json
import sys
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

app = FastAPI()

dynamics_model = None
vae_model = None
device = None
opening_moves = None  # list of move strings

# Server-side session state
context = None  # (1024,) tensor on device


class InitRequest(BaseModel):
    mode: str = "opening"


@app.post("/api/init")
def init_game(req: InitRequest):
    """Bootstrap context and return initial frames + opening info."""
    global context

    if req.mode == "startpos":
        context = bootstrap_startpos(device)
        moves = ""
    else:
        context, idx = bootstrap_random_opening(device)
        moves = opening_moves[idx]

    k = context.shape[0] // 256
    frames = []
    for i in range(k):
        frame_tokens = context[i * 256 : (i + 1) * 256]
        frames.append(pil_to_base64(decode_tokens_to_pil(vae_model, frame_tokens, device)))

    return {"frames": frames, "opening": moves}


@app.get("/api/step")
def step_game(temperature: float = 0.0, top_k: int = 0):
    """Generate one frame with SSE streaming of partial renders."""
    global context

    if context is None:
        raise HTTPException(400, "No active session. Call /api/init first.")

    def event_stream():
        global context
        final_tokens = None

        for result, is_final in generate_streaming(
            dynamics_model, vae_model, context, device,
            temperature=temperature, top_k=top_k,
        ):
            if is_final is None:
                # This is the token tensor for context update
                final_tokens = result
                continue

            b64 = pil_to_base64(result)
            event = "done" if is_final else "partial"
            yield f"event: {event}\ndata: {json.dumps({'frame': b64})}\n\n"

        if final_tokens is not None:
            context = torch.cat([context[256:], final_tokens])

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def main():
    global dynamics_model, vae_model, device, opening_moves

    parser = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--dynamics-ckpt", default=str(project_root / "weights" / "dynamics_best.pt"))
    parser.add_argument("--vae-ckpt", default=str(project_root / "weights" / "tokenizer_best.pt"))
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading models on {device}...")
    vae_model = load_vae(args.vae_ckpt, device)
    dynamics_model = load_dynamics(args.dynamics_ckpt, device)
    _, opening_moves = load_openings()
    print(f"Models loaded. {len(opening_moves)} openings available. Starting server...")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
