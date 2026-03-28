"""
RL post-training for ChessNGen dynamics model.

Uses REINFORCE with an EMA baseline to optimize move legality,
with supervised CE loss as regularization to prevent catastrophic forgetting.

Two-pass trick for efficiency:
  Pass 1 (no grad): generate 256 tokens autoregressively, compute reward
  Pass 2 (with grad): single teacher-forcing forward pass to get log-probs

Usage:
  python train_rl.py
  python train_rl.py --dynamics-ckpt weights/dynamics_v2.1.pt --lr 1e-5 --lambda-sft 1.0
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from aim import Run as AimRun

from train_tokenizer import FSQVAE, lr_at_step
from train_dynamics import DynamicsTransformer, TransitionsTokenDataset, generate
from train_board_recognizer import BoardRecognizer, grid_to_fen
from infer import load_vae, decode_tokens_to_pil
from eval_legality import load_recognizer, recognize_pil, is_legal_transition


# -------------------------------------------------------------------
# Reward baseline
# -------------------------------------------------------------------
class RewardBaseline:
    """Exponential moving average of rewards for variance reduction."""

    def __init__(self, decay: float = 0.99):
        self.value = 0.0
        self.decay = decay
        self.initialized = False

    def update(self, rewards: torch.Tensor):
        batch_mean = rewards.mean().item()
        if not self.initialized:
            self.value = batch_mean
            self.initialized = True
        else:
            self.value = self.decay * self.value + (1 - self.decay) * batch_mean

    def get(self) -> float:
        return self.value


# -------------------------------------------------------------------
# Core RL functions
# -------------------------------------------------------------------
def load_dynamics_for_training(
    ckpt_path: str, device: torch.device, dropout: float = 0.1,
) -> DynamicsTransformer:
    """Load dynamics model with dropout enabled for training."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = DynamicsTransformer(
        vocab_size=args["vocab_size"],
        d_model=args["d_model"],
        n_heads=args["n_heads"],
        n_layers=args["n_layers"],
        d_ff=args["d_ff"],
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model


def compute_log_probs(
    model: DynamicsTransformer,
    context: torch.Tensor,
    sampled_tokens: torch.Tensor,
    temperature: float = 0.8,
) -> torch.Tensor:
    """
    Single forward pass to get log P(sampled_token_t | context, sampled_{<t}).

    context:        (B, 1024) token indices
    sampled_tokens: (B, 256)  tokens from generate(), detached
    temperature:    sampling temperature used during generation

    Returns: (B, 256) log-probabilities
    """
    # Teacher-forcing input: [context, sampled_0, ..., sampled_254]
    inp = torch.cat([context, sampled_tokens[:, :-1]], dim=1)  # (B, 1279)
    logits = model(inp)  # (B, 1279, vocab)

    # Positions 1023..1278 predict target tokens 0..255
    pred_logits = logits[:, 1023:, :] / temperature  # (B, 256, vocab)

    log_probs_all = F.log_softmax(pred_logits, dim=-1)  # (B, 256, vocab)
    log_probs = log_probs_all.gather(2, sampled_tokens.unsqueeze(2)).squeeze(2)
    return log_probs  # (B, 256)


def compute_reward(
    vae: FSQVAE,
    recognizer: BoardRecognizer,
    context_tokens: torch.Tensor,
    generated_tokens: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """
    Compute shaped reward for each sample in batch.

    Rewards: legal=+1.0, identical=-0.1, illegal=-0.5, pieces_vanished=-1.0

    Returns: (rewards (B,), stats dict)
    """
    B = generated_tokens.shape[0]
    rewards = torch.zeros(B, device=device)
    stats = {"legal": 0, "identical": 0, "illegal": 0, "pieces_vanished": 0}

    for i in range(B):
        last_ctx = context_tokens[i, -256:]
        gen = generated_tokens[i]

        img_before = decode_tokens_to_pil(vae, last_ctx, device)
        img_after = decode_tokens_to_pil(vae, gen, device)

        fen_before = recognize_pil(recognizer, img_before, device)
        fen_after = recognize_pil(recognizer, img_after, device)

        result = is_legal_transition(fen_before, fen_after)

        if result["legal"]:
            rewards[i] = 1.0
            stats["legal"] += 1
        elif result["identical"]:
            rewards[i] = -0.1
            stats["identical"] += 1
        else:
            if result["pieces_vanished"]:
                rewards[i] = -1.0
                stats["pieces_vanished"] += 1
            else:
                rewards[i] = -0.5
            stats["illegal"] += 1

    return rewards, stats


def compute_supervised_loss(
    model: DynamicsTransformer,
    context: torch.Tensor,
    target: torch.Tensor,
    change_weight: float,
    label_smoothing: float,
    vocab_size: int,
) -> torch.Tensor:
    """Change-weighted CE loss (same as train_dynamics.py)."""
    inp = torch.cat([context, target[:, :-1]], dim=1)  # (B, 1279)
    logits = model(inp)  # (B, 1279, vocab)
    pred_logits = logits[:, 1023:, :]  # (B, 256, vocab)

    last_ctx_frame = context[:, -256:]
    changed = (target != last_ctx_frame).float()
    weight = 1.0 + (change_weight - 1.0) * changed

    loss_per_tok = F.cross_entropy(
        pred_logits.reshape(-1, vocab_size),
        target.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    )
    return (loss_per_tok * weight.reshape(-1)).mean()


def evaluate_legality(
    model: DynamicsTransformer,
    vae: FSQVAE,
    recognizer: BoardRecognizer,
    dataset: TransitionsTokenDataset,
    device: torch.device,
    n_samples: int = 50,
) -> dict:
    """Evaluate legality rate on a sample of the dataset (greedy decoding)."""
    model.eval()
    n = min(n_samples, len(dataset))
    indices = np.random.choice(len(dataset), size=n, replace=False)

    stats = {"legal": 0, "identical": 0, "illegal": 0, "pieces_vanished": 0, "total": n}

    with torch.no_grad():
        for idx in indices:
            context, _ = dataset[idx]
            context = context.to(device)

            pred_tokens = generate(
                model, context.unsqueeze(0), n_tokens=256, temperature=0.0,
            )[0]

            img_before = decode_tokens_to_pil(vae, context[-256:], device)
            img_after = decode_tokens_to_pil(vae, pred_tokens, device)

            fen_before = recognize_pil(recognizer, img_before, device)
            fen_after = recognize_pil(recognizer, img_after, device)

            result = is_legal_transition(fen_before, fen_after)

            if result["legal"]:
                stats["legal"] += 1
            elif result["identical"]:
                stats["identical"] += 1
            else:
                stats["illegal"] += 1
                if result["pieces_vanished"]:
                    stats["pieces_vanished"] += 1

    return stats


# -------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------
def train_rl(args):
    device = torch.device(args.device)
    use_amp = (device.type == "cuda") and not args.no_amp
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {device}")
    print("Loading models...")

    # Frozen models
    vae = load_vae(args.vae_ckpt, device)
    recognizer = load_recognizer(args.recognizer_ckpt, device)

    # Trainable dynamics model (with dropout)
    dynamics = load_dynamics_for_training(args.dynamics_ckpt, device, dropout=args.dropout)
    n_params = sum(p.numel() for p in dynamics.parameters())
    print(f"Dynamics: {n_params / 1e6:.1f}M params")

    # Datasets
    print("Loading datasets...")
    data_root = Path(args.data)
    parquet = str(data_root / "transitions_tokens.parquet")
    train_ds = TransitionsTokenDataset(parquet, str(data_root), str(data_root / "train.txt"))
    val_ds = TransitionsTokenDataset(parquet, str(data_root), str(data_root / "val.txt"))

    rl_loader = DataLoader(
        train_ds, batch_size=args.rl_batch_size,
        shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
    )
    sft_loader = DataLoader(
        train_ds, batch_size=args.sft_batch_size,
        shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
    )

    # Optimizer
    opt = torch.optim.AdamW(
        dynamics.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-2,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    baseline = RewardBaseline(decay=args.baseline_decay)

    # Aim tracking
    aim_run = AimRun(repo="runs", experiment=out.name)
    aim_run["hparams"] = vars(args)

    # Infinite iterators
    def inf_iter(loader):
        while True:
            yield from loader

    rl_iter = inf_iter(rl_loader)
    sft_iter = inf_iter(sft_loader)

    # Baseline legality eval before training
    print("Evaluating baseline legality...")
    baseline_stats = evaluate_legality(dynamics, vae, recognizer, val_ds, device, n_samples=50)
    baseline_rate = baseline_stats["legal"] / baseline_stats["total"]
    print(f"  Baseline legality: {100 * baseline_rate:.1f}%")
    aim_run.track(baseline_rate, name="legality_rate", step=0, context={"subset": "val"})

    print(f"\nTraining for {args.steps} steps "
          f"(B_rl={args.rl_batch_size}, B_sft={args.sft_batch_size}, "
          f"accum={args.accum_steps})")
    print(f"  Effective RL batch: {args.rl_batch_size * args.accum_steps}")
    print(f"  lambda_sft={args.lambda_sft}, temp={args.temperature}, top_k={args.top_k}\n")

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)

        # LR schedule
        lr = lr_at_step(step, args.steps, args.warmup, args.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr

        total_rl_loss = 0.0
        total_sft_loss = 0.0
        total_reward = 0.0
        total_stats = {"legal": 0, "identical": 0, "illegal": 0, "pieces_vanished": 0}
        total_rl_samples = 0

        for _ in range(args.accum_steps):
            # --- RL batch ---
            rl_context, _ = next(rl_iter)
            rl_context = rl_context.to(device, non_blocking=True)

            # Pass 1: generate (no grad)
            dynamics.eval()
            with torch.no_grad():
                sampled = generate(
                    dynamics, rl_context, n_tokens=256,
                    temperature=args.temperature, top_k=args.top_k,
                )  # (B_rl, 256)

                rewards, stats = compute_reward(
                    vae, recognizer, rl_context, sampled, device,
                )

            # Pass 2: log-probs (with grad)
            dynamics.train()
            with torch.amp.autocast("cuda", enabled=use_amp):
                log_probs = compute_log_probs(
                    dynamics, rl_context, sampled.detach(), args.temperature,
                )  # (B_rl, 256)

                advantages = (rewards - baseline.get()).detach()
                seq_log_probs = log_probs.sum(dim=1)  # (B_rl,)
                rl_loss = -(advantages * seq_log_probs).mean()

                # --- SFT batch ---
                sft_context, sft_target = next(sft_iter)
                sft_context = sft_context.to(device, non_blocking=True)
                sft_target = sft_target.to(device, non_blocking=True)

                sft_loss = compute_supervised_loss(
                    dynamics, sft_context, sft_target,
                    args.change_weight, args.label_smoothing, args.vocab_size,
                )

                loss = (rl_loss + args.lambda_sft * sft_loss) / args.accum_steps

            scaler.scale(loss).backward()

            baseline.update(rewards)

            # Accumulate stats
            total_rl_loss += rl_loss.item()
            total_sft_loss += sft_loss.item()
            total_reward += rewards.sum().item()
            total_rl_samples += rewards.shape[0]
            for k in total_stats:
                total_stats[k] += stats[k]

        # Gradient step
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(dynamics.parameters(), args.grad_clip).item()
        else:
            grad_norm = 0.0
        scaler.step(opt)
        scaler.update()

        # Logging
        avg_rl_loss = total_rl_loss / args.accum_steps
        avg_sft_loss = total_sft_loss / args.accum_steps
        avg_reward = total_reward / total_rl_samples
        legality_rate = total_stats["legal"] / total_rl_samples
        identical_rate = total_stats["identical"] / total_rl_samples

        if step % args.log_every == 0 or step == 1:
            print(
                f"step {step:5d} | rl {avg_rl_loss:+.4f} | sft {avg_sft_loss:.4f} | "
                f"R {avg_reward:+.3f} | legal {100*legality_rate:.0f}% | "
                f"ident {100*identical_rate:.0f}% | gnorm {grad_norm:.2f} | lr {lr:.1e}"
            )

        aim_run.track(avg_rl_loss, name="rl_loss", step=step, context={"subset": "train"})
        aim_run.track(avg_sft_loss, name="sft_loss", step=step, context={"subset": "train"})
        aim_run.track(avg_reward, name="reward_mean", step=step, context={"subset": "train"})
        aim_run.track(legality_rate, name="legality_rate", step=step, context={"subset": "train"})
        aim_run.track(identical_rate, name="identical_rate", step=step, context={"subset": "train"})
        aim_run.track(baseline.get(), name="baseline", step=step)
        aim_run.track(grad_norm, name="grad_norm", step=step)
        aim_run.track(lr, name="lr", step=step)

        # Periodic validation
        if step % args.eval_every == 0:
            eval_stats = evaluate_legality(dynamics, vae, recognizer, val_ds, device, n_samples=50)
            val_legal = eval_stats["legal"] / eval_stats["total"]
            val_identical = eval_stats["identical"] / eval_stats["total"]
            print(f"  [val] legality {100*val_legal:.1f}% | identical {100*val_identical:.1f}%")
            aim_run.track(val_legal, name="legality_rate", step=step, context={"subset": "val"})
            aim_run.track(val_identical, name="identical_rate", step=step, context={"subset": "val"})

        # Checkpoint
        if step % args.save_every == 0:
            ckpt = {
                "step": step,
                "model": dynamics.state_dict(),
                "opt": opt.state_dict(),
                "baseline": baseline.value,
                "args": vars(args),
            }
            torch.save(ckpt, out / f"ckpt_{step:06d}.pt")
            print(f"  [saved] {out / f'ckpt_{step:06d}.pt'}")

    # Final save
    torch.save(
        {
            "step": args.steps,
            "model": dynamics.state_dict(),
            "opt": opt.state_dict(),
            "baseline": baseline.value,
            "args": vars(args),
        },
        out / "ckpt_final.pt",
    )

    # Final eval
    final_stats = evaluate_legality(dynamics, vae, recognizer, val_ds, device, n_samples=100)
    final_rate = final_stats["legal"] / final_stats["total"]
    print(f"\nDone. {args.steps} steps, output in {out}")
    print(f"  Baseline legality: {100 * baseline_rate:.1f}%")
    print(f"  Final legality:    {100 * final_rate:.1f}%")

    aim_run.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RL post-training for dynamics model")
    p.add_argument("--dynamics-ckpt", default="weights/dynamics_v2.1.pt")
    p.add_argument("--vae-ckpt", default="weights/tokenizer_best.pt")
    p.add_argument("--recognizer-ckpt", default="weights/board_recognizer.pt")
    p.add_argument("--data", default="datasets/lichess_1k")
    p.add_argument("--out-dir", default="runs/dynamics_rl_v1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--rl-batch-size", type=int, default=8)
    p.add_argument("--sft-batch-size", type=int, default=32)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--lambda-sft", type=float, default=1.0)
    p.add_argument("--change-weight", type=float, default=5.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--vocab-size", type=int, default=1000)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--baseline-decay", type=float, default=0.99)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_rl(args)
