"""Quick smoke test for v2 changes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_dynamics import DynamicsTransformer, TransitionsTokenDataset
import torch
import torch.nn.functional as F

model = DynamicsTransformer(d_model=384, d_ff=1536)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params / 1e6:.1f}M")

x = torch.randint(0, 1000, (2, 1279))
logits = model(x)
print(f"Input: {x.shape} -> Output: {logits.shape}")

ds = TransitionsTokenDataset(
    "datasets/lichess_1k/transitions_tokens.parquet",
    "datasets/lichess_1k",
    "datasets/lichess_1k/train.txt",
)
ctx, tgt = ds[0]
ctx, tgt = ctx.unsqueeze(0), tgt.unsqueeze(0)

inp = torch.cat([ctx, tgt[:, :-1]], dim=1)
logits = model(inp)
pred_logits = logits[:, 1023:, :]

last_ctx_frame = ctx[:, -256:]
changed = (tgt != last_ctx_frame).float()
weight = 1.0 + 4.0 * changed
n_changed = changed.sum().item()
frac = n_changed / tgt.numel()
print(f"Changed tokens: {int(n_changed)}/{tgt.numel()} ({frac:.3f})")

loss_per_tok = F.cross_entropy(
    pred_logits.reshape(-1, 1000), tgt.reshape(-1),
    reduction="none", label_smoothing=0.1,
)
loss = (loss_per_tok * weight.reshape(-1)).mean()
print(f"Weighted loss: {loss.item():.4f}")

preds = pred_logits.argmax(-1)
correct = (preds == tgt).float()
changed_mask = (tgt != last_ctx_frame)
acc_chg = correct[changed_mask].mean().item() if changed_mask.sum() > 0 else float("nan")
acc_unch = correct[~changed_mask].mean().item()
print(f"acc_chg: {acc_chg:.4f} | acc_unch: {acc_unch:.4f}")
print("All checks passed")
