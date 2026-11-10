"""Full-table, label-free checkpoint selection for the paper benchmark."""

import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from .reference import ConditionalVectorField, ContextBehaviorDataset, set_seed


CANDIDATE_EPOCHS = (50, 100, 200, 500, 1000, 2000, 5000, 10000)
TAIL_FRACTION = 0.05
N_Z = N_T = 16
N_DRAWS = N_Z * N_T
CHUNK_SIZE = 4096


class _BatchedDataset(ContextBehaviorDataset):
    def __getitems__(self, indices):
        return torch.from_numpy(self.context[indices]), torch.from_numpy(self.behavior[indices])


class _SharedDraws:
    def __init__(self, context, behavior, seed):
        self.context = torch.as_tensor(context, dtype=torch.float32)
        self.behavior = torch.as_tensor(behavior, dtype=torch.float32)
        self.n_rows, self.n_features = self.behavior.shape
        median = np.median(behavior, axis=0)
        mad = np.median(np.abs(behavior - median), axis=0)
        self.scale = torch.as_tensor(np.maximum(mad, 1e-6), dtype=torch.float32).view(1, -1)
        generator = torch.Generator(device="cpu").manual_seed(seed + 1000003)
        self.noise = torch.empty((self.n_rows, N_Z, self.n_features), dtype=torch.float32)
        self.time = torch.empty((self.n_rows, N_Z, N_T), dtype=torch.float32)
        # Preserve the reference draw order: row, Gaussian vector, time block.
        for row in range(self.n_rows):
            for z in range(N_Z):
                self.noise[row, z] = torch.randn((1, self.n_features), generator=generator)[0]
                self.time[row, z] = torch.rand((N_T, 1), generator=generator)[:, 0]

    def clipped_residuals(self, model):
        model.eval()
        with torch.no_grad():
            for start in range(0, self.n_rows * N_DRAWS, CHUNK_SIZE):
                stop = min(start + CHUNK_SIZE, self.n_rows * N_DRAWS)
                flat = torch.arange(start, stop)
                rows = torch.div(flat, N_DRAWS, rounding_mode="floor")
                noise_index = torch.div(flat, N_T, rounding_mode="floor") % N_Z
                t = self.time.reshape(-1)[start:stop].view(-1, 1)
                z, b = self.noise[rows, noise_index], self.behavior[rows]
                xt, target = (1.0 - t) * z + t * b, b - z
                predicted = model(xt, self.context[rows], t)
                clipped = torch.clamp(torch.abs(predicted - target) / self.scale, max=10.0)
                yield rows.numpy(), clipped

    def scores(self, model):
        totals = np.zeros(self.n_rows, dtype=np.float64)
        for rows, clipped in self.clipped_residuals(model):
            np.add.at(totals, rows, clipped.sum(dim=1).numpy())
        return totals / N_DRAWS

    def contributions(self, model):
        totals = np.zeros((self.n_rows, self.n_features), dtype=np.float64)
        for rows, clipped in self.clipped_residuals(model):
            np.add.at(totals, rows, clipped.numpy())
        return totals / N_DRAWS


def _train_epoch(model, loader, optimizer):
    model.train()
    losses = []
    for context, behavior in loader:
        z = torch.randn_like(behavior)
        t = torch.rand(len(behavior), 1, device="cpu")
        xt, target = (1.0 - t) * z + t * behavior, behavior - z
        loss = torch.mean((model(xt, context, t) - target) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if not math.isfinite(float(np.mean(losses))):
        raise ValueError("Non-finite Flow Matching training loss")


def _contrast_score_margin(scores, k):
    if not np.isfinite(scores).all():
        raise ValueError("CSM requires finite anomaly scores")
    split = len(scores) - k
    partition = np.partition(scores, split)
    rest, tail = partition[:split], partition[split:]
    sd = math.sqrt(float(tail.var(ddof=1) + rest.var(ddof=1)))
    if not math.isfinite(sd) or sd == 0:
        return None
    margin = float((tail.mean() - rest.mean()) / sd)
    return margin if math.isfinite(margin) else None


def select_checkpoint(context, behavior, seed):
    """Train on CPU and maximize 5% CSM over the paper's checkpoint grid.

    Set PyTorch to one CPU thread for the paper benchmark. Inputs are aligned
    feature matrices, with no labels. The same scoring draws are used across
    all candidates, and ties favor the earlier checkpoint. The winning raw
    scores and feature contributions use those same draws.
    """
    if len(behavior) < 4:
        raise ValueError("CSM requires at least four rows")
    k = min(len(behavior) - 2, max(2, math.floor(TAIL_FRACTION * len(behavior))))
    set_seed(seed)
    loader = DataLoader(_BatchedDataset(context, behavior), batch_size=128, shuffle=True,
                        drop_last=False, collate_fn=lambda batch: batch)
    model = ConditionalVectorField(context.shape[1], behavior.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    draws = _SharedDraws(context, behavior, seed)

    candidates = []
    best = None
    for epoch in range(1, CANDIDATE_EPOCHS[-1] + 1):
        _train_epoch(model, loader, optimizer)
        if epoch not in CANDIDATE_EPOCHS:
            continue
        scores = draws.scores(model)
        csm = _contrast_score_margin(scores, k)
        candidates.append({"epoch": epoch, "csm": csm})
        if csm is not None and (best is None or csm > best["csm"]):
            best = {"epoch": epoch, "csm": csm, "scores": scores}
            best_state = {name: param.clone() for name, param in model.state_dict().items()}
    if best is None:
        raise ValueError("All candidate checkpoints have undefined CSM")

    model.load_state_dict(best_state)
    return {**best, "k": k, "contributions": draws.contributions(model), "candidates": candidates}
