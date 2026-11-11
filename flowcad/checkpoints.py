"""Shared training and scoring utilities for the fixed-budget benchmark."""

import math

import numpy as np
import torch

from .reference import ContextBehaviorDataset


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


