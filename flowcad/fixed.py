"""Fixed-budget training for the paired benchmark."""

import torch
from torch.utils.data import DataLoader

from .checkpoints import _BatchedDataset, _SharedDraws as ScoringDraws, _train_epoch
from .reference import ConditionalVectorField, set_seed


DEFAULT_EPOCHS = 1000


def fit_fixed(context, behavior, seed, epochs=DEFAULT_EPOCHS):
    """Fit the benchmark velocity field on CPU for a predetermined budget."""
    if epochs < 1:
        raise ValueError("The training budget must be positive")
    set_seed(seed)
    loader = DataLoader(_BatchedDataset(context, behavior), batch_size=128, shuffle=True,
                        drop_last=False, collate_fn=lambda batch: batch)
    model = ConditionalVectorField(context.shape[1], behavior.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(epochs):
        _train_epoch(model, loader, optimizer)
    return model
