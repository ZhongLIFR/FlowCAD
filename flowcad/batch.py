"""Fixed-budget Flow-CAD for scoring a fitted table or a later batch."""

from numbers import Integral

import numpy as np
import torch

from .checkpoints import N_DRAWS
from .detector import FlowCAD
from .fixed import ScoringDraws, fit_fixed


def score_with_contributions(draws, model):
    """Compute the benchmark row and feature reductions in one field pass."""
    scores = np.zeros(draws.n_rows, dtype=np.float64)
    contributions = np.zeros((draws.n_rows, draws.n_features), dtype=np.float64)
    for rows, clipped in draws.clipped_residuals(model):
        # Keep the original per-draw float32 sum before accumulation.
        np.add.at(scores, rows, clipped.sum(dim=1).numpy())
        np.add.at(contributions, rows, clipped.numpy())
    return scores / N_DRAWS, contributions / N_DRAWS


class FixedFlowCAD:
    """Fit the CPU benchmark configuration and reuse its MAD on later batches."""

    def __init__(self, seed=201):
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
            raise ValueError("Training seed must be an integer in [0, 2**32)")
        self.seed = int(seed)

    def fit(self, context, behavior):
        """Fit for 1,000 epochs on aligned, preprocessed feature matrices."""
        context, behavior = FlowCAD._paired_inputs(context, behavior)
        median = np.median(behavior, axis=0)
        scale = np.maximum(np.median(np.abs(behavior - median), axis=0), 1e-6)
        model = fit_fixed(context, behavior, self.seed)
        self.model_ = model
        self.context_dim_ = context.shape[1]
        self.behavior_dim_ = behavior.shape[1]
        self.mad_scale_ = scale
        return self

    def score_samples(self, context, behavior, seed=None, return_contributions=False):
        """Score a batch in the fitted feature representation using the fitted scales."""
        if not hasattr(self, "model_"):
            raise ValueError("Call fit before score_samples")
        context, behavior = FlowCAD._paired_inputs(context, behavior)
        if context.shape[1] != self.context_dim_ or behavior.shape[1] != self.behavior_dim_:
            raise ValueError("Scoring feature dimensions must match the fitted matrices")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)
                                 or not 0 <= seed < 2**32):
            raise ValueError("Scoring seed must be an integer in [0, 2**32)")
        scoring_seed = self.seed if seed is None else int(seed)
        draws = ScoringDraws(np.ascontiguousarray(context), np.ascontiguousarray(behavior),
                             scoring_seed)
        draws.scale = torch.as_tensor(self.mad_scale_, dtype=torch.float32).view(1, -1)
        if return_contributions:
            return score_with_contributions(draws, self.model_)
        return draws.scores(self.model_)
