"""Flow-CAD fitting, residual scoring, and behavioral contributions."""

import numbers
import numpy as np
import torch
from torch.utils.data import DataLoader

from .reference import (
    ConditionalVectorField, ContextBehaviorDataset, compute_flow_cad_scores,
    set_seed, train_flow_matching,
)


def _matrix(value, name):
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric two-dimensional matrix") from error
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{name} must have at least one row and one column")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains missing or non-finite values")
    return array


class FlowCAD:
    """Conditional flow matching with MAD scaling, truncation and L1 pooling.

    Inputs are numerical, preprocessed context and behavior matrices. Larger
    scores indicate more anomalous behavior under the supplied context. Labels
    are neither accepted nor used by fit/score. Score sampling is stochastic;
    pass ``seed`` to score_samples for a repeatable call on the same hardware.
    """

    def __init__(self, n_epochs=1000, batch_size=128, lr=1e-3, hidden_dim=64,
                 context_emb_dim=32, num_layers=3, n_z_samples=16,
                 n_t_samples=16, tau=10.0, random_state=42, device="cpu",
                 verbose=False):
        for name, value in {
            "n_epochs": n_epochs, "batch_size": batch_size,
            "hidden_dim": hidden_dim, "context_emb_dim": context_emb_dim,
            "num_layers": num_layers, "n_z_samples": n_z_samples,
            "n_t_samples": n_t_samples,
        }.items():
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, int(value))
        for name, value in {"lr": lr, "tau": tau}.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
            setattr(self, name, float(value))
        self.random_state = int(random_state)
        self.device = str(torch.device(device))
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use device='cpu'")
        self.verbose = bool(verbose)

    def fit(self, context, behavior):
        """Fit using every supplied row, without anomaly labels."""
        context, behavior = self._paired_inputs(context, behavior)
        set_seed(self.random_state)
        self.context_dim_ = context.shape[1]
        self.behavior_dim_ = behavior.shape[1]
        median = np.median(behavior, axis=0)
        self.mad_scale_ = np.maximum(
            np.median(np.abs(behavior - median), axis=0), 1e-6,
        ).astype(np.float32)
        loader = DataLoader(ContextBehaviorDataset(context, behavior),
                            batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.model_ = ConditionalVectorField(
            context_dim=self.context_dim_, behavior_dim=self.behavior_dim_,
            hidden_dim=self.hidden_dim, context_emb_dim=self.context_emb_dim,
            num_layers=self.num_layers,
        )
        self.model_ = train_flow_matching(
            self.model_, loader, n_epochs=self.n_epochs, lr=self.lr,
            device=self.device, verbose=self.verbose,
        )
        return self

    @staticmethod
    def _paired_inputs(context, behavior):
        context = _matrix(context, "context")
        behavior = _matrix(behavior, "behavior")
        if len(context) != len(behavior):
            raise ValueError("context and behavior must have the same number of rows")
        return context, behavior

    def score_samples(self, context, behavior, seed=None, return_contributions=False):
        """Return raw energy, optionally with additive behavioral contributions.

        Contributions have shape (n_rows, n_behavior_features) and sum to the
        raw score, up to floating-point rounding. They explain the score and
        are not causal effects. MAD scales come from fit, including when
        scoring a new batch. Percentile conversion is a separate operation.
        """
        if not hasattr(self, "model_"):
            raise ValueError("Call fit before score_samples")
        context, behavior = self._paired_inputs(context, behavior)
        if context.shape[1] != self.context_dim_ or behavior.shape[1] != self.behavior_dim_:
            raise ValueError("Scoring feature dimensions must match the fitted matrices")
        if seed is not None:
            set_seed(int(seed))
        if not return_contributions:
            return compute_flow_cad_scores(
                self.model_, context, behavior, device=self.device,
                n_z_samples=self.n_z_samples, n_t_samples=self.n_t_samples,
                scale_vec=self.mad_scale_, tau=self.tau,
            )
        # Reuse each noise vector across time draws, matching the raw scorer.
        self.model_.eval()
        contexts = torch.tensor(context, dtype=torch.float32, device=self.device)
        behaviors = torch.tensor(behavior, dtype=torch.float32, device=self.device)
        scale = torch.tensor(self.mad_scale_, device=self.device).view(1, -1)
        scores, contributions = [], []
        with torch.no_grad():
            for i in range(len(context)):
                c, b = contexts[i:i + 1], behaviors[i:i + 1]
                totals, features = [], []
                for _ in range(self.n_z_samples):
                    z = torch.randn_like(b)
                    target = b - z
                    for _ in range(self.n_t_samples):
                        t = torch.rand(1, 1, device=self.device)
                        xt = (1.0 - t) * z + t * b
                        residual = torch.abs(self.model_(xt, c, t) - target)
                        clipped = torch.clamp(residual / scale, max=self.tau)
                        totals.append(torch.sum(clipped, dim=1).item())
                        features.append(clipped[0].cpu().numpy())
                scores.append(float(np.mean(totals)))
                contributions.append(np.mean(np.asarray(features, dtype=np.float64), axis=0))
        return np.asarray(scores), np.asarray(contributions)
