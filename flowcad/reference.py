"""Conditional flow matching and nested Monte Carlo residual scoring."""

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

def set_seed(seed: int=42):
    """Seed Python, NumPy and PyTorch; configure deterministic cuDNN execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ContextBehaviorDataset(Dataset):
    """Aligned float32 context (N, P) and behavior (N, Q) arrays."""

    def __init__(self, context_array: np.ndarray, behavior_array: np.ndarray):
        """Store arrays with matching row counts as float32."""
        assert context_array.shape[0] == behavior_array.shape[0]
        self.context = context_array.astype(np.float32)
        self.behavior = behavior_array.astype(np.float32)

    def __len__(self):
        return self.context.shape[0]

    def __getitem__(self, idx: int):
        c = self.context[idx]
        b = self.behavior[idx]
        return (c, b)

class ConditionalVectorField(nn.Module):
    """Conditional velocity model with a two-layer ReLU context encoder.

    Inputs are x_t (batch, behavior_dim), context (batch, context_dim),
    and time (batch, 1). The ReLU backbone predicts behavioral velocity.
    """

    def __init__(self, context_dim: int, behavior_dim: int, hidden_dim: int=64, context_emb_dim: int=32, num_layers: int=3):
        super().__init__()
        self.context_dim = context_dim
        self.behavior_dim = behavior_dim
        self.context_encoder = nn.Sequential(nn.Linear(context_dim, context_emb_dim), nn.ReLU(), nn.Linear(context_emb_dim, context_emb_dim), nn.ReLU())
        input_dim = behavior_dim + context_emb_dim + 1
        layers = []
        dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, behavior_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return predicted velocity with shape (batch, behavior_dim)."""
        c_emb = self.context_encoder(c)
        inp = torch.cat([x_t, c_emb, t], dim=1)
        v_pred = self.net(inp)
        return v_pred

def train_flow_matching(model: nn.Module, train_loader: DataLoader, n_epochs: int=100, lr: float=0.001, device: str='cpu', verbose: bool=True):
    """Train for n_epochs with Adam and squared velocity-matching loss.

    For each batch, sample Gaussian noise z and uniform times t, construct
    x_t = (1 - t) * z + t * behavior, and match velocity behavior - z.
    Returns the trained model on the requested device.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for batch_c, batch_b in train_loader:
            batch_c = batch_c.to(device)
            batch_b = batch_b.to(device)
            batch_size, behavior_dim = batch_b.shape
            z = torch.randn_like(batch_b)
            t = torch.rand(batch_size, 1, device=device)
            x_t = (1.0 - t) * z + t * batch_b
            v_t = batch_b - z
            v_pred = model(x_t, batch_c, t)
            loss = torch.mean((v_pred - v_t) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if verbose and epoch % max(1, n_epochs // 10) == 0:
            print(f'[Flow Matching] Epoch {epoch}/{n_epochs}, Loss = {epoch_loss / max(1, n_batches):.6f}')
    return model

def compute_flow_cad_scores(model: nn.Module, context_array: np.ndarray, behavior_array: np.ndarray, device: str='cpu', n_z_samples: int=4, n_t_samples: int=4, scale_vec: np.ndarray=None, tau: float=10.0) -> np.ndarray:
    """Return one residual energy per observation; larger values are more anomalous.

    For each of n_z_samples Gaussian vectors, draw n_t_samples fresh uniform
    times while reusing that vector. At each path point, divide absolute
    velocity residuals by scale_vec (Q,) when supplied, clip each dimension
    at tau, sum over behavioral dimensions, and average over all draws.
    Context and behavior have shapes (N, P) and (N, Q); output has shape (N,).
    """
    model.eval()
    model = model.to(device)
    N, P = context_array.shape
    _, Q = behavior_array.shape
    context_tensor = torch.tensor(context_array, dtype=torch.float32, device=device)
    behavior_tensor = torch.tensor(behavior_array, dtype=torch.float32, device=device)
    if scale_vec is not None:
        scale_tensor = torch.tensor(scale_vec, dtype=torch.float32, device=device).view(1, -1)
    else:
        scale_tensor = None
    all_scores = []
    with torch.no_grad():
        for i in range(N):
            c_i = context_tensor[i:i + 1, :]
            b_i = behavior_tensor[i:i + 1, :]
            residuals = []
            for _ in range(n_z_samples):
                z = torch.randn_like(b_i)
                v_true = b_i - z
                for _ in range(n_t_samples):
                    t = torch.rand(1, 1, device=device)
                    x_t = (1.0 - t) * z + t * b_i
                    v_pred = model(x_t, c_i, t)
                    diff = torch.abs(v_pred - v_true)
                    if scale_tensor is not None:
                        diff_norm = diff / scale_tensor
                    else:
                        diff_norm = diff
                    diff_clip = torch.clamp(diff_norm, max=tau)
                    res = torch.sum(diff_clip, dim=1)
                    residuals.append(res.item())
            score_i = float(np.mean(residuals))
            all_scores.append(score_i)
    return np.array(all_scores, dtype=np.float64)
