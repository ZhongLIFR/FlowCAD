"""Load configured benchmark features and optional evaluation labels."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BenchmarkData:
    context: np.ndarray
    behavior: np.ndarray
    labels: Optional[np.ndarray]
    frame: pd.DataFrame
    context_feature_names: list
    behavior_feature_names: list
    dropped_rows: int
    sha256: str


def load_configs(path=None):
    path = Path(path) if path is not None else PROJECT_ROOT / "configs" / "datasets.json"
    with path.open(encoding="utf-8") as source:
        return json.load(source)["datasets"]


def load_benchmark(config, data_dir=None, require_labels=True):
    """Load only configured features; never infer features from other columns.

    Prepared numerical values are used as stored. Categorical context columns
    are one-hot encoded. If applying the model to new data, fit one shared
    encoder on the training context and retain its columns for future scoring.
    """
    data_dir = Path(data_dir) if data_dir is not None else PROJECT_ROOT / "data" / "GenData"
    path = data_dir / config["file"]
    context_names = list(config["context_columns"])
    behavior_names = list(config["behavior_columns"])
    categories = list(config.get("categorical_columns", []))
    if not context_names or not behavior_names:
        raise ValueError("Both context_columns and behavior_columns must be nonempty")
    features = context_names + behavior_names
    if len(set(features)) != len(features) or "ground_truth" in features:
        raise ValueError("Feature partitions must be unique, disjoint, and exclude ground_truth")
    if not set(categories).issubset(context_names):
        raise ValueError("categorical_columns must be a subset of context_columns")
    selected = features + (["ground_truth"] if require_labels else [])
    frame = pd.read_csv(path, usecols=lambda name: name in selected)
    missing = sorted(set(selected) - set(frame.columns))
    if missing:
        raise ValueError(f"{config['file']}: missing required columns: {', '.join(missing)}")
    before = len(frame)
    frame = frame.loc[:, selected].dropna(subset=features).copy()
    if frame.empty:
        raise ValueError(f"{config['file']}: no rows remain after selected-feature missingness handling")
    context = pd.get_dummies(frame[context_names], columns=categories, dtype=np.float32)
    try:
        context_array = context.to_numpy(dtype=np.float32)
        behavior_array = frame[behavior_names].to_numpy(dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("Numerical features must be numeric; declare context categories explicitly") from error
    if not np.isfinite(context_array).all() or not np.isfinite(behavior_array).all():
        raise ValueError(f"{config['file']}: selected features contain non-finite values")
    labels = frame["ground_truth"].to_numpy() if require_labels else None
    if labels is not None and not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{config['file']}: ground_truth must contain only 0 and 1")
    return BenchmarkData(context_array, behavior_array, labels, frame,
                         context.columns.tolist(), behavior_names, before - len(frame),
                         hashlib.sha256(path.read_bytes()).hexdigest())
