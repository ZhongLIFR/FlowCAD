"""Score the LoPAD and ROCOD implementations released with the QCAD benchmark."""

from functools import lru_cache
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


QCAD_COMMIT = "f3b887f3d4466607a9d7147e9715f6b0b27a5954"
BLANKETS_DIR = "Data/GenData/MB"
LOPAD_REPEATS = 20
ROCOD_DISTANCE = 0.9
ROCOD_BLOCK_SIZE = 256


def _qcad_git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True)


@lru_cache(maxsize=4)
def _blanket_files(repo):
    paths = _qcad_git(repo, "ls-tree", "--name-only", f"{QCAD_COMMIT}:{BLANKETS_DIR}").splitlines()
    return {name.casefold(): name for name in paths if name.endswith(".csv")}


@lru_cache(maxsize=20)
def _markov_blankets(repo, raw_path, columns):
    basename = Path(raw_path).stem + "Gene.csv"
    filename = _blanket_files(repo)[basename.casefold()]
    lines = _qcad_git(repo, "show", f"{QCAD_COMMIT}:{BLANKETS_DIR}/{filename}").splitlines()
    if len(lines) != len(columns):
        raise ValueError(f"Markov blankets do not align with benchmark columns: {filename}")
    positions = {column: index for index, column in enumerate(columns)}
    return tuple(tuple(positions[name.replace(".", " ")] for name in line.split(";") if name) for line in lines)


def _score_lopad(features, spec, seed, qcad_repo):
    columns = tuple(spec["benchmark_columns"])
    predictors = _markov_blankets(str(Path(qcad_repo).resolve()), spec["raw_path"], columns)
    values = features.loc[:, list(columns)].to_numpy(dtype=np.float64)
    totals = np.zeros(len(values), dtype=np.float64)
    visits = np.zeros(len(values), dtype=np.int64)
    tree = DecisionTreeRegressor(random_state=np.random.RandomState(seed))
    for sample_seed in range(LOPAD_REPEATS):
        # The released code uses 20 random 80% subsets without replacement.
        # The 200 repetitions reported in the paper differ from this release.
        indices = np.random.RandomState(sample_seed).choice(len(values), round(0.8 * len(values)), replace=False)
        sample = values[indices]
        scores = np.zeros(len(indices), dtype=np.float64)
        for target, blanket in enumerate(predictors):
            if blanket:
                design, observed = sample[:, blanket], sample[:, target]
                predicted = tree.fit(design, observed).predict(design)
                residual = np.abs(observed - predicted)
                scores += np.maximum((residual - residual.mean()) / max(residual.std(), 1e-4), 0.0)
        totals[indices] += scores
        visits[indices] += 1
    if np.any(visits == 0):
        raise ValueError("The released LoPAD subsamples left an observation unscored")
    return totals / visits


def _local_predictions(context_frame, behavior):
    # ROCOD_Basic casts all context columns to strings for Gower. With
    # equal weights and categorical columns only, distance is mismatch rate.
    context = np.column_stack([pd.factorize(context_frame[column].astype(str))[0] for column in context_frame])
    unique, inverse, counts = np.unique(context, axis=0, return_inverse=True, return_counts=True)
    dimension = context.shape[1]
    max_mismatches = int(np.floor(ROCOD_DISTANCE * dimension))
    group_totals = np.zeros((len(unique), behavior.shape[1] + 1), dtype=np.float64)
    group_totals[:, 0] = counts
    np.add.at(group_totals[:, 1:], inverse, behavior)
    nearby_totals = np.empty_like(group_totals)
    for start in range(0, len(unique), ROCOD_BLOCK_SIZE):
        stop = min(start + ROCOD_BLOCK_SIZE, len(unique))
        block = unique[start:stop]
        mismatches = np.zeros((len(block), len(unique)), dtype=np.int32)
        for column in range(dimension):
            mismatches += block[:, column, None] != unique[None, :, column]
        nearby_totals[start:stop] = (mismatches <= max_mismatches) @ group_totals
    row_totals = nearby_totals[inverse]
    neighbor_counts = row_totals[:, 0] - 1
    # Isolated rows use the global model: their local weight is zero. Use
    # a finite placeholder so 0 * NaN cannot poison all feature weights.
    local = np.divide(row_totals[:, 1:] - behavior, neighbor_counts[:, None],
                      out=np.zeros_like(behavior), where=neighbor_counts[:, None] > 0)
    return neighbor_counts, local


def _score_rocod(features, spec):
    context_frame = features.loc[:, spec["context_columns"]]
    context = context_frame.to_numpy(dtype=np.float64)
    behavior = features.loc[:, spec["behavior_columns"]].to_numpy(dtype=np.float64)
    counts, local = _local_predictions(context_frame, behavior)
    predicted = np.column_stack([LinearRegression().fit(context, target).predict(context) for target in behavior.T])
    local_weight = np.sqrt(counts / max(counts.max(), 1))[:, None]
    squared_error = np.square(behavior - (local_weight * local + (1 - local_weight) * predicted))
    total_variation = np.square(behavior - behavior.mean(axis=0)).sum(axis=0)
    ratio = np.divide(squared_error.sum(axis=0), total_variation, out=np.ones_like(total_variation),
                      where=total_variation > 0)
    return squared_error @ np.maximum(1 - ratio, 0.0)


def score_context_baseline(method, features, spec, seed, qcad_repo):
    """Return one unlabeled raw anomaly score per input row, in input order."""
    if method == "LoPAD":
        return _score_lopad(features, spec, seed, qcad_repo)
    if method == "ROCOD":
        return _score_rocod(features, spec)
    raise ValueError(f"Unknown contextual baseline: {method}")
