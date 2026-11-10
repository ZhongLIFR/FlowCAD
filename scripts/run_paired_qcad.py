"""Evaluate the QCAD-compatible mixed-type Gower port on matched tables."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors

from replay import MANIFEST, REPLAY, ROOT, read_manifest

sys.path.insert(0, str(ROOT))
from flowcad.metrics import evaluate_scores

QUANTILES = np.r_[np.arange(101), 0.001]


def quantiles(x, y, x_new):
    """skgarden-style weighted leaf CDF using sklearn's maintained RF backend."""
    forest = RandomForestRegressor(n_estimators=10, random_state=0, min_samples_split=10,
                                   max_features=x.shape[1], n_jobs=1).fit(x, y)
    leaves = forest.apply(x)
    query_leaves = forest.apply(x_new[None, :])[0]
    weights = np.zeros(len(y), dtype=np.float32)
    for i, tree in enumerate(forest.estimators_):
        boot = np.random.RandomState(tree.random_state).randint(len(y), size=len(y))
        in_leaf = boot[leaves[boot, i] == query_leaves[i]]
        weights += (np.bincount(in_leaf, minlength=len(y)) / len(in_leaf)).astype(np.float32)
    sorted_indices = np.argsort(y)
    order = sorted_indices[weights[sorted_indices] > 0]
    weights = weights[order]
    acc = np.cumsum(weights)
    midpoints = (100.0 / acc[-1]) * (acc - weights / 2.0)
    return np.interp(QUANTILES, midpoints, y.astype(np.float32)[order])


def contribution(q, observed):
    # Preserve released QCAD's last matching interval and location-1 lookup.
    widths = np.diff(q[:100])
    bins = np.flatnonzero((q[:100] <= observed) & (observed <= q[1:101]))
    if len(bins):
        return 1000 * widths[bins[-1] - 1], False
    spread = q[75] - q[25]
    endpoint = q[-1] if observed < q[-1] else q[100]
    # The released lower-tail branch lacks the 1e-9 IQR floor of the upper-tail branch.
    repaired = observed < q[-1] and spread == 0
    denom = spread if observed < q[-1] and spread > 0 else max(1e-9, spread)
    return 1000 * np.max(widths) * (1 + abs(observed - endpoint) / denom), repaired


def distance_matrix(context, categories):
    n, p = context.shape
    distance = np.zeros((n, n), dtype=np.float32)
    for column in context:
        if column in categories:
            values = context[column].astype(str).to_numpy()
            distance += values[:, None] != values[None, :]
        else:
            values = context[column].to_numpy(dtype=np.float32)
            span = np.ptp(values)
            if span:
                distance += np.abs(values[:, None] - values[None, :]) / span
    return distance / p


def run_one(task):
    key, seed, path, spec, config, mode, outdir = task
    suffix = f'_seed_{seed}' if seed is not None else ''
    target = outdir / f'{mode}_{key}{suffix}.json'
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha256 != spec['sha256']:
        raise ValueError(f'Frozen input changed: {key}')
    if target.exists():
        prior = json.loads(target.read_text())
        if prior['sha256'] != sha256:
            raise ValueError(f'Result already exists for a different input: {target}')
        return prior
    start = perf_counter()
    ctx, behavior = config['context_columns'], config['behavior_columns']
    categories = set(config['categorical_columns'])
    table = pd.read_csv(path, usecols=ctx + behavior)
    if len(table) != spec['n_rows'] or table.isna().any().any():
        raise ValueError(f'Unexpected rows or missing features: {key}')
    x = np.column_stack([pd.Categorical(table[column]).codes if column in categories
                         else pd.to_numeric(table[column]) for column in ctx]).astype(np.float32)
    y = table[behavior].to_numpy()
    n_neighbors = spec.get('benchmark_k', min(len(table) // 2, 500))
    distance = distance_matrix(table[ctx], categories)
    neighbors = NearestNeighbors(algorithm='brute', metric='precomputed',
                                 n_neighbors=n_neighbors, n_jobs=1).fit(distance).kneighbors(distance, return_distance=False)
    raw = np.empty(y.shape, dtype=float)
    repaired = 0
    for i, indices in enumerate(neighbors):
        indices = np.sort(indices[indices != i])
        for j in range(len(behavior)):
            raw[i, j], used_floor = contribution(quantiles(x[indices], y[indices, j], x[i]), y[i, j])
            repaired += used_floor
    if len(behavior) > 1:
        raw = np.minimum(raw, 100)
        totals = []
        for col in raw.T:
            bins = np.linspace(0, col.max(), 101)
            totals.append(np.sum((bins + 1) * ((col[:, None] > bins) & (col[:, None] < bins + 1)).sum(axis=0)))
        totals = np.array(totals)
        if totals.max() > 0:
            raw *= 2 - totals / totals.max()
    scores = raw.mean(axis=1)
    # Parse labels strictly after the unsupervised detector has scored every row.
    labels = pd.read_csv(path, usecols=['ground_truth'])['ground_truth'].to_numpy()
    if int(labels.sum()) != spec['n_anomalies']:
        raise ValueError(f'Unexpected count of injected anomalies: {key}')
    result = dict(dataset=key, mode=mode, sha256=sha256, rows=len(table), k=n_neighbors,
                  lower_tail_zero_iqr_floors=int(repaired), sklearn=sklearn.__version__,
                  seconds=perf_counter() - start, **evaluate_scores(labels, scores))
    if seed is not None:
        result['injection_seed'] = seed
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2) + '\n')
    tmp.replace(target)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=8, help="number of smallest datasets to run")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=REPLAY / "paired_qcad")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--datasets", nargs="+", help="select dataset keys instead of the smallest available")
    args = parser.parse_args()
    manifest = read_manifest(args.manifest)["datasets"]
    chosen = args.datasets or sorted(manifest, key=lambda k: (manifest[k]["n_rows"], k))[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for key in chosen:
        spec = manifest[key]
        for seed in args.seeds:
            entry = spec["files"][str(seed)]
            jobs.append((key, seed, args.manifest.parent / entry["path"],
                         {**spec, **entry}, spec, "paper", args.output))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(run_one, job) for job in jobs]):
            print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    main()
