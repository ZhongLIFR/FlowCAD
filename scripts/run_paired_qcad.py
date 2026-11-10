"""Evaluate the QCAD-compatible mixed-type Gower port on matched tables."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from replay import MANIFEST, REPLAY, ROOT, read_manifest

sys.path.insert(0, str(ROOT))
from flowcad.metrics import evaluate_scores

QUANTILES = np.r_[np.arange(101), 0.001]
FOREST_SIZE = 10
FOREST_SEED = 0
MIN_SAMPLES_SPLIT = 10
METRICS = ("pr_auc", "roc_auc", "p_at_n")
TREE_SEEDS = tuple(int(seed) for seed in np.random.RandomState(FOREST_SEED).randint(
    np.iinfo(np.int32).max, size=FOREST_SIZE
))
FAST_FOREST = sklearn.__version__ == "1.5.2"


def _forest_leaf_weights(x, y, x_new):
    forest = RandomForestRegressor(n_estimators=FOREST_SIZE, random_state=FOREST_SEED,
                                   min_samples_split=MIN_SAMPLES_SPLIT,
                                   max_features=x.shape[1], n_jobs=1).fit(x, y)
    leaves = forest.apply(x)
    query_leaves = forest.apply(x_new[None, :])[0]
    weights = np.zeros(len(y), dtype=np.float32)
    for i, tree in enumerate(forest.estimators_):
        boot = np.random.RandomState(tree.random_state).randint(len(y), size=len(y))
        in_leaf = boot[leaves[boot, i] == query_leaves[i]]
        weights += (np.bincount(in_leaf, minlength=len(y)) / len(in_leaf)).astype(np.float32)
    return weights


@lru_cache(maxsize=32)
def _bootstrap_weights(n_samples):
    draws = []
    for seed in TREE_SEEDS:
        fit_samples = np.random.RandomState(seed).randint(
            0, n_samples, size=n_samples, dtype=np.int32
        )
        cdf_samples = np.random.RandomState(seed).randint(n_samples, size=n_samples)
        fit_weights = np.bincount(fit_samples, minlength=n_samples).astype(np.float64)
        cdf_counts = np.bincount(cdf_samples, minlength=n_samples)
        draws.append((seed, fit_weights, cdf_counts))
    return draws


def _fast_leaf_weights(x, y, x_new):
    # sklearn 1.5.2 fits the bootstrap trees on the full matrix with multiplicity weights.
    targets = np.ascontiguousarray(y.reshape(-1, 1), dtype=np.float64)
    queries = np.concatenate((x, x_new[None, :]))
    weights = np.zeros(len(y), dtype=np.float32)
    for seed, fit_weights, cdf_counts in _bootstrap_weights(len(y)):
        tree = DecisionTreeRegressor(random_state=seed, min_samples_split=MIN_SAMPLES_SPLIT,
                                     max_features=x.shape[1])
        tree._fit(x, targets, sample_weight=fit_weights, check_input=False)
        leaves = tree.apply(queries, check_input=False)
        in_leaf = cdf_counts * (leaves[:-1] == leaves[-1])
        weights += (in_leaf / in_leaf.sum()).astype(np.float32)
    return weights


def quantiles(x, y, x_new):
    """Compute weighted leaf quantiles with the configured sklearn forest."""
    if (FAST_FOREST and x.dtype == np.float32 and x_new.dtype == np.float32
            and y.dtype.kind in "bifu" and np.isfinite(x).all()
            and np.isfinite(x_new).all() and np.isfinite(y).all()):
        weights = _fast_leaf_weights(x, y, x_new)
    else:
        weights = _forest_leaf_weights(x, y, x_new)
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


def neighbor_indices(context, categories, n_neighbors, block_rows=256):
    """Compute Gower neighborhoods in row blocks with the original tie handling."""
    n, p = context.shape
    columns = []
    for column in context:
        if column in categories:
            columns.append((context[column].astype(str).to_numpy(), None))
        else:
            values = context[column].to_numpy(dtype=np.float32)
            columns.append((values, np.ptp(values)))
    neighbors = np.empty((n, n_neighbors), dtype=np.intp)
    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        distance = np.zeros((stop - start, n), dtype=np.float32)
        for values, span in columns:
            query = values[start:stop, None]
            if span is None:
                distance += query != values[None, :]
            elif span:
                distance += np.abs(query - values[None, :]) / span
        distance /= p
        # Match the dense sklearn 1.5.2 precomputed-distance reducer row by row.
        nearest = np.argpartition(distance, n_neighbors - 1, axis=1)[:, :n_neighbors]
        rows = np.arange(stop - start)[:, None]
        neighbors[start:stop] = nearest[rows, np.argsort(distance[rows, nearest])]
    return neighbors


def _compare_metrics(result, reference, target):
    if reference is None:
        return
    fields = ("dataset", "injection_seed", "mode", "sha256", "rows", "k", "sklearn",
              "lower_tail_zero_iqr_floors", "n", *METRICS)
    changed = [field for field in fields if result.get(field) != reference.get(field)]
    if changed:
        raise ValueError(f"Archived QCAD result differs for {target.name}: {', '.join(changed)}")


def _score_metadata(result):
    fields = ("dataset", "injection_seed", "sha256", "mode", "k", "sklearn",
              "lower_tail_zero_iqr_floors", *METRICS)
    return {**{field: result[field] for field in fields},
            "n_rows": result["rows"], "n_anomalies": result["n"]}


def run_one(task, *, score_output=None, compare_against=None):
    key, seed, path, spec, config, mode, outdir = task
    suffix = f'_seed_{seed}' if seed is not None else ''
    target = outdir / f'{mode}_{key}{suffix}.json'
    score_path = None if score_output is None else score_output / f'{key}{suffix}.npz'
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha256 != spec['sha256']:
        raise ValueError(f'Frozen input changed: {key}')
    n_neighbors = spec.get('benchmark_k', min(spec['n_rows'] // 2, 500))
    reference = None if compare_against is None else json.loads(
        (compare_against / target.name).read_text()
    )
    if target.exists():
        prior = json.loads(target.read_text())
        expected = dict(dataset=key, mode=mode, sha256=sha256, rows=spec['n_rows'],
                        k=n_neighbors, sklearn=sklearn.__version__)
        if seed is not None:
            expected['injection_seed'] = seed
        if any(prior.get(field) != value for field, value in expected.items()):
            raise ValueError(f'Result does not match the requested case: {target}')
        _compare_metrics(prior, reference, target)
        if score_path is None:
            return prior
        if score_path.exists():
            with np.load(score_path, allow_pickle=False) as record:
                metadata = _score_metadata(prior)
                scores = record["scores"]
                if (any(record[field].item() != value for field, value in metadata.items())
                        or scores.shape != (spec["n_rows"],) or not np.isfinite(scores).all()):
                    raise ValueError(f"Saved scores differ from the QCAD record: {score_path}")
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
    neighbors = neighbor_indices(table[ctx], categories, n_neighbors)
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
    _compare_metrics(result, reference, target)
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2) + '\n')
    tmp.replace(target)
    if score_path is not None:
        score_tmp = score_path.with_suffix(".tmp")
        with score_tmp.open("wb") as stream:
            np.savez_compressed(stream, scores=scores, **_score_metadata(result))
        score_tmp.replace(score_path)
    return result


def _eligible_cases(path, manifest):
    expected = {(key, int(seed)): entry["sha256"] for key, spec in manifest.items()
                for seed, entry in spec["files"].items()}
    cases = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["scope"] != "in_range":
                continue
            case = row["dataset"], int(row["injection_seed"])
            if (case in cases or expected.get(case) != row["sha256"]
                    or row["defined"] not in ("0", "1")):
                raise ValueError(f"Invalid in-range case in {path}: {case}")
            cases[case] = row["defined"] == "1"
    if cases.keys() != expected.keys():
        raise ValueError(f"In-range case list does not cover the frozen replay: {path}")
    return {case for case, eligible in cases.items() if eligible}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="number of smallest datasets to run (default: 8, or all with --all-injections)")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=REPLAY / "paired_qcad")
    parser.add_argument("--score-output", type=Path, help="Save raw scores in input row order as NPZ files")
    parser.add_argument("--compare-against", type=Path, help="Verify full-table metrics against an archived QCAD JSON directory")
    parser.add_argument("--in-range-cases", type=Path, help="Use only eligible cases from the conditional-mean per-run CSV")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    seeds.add_argument("--all-injections", action="store_true", help="run every manifest injection for each selected dataset")
    parser.add_argument("--datasets", nargs="+", help="select dataset keys instead of the smallest available")
    args = parser.parse_args()
    if args.jobs < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--jobs and --limit must be positive")
    manifest = read_manifest(args.manifest)["datasets"]
    limit = args.limit or (len(manifest) if args.all_injections else 8)
    chosen = args.datasets or sorted(manifest, key=lambda k: (manifest[k]["n_rows"], k))[:limit]
    if len(set(chosen)) != len(chosen) or set(chosen) - manifest.keys():
        parser.error("--datasets must contain unique, known dataset keys")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    eligible = None if args.in_range_cases is None else _eligible_cases(args.in_range_cases, manifest)
    jobs = []
    for key in chosen:
        spec = manifest[key]
        injection_seeds = spec["injection_seeds"] if args.all_injections else args.seeds
        for seed in injection_seeds:
            if str(seed) not in spec["files"]:
                parser.error(f"seed {seed} is unavailable for {key}")
            if eligible is not None and (key, seed) not in eligible:
                continue
            entry = spec["files"][str(seed)]
            jobs.append((key, seed, args.manifest.parent / entry["path"],
                         {**spec, **entry}, spec, "paper", args.output))
    if args.all_injections:
        jobs.sort(key=lambda job: -job[3]["n_rows"] * len(job[4]["behavior_columns"]))
    if not jobs:
        parser.error("No eligible cases match the requested datasets and seeds")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.score_output is not None:
        args.score_output.mkdir(parents=True, exist_ok=True)
    records = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        tasks = [pool.submit(run_one, job, score_output=args.score_output,
                             compare_against=args.compare_against) for job in jobs]
        for future in as_completed(tasks):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
    records.sort(key=lambda record: (record["dataset"], record["injection_seed"]))
    report = args.output / "per_run.csv"
    temporary = report.with_suffix(".tmp")
    fields = ("dataset", "injection_seed", "sha256", "mode", "rows", "k", "sklearn",
              "seconds", "lower_tail_zero_iqr_floors", "pr_auc", "roc_auc", "p_at_n", "n")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(report)


if __name__ == "__main__":
    main()
