"""Run the point and contextual baselines on the matched benchmark tables."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter

from replay import MAIN_MANIFEST as MANIFEST, MAIN_REPLAY as REPLAY, ROOT, read_manifest


POINT = ("IForest", "LOF", "k-NN", "SOD", "HBOS")
CONTEXT = ("ROCOD", "LoPAD")
METHODS = POINT + CONTEXT
CAD_METHODS = ("CAD",)


def run_one(job):
    key, seed, spec, entry, source, output, methods, qcad_repo = job
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import pandas as pd
    from flowcad.metrics import evaluate_scores, percentile_ranks

    digest = sha256(source.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"Unexpected input hash: {key}/{seed}")
    columns = spec["benchmark_columns"]
    features = pd.read_csv(source, usecols=columns).loc[:, columns]
    if len(features) != spec["n_rows"] or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError(f"Unexpected feature values or row count: {key}/{seed}")
    training_seed = 201 + seed - 42
    scored = []
    feature_counts = dict.fromkeys(methods, len(columns))
    point_methods = set(methods) & set(POINT)
    if point_methods:
        from qcad_point_baselines import score_point_baselines

        started = perf_counter()
        for method, scores in score_point_baselines(features, training_seed, point_methods):
            now = perf_counter()
            scored.append((method, scores, now - started))
            started = now
    if set(methods) & set(CONTEXT):
        from qcad_context_baselines import score_context_baseline

        for method in methods:
            if method in CONTEXT:
                started = perf_counter()
                scores = score_context_baseline(method, features, spec, training_seed, qcad_repo)
                scored.append((method, scores, perf_counter() - started))
    if "CAD" in methods:
        from song_cad import score_cad_full

        started = perf_counter()
        scored.append(("CAD", score_cad_full(features, spec), perf_counter() - started))
        feature_counts["CAD"] = len(spec["context_columns"]) + len(spec["behavior_columns"])
    if sorted(method for method, _, _ in scored) != sorted(methods):
        raise ValueError(f"Missing or unexpected detector output: {key}/{seed}")
    labels = pd.read_csv(source, usecols=["ground_truth"])["ground_truth"].to_numpy()
    if int(np.sum(labels == 1)) != spec["n_anomalies"]:
        raise ValueError(f"Unexpected anomaly count: {key}/{seed}")
    for method, scores, elapsed in scored:
        row = {"method": method, "dataset": key, "injection_seed": seed,
               "training_seed": 0 if method in CAD_METHODS else training_seed,
               "sha256": digest, "n_rows": len(features), "n_features": feature_counts[method],
               **evaluate_scores(labels, percentile_ranks(scores)), "fit_score_seconds": elapsed,
               "sklearn_version": version("scikit-learn"),
               "pyod_version": version("pyod") if method in POINT else None}
        target = output / method / f"{key}_seed_{seed}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
        temporary.replace(target)
    return key, seed, list(methods)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=REPLAY / "classical_baselines")
    parser.add_argument("--qcad-repo", type=Path, help="Pinned QCAD checkout supplying LoPAD blankets")
    parser.add_argument("--methods", nargs="+", choices=METHODS + CAD_METHODS, default=METHODS)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if "LoPAD" in args.methods and not args.qcad_repo:
        parser.error("--qcad-repo is required for LoPAD")
    if args.jobs < 1 or len(set(args.methods)) != len(args.methods):
        parser.error("Specify a positive worker count and distinct methods")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[variable] = "1"
    output = args.output.resolve()
    for method in args.methods:
        (output / method).mkdir(parents=True, exist_ok=True)
    datasets = read_manifest(args.manifest)["datasets"]
    chosen = args.datasets or sorted(datasets, key=lambda key: (datasets[key]["n_rows"], key))
    jobs = []
    for key in chosen:
        spec = datasets[key]
        for seed in spec["injection_seeds"]:
            if args.seeds and seed not in args.seeds:
                continue
            entry = spec["files"][str(seed)]
            pending = []
            for method in args.methods:
                target = output / method / f"{key}_seed_{seed}.json"
                if not target.exists():
                    pending.append(method)
                    continue
                row = json.loads(target.read_text())
                if (row["method"], row["dataset"], row["injection_seed"], row["sha256"]) != (
                    method, key, seed, entry["sha256"]
                ):
                    raise ValueError(f"Stale result: {target}")
            if pending:
                jobs.append((key, seed, spec, entry, args.manifest.parent / entry["path"],
                             output, pending, args.qcad_repo))
    print(f"{len(jobs)} pending table runs", flush=True)
    if args.jobs == 1:
        for job in jobs:
            print(*run_one(job), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for future in as_completed([pool.submit(run_one, job) for job in jobs]):
                print(*future.result(), flush=True)


if __name__ == "__main__":
    main()
