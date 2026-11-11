"""Recover row-aligned Flow-CAD scores from the fixed 1,000-epoch benchmark."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import time

from replay import MANIFEST, MODEL_COMMIT, ROOT, read_manifest, verify_model

METRICS = ("pr_auc", "roc_auc", "p_at_n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def expected_metadata(record, reference_sha, run_sha):
    metadata = {key: record[key] for key in ("dataset", "injection_seed", "sha256", "training_seed",
                                             "n_rows", "n_anomalies", "selected_epoch", *METRICS)}
    return {**metadata, "flowcad_commit": MODEL_COMMIT,
            "reference_sha256": reference_sha, "run_sha256": run_sha}


def read_existing(target, metadata):
    import numpy as np

    with np.load(target, allow_pickle=False) as saved:
        if any(saved[key].item() != value for key, value in metadata.items()):
            raise ValueError(f"Existing score metadata differs from the replay inputs: {target}")
        scores = saved["scores"]
        if scores.shape != (metadata["n_rows"],) or scores.dtype != np.float64 or not np.isfinite(scores).all():
            raise ValueError(f"Existing record has invalid row scores: {target}")
        return {key: saved[key].item() for key in ("source", "replay_cpu_s", "replay_wall_s")}


def evaluate(job):
    spec, path, target, metadata, sweep_path = job
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import pandas as pd
    from flowcad.data import load_benchmark
    from flowcad.metrics import evaluate_scores

    started_wall, started_cpu = time.perf_counter(), time.process_time()
    data = load_benchmark({**spec, "file": path.name}, path.parent, require_labels=False)
    if data.sha256 != metadata["sha256"] or len(data.frame) != metadata["n_rows"] or data.dropped_rows:
        raise ValueError(f"Input does not match the frozen replay: {path}")

    if sweep_path is None:
        import torch
        from flowcad.fixed import ScoringDraws, fit_fixed

        torch.set_num_threads(1)
        seed = metadata["training_seed"]
        model = fit_fixed(data.context, data.behavior, seed)
        scores = ScoringDraws(data.context, data.behavior, seed).scores(model)
        source = "refit"
    else:
        with np.load(sweep_path, allow_pickle=False) as saved:
            if (saved["sha256"].item() != metadata["sha256"]
                    or not np.array_equal(saved["rows"], data.frame.index.to_numpy())):
                raise ValueError(f"Scoring sweep input or row order differs: {sweep_path}")
            scores = saved["scores"]
        source = "scoring_sweep"

    labels = pd.read_csv(path, usecols=["ground_truth"]).loc[data.frame.index, "ground_truth"].to_numpy()
    evaluation = evaluate_scores(labels, scores)
    if evaluation["n"] != metadata["n_anomalies"] or any(evaluation[key] != metadata[key] for key in METRICS):
        raise ValueError(f"Full-table metrics differ from the archived replay: {target.name}")

    timing = {"source": source, "replay_cpu_s": time.process_time() - started_cpu,
              "replay_wall_s": time.perf_counter() - started_wall}
    temp = target.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, scores=scores, **metadata, **timing)
    temp.replace(target)
    return timing


def write_summary(output, records):
    temp = output / "per_run.tmp"
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: (row["dataset"], row["injection_seed"])))
    temp.replace(output / "per_run.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--reference", type=Path, help="Directory of fixed-epoch Flow-CAD JSON records")
    parser.add_argument("--sweep-records", type=Path, help="Reuse the fixed scoring sweep's row-score files")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--datasets", default="all", help="all or comma-separated dataset keys")
    parser.add_argument("--injection-seeds", default="all", help="all or comma-separated injection seeds")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"

    verify_model()
    inputs = read_manifest(args.manifest)["datasets"]
    replay_dir = args.manifest.parent.parent
    reference_dir = args.reference or replay_dir / "flowcad_fixed_1000"
    protocol = {
        "manifest_sha256": digest(args.manifest.read_bytes()),
        "runner_sha256": digest(Path(__file__).read_bytes()), "flowcad_commit": MODEL_COMMIT,
        "versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch")},
        "protocol": "Recover the row-aligned raw scores from the fixed 1,000-epoch fit. "
                    "Verify exact full-table metrics against the main replay.",
    }
    run_sha = digest(json.dumps(protocol, sort_keys=True).encode())
    output = (args.output or replay_dir / "qcad_in_range_flow_fixed_1000").resolve()
    (output / "records").mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_bytes())["run_sha256"] != run_sha:
        raise ValueError(f"Output directory uses a different score replay protocol: {output}")
    write_json(protocol_path, {"run_sha256": run_sha, **protocol})

    datasets = list(inputs) if args.datasets == "all" else args.datasets.split(",")
    jobs, records = [], []
    for dataset in datasets:
        spec = inputs[dataset]
        seeds = (spec["injection_seeds"] if args.injection_seeds == "all"
                 else [int(seed) for seed in args.injection_seeds.split(",")])
        for seed in seeds:
            name = f"{dataset}_seed_{seed}"
            entry = spec["files"][str(seed)]
            reference = (reference_dir / f"{name}.json").read_bytes()
            record = json.loads(reference)
            keys = ("dataset", "injection_seed", "training_seed", "sha256", "n_rows", "n_anomalies",
                    "flowcad_commit", "mode", "selected_epoch")
            actual = tuple(record[key] for key in keys)
            expected = (dataset, seed, 201 + seed - 42, entry["sha256"], spec["n_rows"],
                        spec["n_anomalies"], MODEL_COMMIT, "fixed_1000", 1000)
            if actual != expected:
                raise ValueError(f"Archived result differs from the manifest: {name}")
            metadata = expected_metadata(record, digest(reference), run_sha)
            relative = Path("records") / f"{name}.npz"
            target = output / relative
            summary = {**metadata, "path": str(relative)}
            if target.exists():
                records.append({**summary, **read_existing(target, metadata)})
                continue
            sweep_path = args.sweep_records / f"{name}.npz" if args.sweep_records else None
            job = (spec, args.manifest.parent / entry["path"], target, metadata, sweep_path)
            steps = math.ceil(spec["n_rows"] / 128) * metadata["selected_epoch"]
            jobs.append((steps, summary, job))

    print(json.dumps({"validated_existing": len(records), "pending": len(jobs),
                      "workers": args.workers}), flush=True)
    if records:
        write_summary(output, records)
    jobs.sort(key=lambda item: (-item[0], item[1]["dataset"], item[1]["injection_seed"]))
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate, job): summary for _, summary, job in jobs}
        for future in as_completed(futures):
            row = futures[future]
            try:
                timing = future.result()
            except Exception as error:
                failures += 1
                print(f"{row['path']}: {error!r}", file=sys.stderr, flush=True)
                continue
            row = {**row, **timing}
            records.append(row)
            write_summary(output, records)
            print(json.dumps(row), flush=True)
    print(json.dumps({"completed": len(records), "failed": failures}), flush=True)
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
