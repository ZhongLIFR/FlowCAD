"""Run Flow-CAD on the fixed benchmark tables."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import sys
import time


from replay import MAIN_MANIFEST as MANIFEST, ROOT, read_manifest, verify_model


DEFAULT_EPOCHS = 1000
METRICS = ("pr_auc", "roc_auc", "p_at_n")
FIRST_INJECTION_SEED = 42
FIRST_MODEL_SEED = 201


def evaluate(job, training_seed=None):
    release, revision, dataset, injection_seed, spec, path, expected_hash = job
    sys.path.insert(0, str(release))
    import pandas as pd
    import torch

    from flowcad.data import load_benchmark
    from flowcad.fixed import ScoringDraws, fit_fixed
    from flowcad.metrics import evaluate_scores

    torch.set_num_threads(1)
    data = load_benchmark({**spec, "file": path.name}, path.parent, require_labels=False)
    if data.sha256 != expected_hash or len(data.frame) != spec["n_rows"] or data.dropped_rows:
        raise ValueError(f"Input differs from manifest: {dataset}/{injection_seed}")
    if training_seed is None:
        training_seed = FIRST_MODEL_SEED + injection_seed - FIRST_INJECTION_SEED
    started = time.perf_counter()
    epoch = DEFAULT_EPOCHS
    model = fit_fixed(data.context, data.behavior, training_seed, epoch)
    scores = ScoringDraws(data.context, data.behavior, training_seed).scores(model)
    elapsed = time.perf_counter() - started
    labels = pd.read_csv(path, usecols=["ground_truth"]).loc[data.frame.index, "ground_truth"]
    metrics = evaluate_scores(labels.to_numpy(), scores)
    if metrics["n"] != spec["n_anomalies"]:
        raise ValueError(f"Unexpected number of anomalies: {dataset}/{injection_seed}")
    return {
        "dataset": dataset, "injection_seed": injection_seed, "training_seed": training_seed,
        "n_rows": len(data.frame), "n_anomalies": metrics["n"], "sha256": data.sha256,
        "flowcad_commit": revision, "mode": f"fixed_{DEFAULT_EPOCHS}",
        "selected_epoch": epoch,
        **{key: metrics[key] for key in METRICS}, "fit_select_seconds": elapsed,
    }


def write_summary(output, rows):
    records = sorted(rows, key=lambda row: (row["dataset"], row["injection_seed"]))
    fields = list(records[0])
    temporary = output / "summary.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(output / "summary.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--datasets", default="small", help="small (eight smallest), all, or comma-separated keys")
    parser.add_argument("--injection-seeds", default="all", help="all or comma-separated seeds")
    parser.add_argument("--fixed-epochs", type=int, choices=(DEFAULT_EPOCHS,), default=DEFAULT_EPOCHS,
                        help=f"Fixed training budget (the main benchmark uses {DEFAULT_EPOCHS})")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    release = ROOT
    revision = verify_model()
    if args.workers < 1:
        parser.error("Workers must be positive")
    mode = f"fixed_{DEFAULT_EPOCHS}"
    inputs = read_manifest(args.manifest)["datasets"]
    ordered = sorted(inputs, key=lambda name: (inputs[name]["n_rows"], name))
    datasets = ordered[:8] if args.datasets == "small" else ordered
    if args.datasets not in ("small", "all"):
        datasets = args.datasets.split(",")
    output = args.manifest.parent.parent / f"flowcad_{mode}"
    output.mkdir(parents=True, exist_ok=True)
    jobs, records = [], []
    for dataset in datasets:
        spec = inputs[dataset]
        seeds = spec["injection_seeds"] if args.injection_seeds == "all" else [
            int(seed) for seed in args.injection_seeds.split(",")
        ]
        for seed in seeds:
            info = spec["files"][str(seed)]
            path = args.manifest.parent / info["path"]
            record_path = output / f"{dataset}_seed_{seed}.json"
            expected_training_seed = FIRST_MODEL_SEED + seed - FIRST_INJECTION_SEED
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if (
                    record.get("dataset") != dataset
                    or record.get("injection_seed") != seed
                    or record.get("training_seed") != expected_training_seed
                    or record.get("n_rows") != spec["n_rows"]
                    or record.get("sha256") != info["sha256"]
                    or record.get("flowcad_commit") != revision
                ):
                    raise ValueError(f"Previous result has different data or code: {record_path}")
                if record.get("mode") != mode or record.get("selected_epoch") != DEFAULT_EPOCHS:
                    raise ValueError(f"Previous result uses a different training protocol: {record_path}")
                records.append(record)
            else:
                job = (release, revision, dataset, seed, spec, path, info["sha256"])
                jobs.append((record_path, job))
    if records:
        write_summary(output, records)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(evaluate, job): record_path for record_path, job in jobs}
        for future in as_completed(pending):
            record = future.result()
            path = pending[future]
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            temporary.replace(path)
            records.append(record)
            write_summary(output, records)
            print(json.dumps({key: value for key, value in record.items() if key != "candidates"}),
                  flush=True)


if __name__ == "__main__":
    main()
