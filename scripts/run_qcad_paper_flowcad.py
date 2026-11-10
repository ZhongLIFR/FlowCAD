"""Run published Flow-CAD on regenerated QCAD paper-protocol tables."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import sys
import time


from replay import MANIFEST, ROOT, read_manifest, verify_model
METRICS = ("pr_auc", "roc_auc", "p_at_n")
FIRST_INJECTION_SEED = 42
FIRST_MODEL_SEED = 201


def evaluate(job):
    release, revision, dataset, injection_seed, spec, path, expected_hash, fixed_epochs = job
    sys.path.insert(0, str(release))
    import pandas as pd
    import torch

    from flowcad import checkpoints
    from flowcad.data import load_benchmark
    from flowcad.metrics import evaluate_scores

    torch.set_num_threads(1)
    data = load_benchmark({**spec, "file": path.name}, path.parent, require_labels=False)
    if data.sha256 != expected_hash or len(data.frame) != spec["n_rows"] or data.dropped_rows:
        raise ValueError(f"Input differs from manifest: {dataset}/{injection_seed}")
    if fixed_epochs is not None:
        checkpoints.CANDIDATE_EPOCHS = (fixed_epochs,)
    training_seed = FIRST_MODEL_SEED + injection_seed - FIRST_INJECTION_SEED
    started = time.perf_counter()
    result = checkpoints.select_checkpoint(data.context, data.behavior, training_seed)
    elapsed = time.perf_counter() - started
    labels = pd.read_csv(path, usecols=["ground_truth"]).loc[data.frame.index, "ground_truth"]
    metrics = evaluate_scores(labels.to_numpy(), result["scores"])
    if metrics["n"] != spec["n_anomalies"]:
        raise ValueError(f"Unexpected number of anomalies: {dataset}/{injection_seed}")
    return {
        "dataset": dataset, "injection_seed": injection_seed, "training_seed": training_seed,
        "n_rows": len(data.frame), "n_anomalies": metrics["n"], "sha256": data.sha256,
        "flowcad_commit": revision, "mode": "csm_05" if fixed_epochs is None else f"fixed_{fixed_epochs}",
        "selected_epoch": result["epoch"], "candidates": result["candidates"],
        **{key: metrics[key] for key in METRICS}, "fit_select_seconds": elapsed,
    }


def write_summary(output, rows):
    records = sorted(rows, key=lambda row: (row["dataset"], row["injection_seed"]))
    fields = [key for key in records[0] if key != "candidates"]
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
    parser.add_argument("--injection-seeds", default="42", help="all or comma-separated seeds")
    parser.add_argument("--fixed-epochs", type=int, help="Use one fixed epoch instead of the published CSM grid")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    release = ROOT
    revision = verify_model()
    if args.workers < 1 or (args.fixed_epochs is not None and args.fixed_epochs < 1):
        parser.error("Workers and fixed epochs must be positive")
    inputs = read_manifest(args.manifest)["datasets"]
    ordered = sorted(inputs, key=lambda name: (inputs[name]["n_rows"], name))
    datasets = ordered[:8] if args.datasets == "small" else ordered
    if args.datasets not in ("small", "all"):
        datasets = args.datasets.split(",")
    output = args.manifest.parent.parent / (
        "flowcad_csm05" if args.fixed_epochs is None else f"flowcad_fixed_{args.fixed_epochs}"
    )
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
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if record["sha256"] != info["sha256"] or record["flowcad_commit"] != revision:
                    raise ValueError(f"Previous result has different data or code: {record_path}")
                records.append(record)
            else:
                job = (release, revision, dataset, seed, spec, path, info["sha256"], args.fixed_epochs)
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
