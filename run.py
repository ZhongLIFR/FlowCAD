#!/usr/bin/env python3
"""Run Flow-CAD on prepared tabular benchmarks."""

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def parse_seeds(value):
    pieces = [piece.strip() for piece in value.split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise ValueError("--seeds must be a comma-separated list of integers")
    try:
        seeds = [int(piece) for piece in pieces]
    except ValueError as error:
        raise ValueError("--seeds must be a comma-separated list of integers") from error
    if any(seed < 0 or seed > 2**32 - 1 for seed in seeds):
        raise ValueError("Each seed must be between 0 and 4294967295")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds contains duplicate seeds")
    return seeds


def parse_datasets(value, configs):
    if value.strip().lower() == "all":
        return list(configs)
    keys = [piece.strip().lower() for piece in value.split(",")]
    if not keys or any(not key for key in keys):
        raise ValueError("--datasets must contain dataset keys or 'all'")
    unknown = [key for key in keys if key not in configs]
    if unknown:
        raise ValueError(f"Unknown dataset key(s): {', '.join(unknown)}; use --list-datasets")
    if len(set(keys)) != len(keys):
        raise ValueError("--datasets contains duplicate dataset keys")
    return keys


def software_versions():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("flowcad", "numpy", "pandas", "scipy", "scikit-learn", "torch"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            if package == "flowcad":
                from flowcad import __version__
                versions[package] = __version__
            else:
                versions[package] = "not installed"
    return versions


def dataset_metadata(key, config, data, epochs):
    import numpy as np

    if data.labels is not None and len(np.unique(data.labels)) != 2:
        raise ValueError(f"{key}: evaluation needs both normal and anomalous labels")
    median = np.median(data.behavior, axis=0)
    scales = np.maximum(np.median(np.abs(data.behavior - median), axis=0), 1e-6)
    return {
        "dataset": key,
        "display_name": config.get("display_name", key),
        "file": config["file"],
        "sha256": data.sha256,
        "n_rows": int(len(data.frame)),
        "n_anomalies": None if data.labels is None else int(np.sum(data.labels == 1)),
        "dropped_rows": int(data.dropped_rows),
        "context_columns": config["context_columns"],
        "categorical_columns": config.get("categorical_columns", []),
        "context_feature_names": data.context_feature_names,
        "behavior_feature_names": data.behavior_feature_names,
        "behavior_mad_scale": scales.tolist(),
        "epochs": epochs,
        "notes": config.get("notes", ""),
    }


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def synchronize(device, torch):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def run(args, configs, seeds, requested_keys):
    import numpy as np
    import pandas as pd
    import torch
    from flowcad import FlowCAD, evaluate_scores, percentile_ranks
    from flowcad.data import load_benchmark

    use_csm = args.epochs is None and not args.validate_only
    if use_csm:
        from flowcad.checkpoints import CANDIDATE_EPOCHS, select_checkpoint

        if (args.device, args.threads, args.k_z, args.k_t, args.tau, args.batch_size, args.lr) != (
            "cpu", 1, 16, 16, 10.0, 128, 0.001,
        ):
            raise ValueError("CSM mode uses the paper settings; pass --epochs for custom settings")
    torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu or auto")
    keys = list(configs) if args.validate_only else requested_keys
    descriptions = []
    # Check every requested snapshot before beginning a potentially long run.
    # Validation mode checks all configured snapshots without fitting any model.
    for key in keys:
        config = configs[key]
        data = load_benchmark(config, args.data_dir, require_labels=not use_csm)
        descriptions.append(dataset_metadata(key, config, data, args.epochs))
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": "validation_only" if args.validate_only else "benchmark",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "software_versions": software_versions(),
        "settings": {
            "seeds": seeds,
            "epochs_override": args.epochs,
            "device": device,
            "threads": args.threads,
            "n_z_samples": args.k_z,
            "n_t_samples": args.k_t,
            "tau": args.tau,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": 64,
            "context_emb_dim": 32,
            "num_layers": 3,
            "score_rng": ("seed + 1000003, shared across checkpoints" if use_csm
                          else "continues from fitting; no score seed reset"),
            "evaluation_scores": "raw energy" if use_csm else "percentile ranks of raw energy",
            "labels_used_for": "evaluation only",
        },
        "datasets": descriptions,
    }
    if use_csm:
        report["settings"]["checkpoint_selection"] = {
            "criterion": "CSM on the top 5% of raw scores versus the remainder",
            "candidate_epochs": list(CANDIDATE_EPOCHS),
            "tie_break": "earliest epoch",
        }
    save_json(output / "metadata.json", report)
    if args.validate_only:
        print(json.dumps({
            "status": "validated",
            "n_datasets": len(descriptions),
            "training_performed": False,
            "datasets": [{name: item[name] for name in
                          ("dataset", "file", "n_rows", "n_anomalies", "dropped_rows", "sha256")}
                         for item in descriptions],
            "report_file": "metadata.json",
        }, indent=2, allow_nan=False))
        return

    (output / "scores").mkdir(parents=True, exist_ok=True)
    if use_csm:
        (output / "selections").mkdir()
    timing_fields = ["fit_select_seconds"] if use_csm else ["train_seconds", "score_seconds"]
    records = []
    for description in descriptions:
        key = description["dataset"]
        data = load_benchmark(configs[key], args.data_dir, require_labels=not use_csm)
        if data.sha256 != description["sha256"]:
            raise ValueError(f"{key}: input changed during benchmark setup")
        for seed in seeds:
            run_name = f"{key}_seed{seed}"
            if use_csm:
                start = time.perf_counter()
                selected = select_checkpoint(data.context, data.behavior, seed)
                timing = {"fit_select_seconds": time.perf_counter() - start}
                raw, contributions = selected.pop("scores"), selected.pop("contributions")
                epochs = selected["epoch"]
                save_json(output / "selections" / f"{run_name}.json",
                          {"dataset": key, "seed": seed, "sha256": data.sha256, **selected})
                path = args.data_dir / configs[key]["file"]
                labels = pd.read_csv(path, usecols=["ground_truth"]).loc[
                    data.frame.index, "ground_truth"
                ].to_numpy()
            else:
                epochs = description["epochs"]
                estimator = FlowCAD(
                    n_epochs=epochs, batch_size=args.batch_size, lr=args.lr,
                    n_z_samples=args.k_z, n_t_samples=args.k_t, tau=args.tau,
                    random_state=seed, device=device, verbose=False,
                )
                synchronize(device, torch)
                start = time.perf_counter()
                estimator.fit(data.context, data.behavior)
                synchronize(device, torch)
                train_seconds = time.perf_counter() - start
                start = time.perf_counter()
                raw, contributions = estimator.score_samples(
                    data.context, data.behavior, return_contributions=True,
                )
                synchronize(device, torch)
                timing = {"train_seconds": train_seconds, "score_seconds": time.perf_counter() - start}
                labels = data.labels
            percentiles = percentile_ranks(raw)
            metrics = evaluate_scores(labels, raw if use_csm else percentiles)
            description["n_anomalies"] = metrics["n"]
            record = {
                "dataset": key, "seed": seed,
                "n_rows": description["n_rows"], "n_anomalies": description["n_anomalies"],
                "epochs": epochs,
                **{name: metrics[name] for name in ("pr_auc", "roc_auc", "p_at_n")},
                **timing,
            }
            records.append(record)
            scores = pd.DataFrame({
                "original_row_index": data.frame.index.to_numpy(),
                "ground_truth": np.asarray(labels, dtype=int),
                "raw_anomaly_score": raw, "anomaly_score": percentiles,
            })
            for position, name in enumerate(data.behavior_feature_names):
                scores[f"contribution_{name}"] = contributions[:, position]
            scores.to_csv(output / "scores" / f"{run_name}.csv", index=False)
            # Persist completed runs so an interrupted later run remains inspectable.
            pd.DataFrame(records).to_csv(output / "metrics.csv", index=False)
            print(json.dumps(record, allow_nan=False), flush=True)
    results = pd.DataFrame(records)
    measures = ["pr_auc", "roc_auc", "p_at_n", *timing_fields]
    summary = results.groupby("dataset", sort=False)[measures].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    counts = results.groupby("dataset", sort=False).size().rename("n_runs")
    summary = counts.to_frame().join(summary).reset_index()
    # pandas std uses ddof=1: one run yields NaN, not invented zero variability.
    summary.to_csv(output / "summary.csv", index=False)
    report["completed_runs"] = len(records)
    report["summary_std_ddof"] = 1
    report["result_scope"] = "Flow-CAD benchmark"
    save_json(output / "metadata.json", report)
    print(f"Saved {len(records)} run(s) to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="energy", help="Comma-separated dataset keys, or all")
    parser.add_argument("--seeds", default="201", help="Comma-separated unsigned integer seeds")
    parser.add_argument("--epochs", type=positive_integer,
                        help="Use a fixed epoch budget instead of the default CSM selection")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "GenData")
    parser.add_argument("--output-dir", type=Path,
                        help="Fresh output directory; default: a timestamped directory under outputs/")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=positive_integer, default=1)
    parser.add_argument("--k-z", type=positive_integer, default=16)
    parser.add_argument("--k-t", type=positive_integer, default=16)
    parser.add_argument("--tau", type=positive_float, default=10.0)
    parser.add_argument("--batch-size", type=positive_integer, default=128)
    parser.add_argument("--lr", type=positive_float, default=1e-3)
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate all configured snapshots and save JSON metadata; do not train")
    parser.add_argument("--list-datasets", action="store_true", help="List configured dataset keys and exit")
    args = parser.parse_args()
    if args.output_dir is None:
        mode = "validation" if args.validate_only else "benchmarks"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        args.output_dir = PROJECT_ROOT / "outputs" / mode / stamp
    from flowcad.data import load_configs

    try:
        configs = load_configs()
        if args.list_datasets:
            for key, config in configs.items():
                print(f"{key}\t{config.get('display_name', key)}\t{config['file']}")
            return
        seeds = parse_seeds(args.seeds)
        keys = parse_datasets(args.datasets, configs)
        run(args, configs, seeds, keys)
    except (ValueError, FileNotFoundError, KeyError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
