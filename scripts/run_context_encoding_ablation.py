"""Compare one-hot and ordinal context encodings on paired benchmark tables."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from hashlib import sha256
import math
from pathlib import Path
from statistics import mean
import sys

from replay import MANIFEST, ROOT, read_manifest, verify_model
from run_qcad_paper_flowcad import FIRST_INJECTION_SEED, FIRST_MODEL_SEED, METRICS


DATASETS = ("fish", "qsranking")


def read_flowcad_records(path):
    records = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("method", "Flow-CAD") != "Flow-CAD" or row["dataset"] not in DATASETS:
                continue
            key = (row["dataset"], int(row["injection_seed"]))
            if key in records:
                raise ValueError(f"Duplicate Flow-CAD record {key} in {path}")
            records[key] = row
    return records


def evaluate_encodings(dataset, injection_seed, spec, path, expected_hash):
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import pandas as pd
    import torch

    from flowcad.data import load_benchmark
    from flowcad.fixed import ScoringDraws, fit_fixed
    from flowcad.metrics import evaluate_scores

    torch.set_num_threads(1)
    data = load_benchmark({**spec, "file": path.name}, path.parent, require_labels=False)
    if data.sha256 != expected_hash or len(data.frame) != spec["n_rows"] or data.dropped_rows:
        raise ValueError(f"Input differs from manifest: {dataset}/{injection_seed}")
    category, = spec["categorical_columns"]
    numerical = [name for name in spec["context_columns"] if name != category]
    codes, levels = pd.factorize(data.frame[category], sort=True)
    if len(levels) < 2 or np.any(codes < 0):
        raise ValueError(f"Expected at least two observed category levels: {dataset}")
    ordinal = codes.astype(np.float32) / np.float32(len(levels) - 1)
    context = np.column_stack((data.frame[numerical].to_numpy(dtype=np.float32), ordinal))
    training_seed = FIRST_MODEL_SEED + injection_seed - FIRST_INJECTION_SEED
    ordinal_model = fit_fixed(context, data.behavior, training_seed)
    ordinal_scores = ScoringDraws(context, data.behavior, training_seed).scores(ordinal_model)

    labels = pd.read_csv(path, usecols=["ground_truth"]).loc[data.frame.index, "ground_truth"].to_numpy()
    metrics = evaluate_scores(labels, ordinal_scores)
    if metrics["n"] != spec["n_anomalies"]:
        raise ValueError(f"Unexpected number of anomalies: {dataset}/{injection_seed}")
    one_hot_model = fit_fixed(data.context, data.behavior, training_seed)
    one_hot_scores = ScoringDraws(data.context, data.behavior, training_seed).scores(one_hot_model)
    one_hot_metrics = evaluate_scores(labels, one_hot_scores)
    rare_levels = np.bincount(codes)[codes] <= 5
    top = pd.Series(one_hot_scores).sort_values(ascending=False).head(metrics["n"]).index.to_numpy()
    recovered = np.zeros(len(labels), dtype=bool)
    recovered[top] = True
    anomalies = labels == 1
    return {
        "dataset": dataset, "injection_seed": injection_seed, "training_seed": training_seed,
        "sha256": data.sha256, "n_rows": len(data.frame), "category_column": category,
        "n_levels": len(levels), "n_singleton_levels": int(np.sum(np.bincount(codes) == 1)),
        "epoch": 1000,
        **{f"ordinal_{metric}": metrics[metric] for metric in METRICS},
        **{f"one_hot_{metric}": one_hot_metrics[metric] for metric in METRICS},
        "rare_anomalies": int(np.count_nonzero(anomalies & rare_levels)),
        "rare_recovered": int(np.count_nonzero(anomalies & rare_levels & recovered)),
        "other_anomalies": int(np.count_nonzero(anomalies & ~rare_levels)),
        "other_recovered": int(np.count_nonzero(anomalies & ~rare_levels & recovered)),
    }


def write_csv(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(rows):
    summaries = []
    for dataset in DATASETS:
        subset = [row for row in rows if row["dataset"] == dataset]
        summary = {"dataset": dataset, "n_runs": len(subset)}
        for metric in METRICS:
            one_hot = mean(row[f"one_hot_{metric}"] for row in subset)
            ordinal = mean(row[f"ordinal_{metric}"] for row in subset)
            summary.update({
                f"one_hot_{metric}": one_hot,
                f"ordinal_{metric}": ordinal,
                f"delta_ordinal_minus_one_hot_{metric}": ordinal - one_hot,
            })
        for key in ("rare_anomalies", "rare_recovered", "other_anomalies", "other_recovered"):
            summary[key] = sum(row[key] for row in subset)
        summaries.append(summary)
    return summaries


def read_completed(path, jobs, baseline):
    if not path.exists():
        return {}
    completed = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["dataset"], int(row["injection_seed"])
            if key in completed or key not in jobs:
                raise ValueError(f"Duplicate or unexpected cached ordinal run: {key}")
            _, seed, spec, input_path, expected_hash = jobs[key]
            training_seed = FIRST_MODEL_SEED + seed - FIRST_INJECTION_SEED
            category, = spec["categorical_columns"]
            if (row["sha256"] != expected_hash or sha256(input_path.read_bytes()).hexdigest() != expected_hash
                    or int(row["training_seed"]) != training_seed or int(row["n_rows"]) != spec["n_rows"]
                    or row["category_column"] != category
                    or any(float(row[field]) != value for field, value in baseline[key].items())):
                raise ValueError(f"Cached ordinal run differs from its replay inputs: {key}")
            levels, singletons = int(row["n_levels"]), int(row["n_singleton_levels"])
            epoch = int(row["epoch"])
            metrics = {f"ordinal_{metric}": float(row[f"ordinal_{metric}"]) for metric in METRICS}
            counts = {key: int(row[key]) for key in ("rare_anomalies", "rare_recovered",
                                                    "other_anomalies", "other_recovered")}
            if (not 0 <= singletons <= levels <= spec["n_rows"] or levels < 2 or epoch != 1000
                    or not 0 <= counts["rare_recovered"] <= counts["rare_anomalies"]
                    or not 0 <= counts["other_recovered"] <= counts["other_anomalies"]
                    or counts["rare_anomalies"] + counts["other_anomalies"] != spec["n_anomalies"]
                    or any(not 0 <= value <= 1 for value in metrics.values())):
                raise ValueError(f"Invalid cached ordinal result: {key}")
            completed[key] = {
                "dataset": key[0], "injection_seed": seed, "training_seed": training_seed,
                "sha256": expected_hash, "n_rows": spec["n_rows"], "category_column": category,
                "n_levels": levels, "n_singleton_levels": singletons, "epoch": epoch,
                **metrics, **baseline[key], **counts,
            }
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--one-hot-summary", type=Path,
                        help="Use this local fixed-epoch replay summary as the one-hot reference")
    parser.add_argument("--output", type=Path, help="Use a fresh directory to repeat completed runs")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Workers must be positive")

    revision = verify_model()
    datasets = read_manifest(args.manifest)["datasets"]
    replay = args.manifest.parent.parent
    local = args.one_hot_summary or replay / "flowcad_fixed_1000" / "summary.csv"
    use_local = args.one_hot_summary is not None or local.exists()
    reference = read_flowcad_records(local if use_local else ROOT / "results" / "main_per_run.csv")
    output = args.output or replay / "context_encoding_ablation_fixed_1000"
    jobs, baseline = {}, {}
    # Submit the larger dataset first to reduce idle workers near completion.
    for dataset in sorted(DATASETS, key=lambda name: -datasets[name]["n_rows"]):
        spec = datasets[dataset]
        for injection_seed in spec["injection_seeds"]:
            key = dataset, injection_seed
            info = spec["files"][str(injection_seed)]
            row = reference[key]
            if row["sha256"] != info["sha256"]:
                raise ValueError(f"One-hot reference differs from the input manifest: {key}")
            if use_local:
                training_seed = FIRST_MODEL_SEED + injection_seed - FIRST_INJECTION_SEED
                if (row["flowcad_commit"] != revision
                        or row["mode"] != "fixed_1000" or int(row["selected_epoch"]) != 1000
                        or int(row["training_seed"]) != training_seed):
                    raise ValueError(f"One-hot reference uses a different fixed-epoch protocol: {key}")
            baseline[key] = {f"one_hot_{metric}": float(row[metric]) for metric in METRICS}
            jobs[key] = (dataset, injection_seed, spec, args.manifest.parent / info["path"], info["sha256"])

    output.mkdir(parents=True, exist_ok=True)
    per_run = output / "per_run.csv"
    results = read_completed(per_run, jobs, baseline)
    print(f"{len(results)}/{len(jobs)} runs already complete.", flush=True)
    if len(results) < len(jobs):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(evaluate_encodings, *job): key for key, job in jobs.items() if key not in results}
            for completed in as_completed(pending):
                key = pending[completed]
                row = completed.result()
                if any(not math.isclose(row[field], value, abs_tol=1e-12, rel_tol=0)
                       for field, value in baseline[key].items()):
                    raise ValueError(f"One-hot run did not reproduce the reference metrics: {key}")
                results[key] = row
                write_csv(per_run, [results[item] for item in sorted(results)])
                print(f"{len(results)}/{len(jobs)}: {key[0]} seed {key[1]}", flush=True)
    rows = [results[key] for key in sorted(results)]
    write_csv(per_run, rows)
    write_csv(output / "summary.csv", summarize(rows))


if __name__ == "__main__":
    main()
