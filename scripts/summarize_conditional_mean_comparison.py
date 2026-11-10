"""Compare Flow-CAD with the conditional-mean MLP or QCAD on paired rows."""

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import pandas as pd

from replay import MANIFEST, MODEL_COMMIT, ROOT, read_manifest, verify_model

METRICS = ("pr_auc", "roc_auc", "p_at_n")
SCOPES = ("all", "in_range")
TOLERANCE = 1e-8


def read_frozen(path, cases):
    frozen = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["dataset"], int(row["injection_seed"])
            if (key in frozen or key not in cases or row["sha256"] != cases[key]["sha256"]
                    or row["mode"] != "csm_05" or row["flowcad_commit"] != MODEL_COMMIT
                    or int(row["training_seed"]) != 201 + key[1] - 42):
                raise ValueError(f"Unexpected, duplicate, or mismatched frozen Flow-CAD case: {key}")
            frozen[key] = row
    if frozen.keys() != cases.keys():
        raise ValueError(f"Frozen Flow-CAD cases do not match the manifest: {sorted(cases.keys() - frozen.keys())}")
    return frozen


def read_frozen_qcad(path, cases):
    frozen = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["dataset"], int(row["injection_seed"])
            if (key in frozen or key not in cases or row["sha256"] != cases[key]["sha256"]
                    or row["mode"] != "paper"):
                raise ValueError(f"Unexpected, duplicate, or mismatched frozen QCAD case: {key}")
            metrics = {metric: float(row[metric]) for metric in METRICS}
            if not all(0 <= value <= 1 for value in metrics.values()):
                raise ValueError(f"Invalid frozen QCAD metrics: {key}")
            frozen[key] = {**row, "metrics": metrics}
    if frozen.keys() != cases.keys():
        raise ValueError(f"Frozen QCAD cases do not match the manifest: {sorted(cases.keys() - frozen.keys())}")
    return frozen


def find_records(directory, cases):
    expected_names = {f"{key[0]}_seed_{key[1]}.npz": key for key in cases}
    records = {}
    for path in directory.glob("*.npz"):
        if path.name not in expected_names:
            raise ValueError(f"Unexpected score record: {path}")
        records[expected_names[path.name]] = path
    return records


def load_input(path, info, spec):
    behavior = spec["behavior_columns"]
    if not behavior or not set(behavior) <= set(spec["scaled_columns"]):
        raise ValueError(f"Expected all behavioral features to be min–max scaled: {path}")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != info["sha256"]:
        raise ValueError(f"Input hash mismatch: {path}")
    frame = pd.read_csv(io.BytesIO(content), usecols=behavior + ["ground_truth"])
    labels = frame["ground_truth"].to_numpy()
    values = frame[behavior].to_numpy(dtype=np.float64)
    if (len(labels) != spec["n_rows"] or not np.isin(labels, [0, 1]).all()
            or int(labels.sum()) != spec["n_anomalies"] or not np.isfinite(values).all()):
        raise ValueError(f"Rows, labels, or behavior differ from the manifest: {path}")
    inside = np.all((values >= -TOLERANCE) & (values <= 1 + TOLERANCE), axis=1)
    return labels, inside


def validate_scores(path, key, frozen, labels, method, evaluate_scores):
    with np.load(path, allow_pickle=False) as record:
        if (record["dataset"].item(), int(record["injection_seed"].item())) != key:
            raise ValueError(f"Record metadata disagrees with the filename: {path}")
        if record["sha256"].item() != frozen["sha256"]:
            raise ValueError(f"Record input hash differs from the frozen case: {path}")
        if int(record["training_seed"].item()) != int(frozen["training_seed"]):
            raise ValueError(f"Record training seed differs from the frozen case: {path}")
        if (int(record["n_rows"].item()), int(record["n_anomalies"].item())) != (
            len(labels), int(labels.sum())
        ):
            raise ValueError(f"Record row or anomaly count differs from the input: {path}")
        epoch = int(record["selected_epoch"].item())
        if epoch <= 0 or (method == "flowcad" and record["flowcad_commit"].item() != frozen["flowcad_commit"]):
            raise ValueError(f"Record checkpoint or Flow-CAD commit is invalid: {path}")
        scores = record["scores"]
        stored = {metric: float(record[metric].item()) for metric in METRICS}
    metrics = evaluate_scores(labels, scores)
    if any(metrics[metric] != stored[metric] for metric in METRICS):
        raise ValueError(f"Scores do not reproduce recorded full-table metrics: {path}")
    if method == "flowcad" and (
        epoch != int(frozen["selected_epoch"])
        or any(metrics[metric] != float(frozen[metric]) for metric in METRICS)
    ):
        raise ValueError(f"Flow-CAD scores or epoch differ from frozen results: {path}")
    return {"scores": scores, "epoch": epoch, "metrics": metrics}


def validate_qcad_scores(path, key, frozen, labels, evaluate_scores):
    expected = {
        "dataset": key[0], "injection_seed": key[1], "sha256": frozen["sha256"],
        "n_rows": int(frozen["rows"]), "n_anomalies": int(frozen["n"]),
        "mode": frozen["mode"], "k": int(frozen["k"]), "sklearn": frozen["sklearn"],
        "lower_tail_zero_iqr_floors": int(frozen["lower_tail_zero_iqr_floors"]),
        **frozen["metrics"],
    }
    with np.load(path, allow_pickle=False) as record:
        if any(record[field].item() != value for field, value in expected.items()):
            raise ValueError(f"Record metadata or metrics differ from the frozen QCAD case: {path}")
        scores = record["scores"]
    metrics = evaluate_scores(labels, scores)
    if any(metrics[metric] != frozen["metrics"][metric] for metric in METRICS):
        raise ValueError(f"Scores do not reproduce the frozen full-table QCAD metrics: {path}")
    return {"scores": scores, "metrics": metrics, "source": "verified_scores"}


def scope_counts(labels, inside):
    return {"all": (len(labels), int(labels.sum())),
            "in_range": (int(inside.sum()), int(labels[inside].sum()))}


def paired_rows(key, frozen, labels, inside, counts, validated, evaluate_scores, comparator):
    for scope, (n_rows, n_anomalies) in counts.items():
        eligible = 0 < n_anomalies < n_rows
        row = {
            "dataset": key[0], "injection_seed": key[1], "scope": scope,
            "sha256": frozen["sha256"], "training_seed": int(frozen["training_seed"]),
            "n_rows": n_rows, "n_anomalies": n_anomalies, "defined": int(eligible),
        }
        for method in ("flowcad", comparator):
            result = validated[method]
            if method == "qcad":
                row["qcad_full_table_source"] = result["source"]
            else:
                row[f"{method}_epoch"] = result["epoch"]
            if scope == "all":
                metrics = result["metrics"]
            else:
                metrics = evaluate_scores(labels[inside], result["scores"][inside]) if eligible else {}
            row.update({f"{method}_{metric}": metrics.get(metric) for metric in METRICS})
        for metric in METRICS:
            row[f"delta_{metric}"] = (
                row[f"flowcad_{metric}"] - row[f"{comparator}_{metric}"] if eligible else None
            )
        yield row


def write_csv(path, fields, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def per_dataset_metrics(datasets, paired, coverage, comparator):
    groups = defaultdict(list)
    for row in paired:
        groups[row["dataset"], row["scope"]].append(row)
    summaries = []
    for dataset in sorted(datasets):
        status = coverage[dataset]
        for scope in SCOPES:
            cases = groups[dataset, scope]
            eligible = [row for row in cases if row["defined"]]
            summary = {"dataset": dataset, "scope": scope, "expected_runs": status["expected_runs"],
                       "expected_eligible_runs": status[f"{scope}_eligible_runs"],
                       "matched_runs": len(cases), "matched_eligible_runs": len(eligible),
                       "n_anomalies": sum(row["n_anomalies"] for row in cases)}
            for metric in METRICS:
                deltas = [row[f"delta_{metric}"] for row in eligible]
                for method in ("flowcad", comparator):
                    values = [row[f"{method}_{metric}"] for row in eligible]
                    summary[f"{method}_{metric}_mean"] = statistics.mean(values) if values else None
                    summary[f"{method}_{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else None
                summary[f"delta_{metric}_mean"] = statistics.mean(deltas) if deltas else None
                summary[f"flowcad_{metric}_run_wins"] = sum(delta > 0 for delta in deltas)
                summary[f"{comparator}_{metric}_run_wins"] = sum(delta < 0 for delta in deltas)
                summary[f"{metric}_run_ties"] = sum(delta == 0 for delta in deltas)
            summaries.append(summary)
    return summaries


def macro_metrics(summaries, comparator):
    macro = []
    comparator_mean_field = "qcad_mean" if comparator == "qcad" else "conditional_mean"
    for scope in SCOPES:
        defined = [row for row in summaries if row["scope"] == scope and row["matched_eligible_runs"]]
        for metric in METRICS:
            flow = [row[f"flowcad_{metric}_mean"] for row in defined]
            comparison = [row[f"{comparator}_{metric}_mean"] for row in defined]
            deltas = [left - right for left, right in zip(flow, comparison, strict=True)]
            macro.append({
                "scope": scope, "metric": metric, "n_datasets": len(defined),
                "n_runs": sum(row["matched_eligible_runs"] for row in defined),
                "flowcad_mean": statistics.mean(flow) if flow else None,
                comparator_mean_field: statistics.mean(comparison) if comparison else None,
                f"flowcad_minus_{comparator}": statistics.mean(deltas) if deltas else None,
                "flowcad_dataset_wins": sum(delta > 0 for delta in deltas),
                f"{comparator}_dataset_wins": sum(delta < 0 for delta in deltas),
                "dataset_ties": sum(delta == 0 for delta in deltas),
                "flowcad_run_wins": sum(row[f"flowcad_{metric}_run_wins"] for row in defined),
                f"{comparator}_run_wins": sum(row[f"{comparator}_{metric}_run_wins"] for row in defined),
                "run_ties": sum(row[f"{metric}_run_ties"] for row in defined),
            })
    return macro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--comparator", choices=("conditional_mean", "qcad"), default="conditional_mean")
    parser.add_argument("--frozen-flow", type=Path, help="Summary CSV from the complete Flow-CAD replay")
    parser.add_argument("--flow-records", type=Path, help="Directory of row-aligned Flow-CAD NPZ scores")
    parser.add_argument("--mean-records", type=Path, help="Directory of conditional-mean NPZ scores")
    parser.add_argument("--frozen-qcad", type=Path, help="Per-run CSV from the complete QCAD replay")
    parser.add_argument("--qcad-records", type=Path, help="Directory of row-aligned QCAD NPZ scores")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    comparator = args.comparator
    is_qcad = comparator == "qcad"
    if (is_qcad and args.mean_records) or (not is_qcad and (args.frozen_qcad or args.qcad_records)):
        parser.error("Specify score records for the selected comparator")
    verify_model()
    sys.path.insert(0, str(ROOT))
    from flowcad.metrics import evaluate_scores

    replay_dir = args.manifest.parent.parent
    frozen_flow = args.frozen_flow or replay_dir / "flowcad_csm05" / "summary.csv"
    flow_records = args.flow_records or replay_dir / "qcad_in_range_flow" / "records"
    prefix = "qcad_in_range" if is_qcad else "conditional_mean"
    if is_qcad:
        comparator_records = args.qcad_records or replay_dir / "qcad_in_range" / "records"
        output = args.output or replay_dir / "qcad_in_range"
    else:
        comparator_records = args.mean_records or replay_dir / "qcad_conditional_mean" / "records"
        output = args.output or replay_dir / "qcad_conditional_mean"
    manifest = read_manifest(args.manifest)
    datasets = manifest["datasets"]
    if sorted(datasets) != sorted(manifest["expected_datasets"]):
        raise ValueError("The manifest dataset entries do not match its declared coverage")
    cases = {}
    for dataset, spec in datasets.items():
        seeds = spec["injection_seeds"]
        if len(seeds) != spec["benchmark_runs"] or sorted(map(str, seeds)) != sorted(spec["files"]):
            raise ValueError(f"The manifest seed coverage is inconsistent: {dataset}")
        cases.update({(dataset, seed): spec["files"][str(seed)] for seed in seeds})
    frozen = read_frozen(frozen_flow, cases)
    flow = find_records(flow_records, cases)
    comparison = find_records(comparator_records, cases)
    if is_qcad:
        frozen_qcad = read_frozen_qcad(args.frozen_qcad or replay_dir / "all_qcad" / "per_run.csv", cases)
    required_comparison_scores = set()
    pairs, undefined = [], []
    coverage = {dataset: {"expected_runs": len(spec["injection_seeds"]), "all_eligible_runs": 0,
                          "in_range_eligible_runs": 0, "full_anomalies": 0, "in_range_anomalies": 0,
                          "out_of_range_normals": 0} for dataset, spec in datasets.items()}
    for key, info in sorted(cases.items()):
        dataset, seed = key
        labels, inside = load_input(args.manifest.parent / info["path"], info, datasets[dataset])
        status = coverage[dataset]
        counts = scope_counts(labels, inside)
        all_counts, range_counts = counts["all"], counts["in_range"]
        status["all_eligible_runs"] += int(0 < all_counts[1] < all_counts[0])
        status["full_anomalies"] += all_counts[1]
        status["in_range_anomalies"] += range_counts[1]
        status["out_of_range_normals"] += int(np.count_nonzero(~inside & (labels == 0)))
        in_range_eligible = 0 < range_counts[1] < range_counts[0]
        if in_range_eligible:
            status["in_range_eligible_runs"] += 1
        else:
            undefined.append({"dataset": dataset, "injection_seed": seed, "n_rows": range_counts[0],
                              "n_anomalies": range_counts[1]})
        if in_range_eligible or not is_qcad:
            required_comparison_scores.add(key)
        validated = {}
        if key in flow:
            validated["flowcad"] = validate_scores(flow[key], key, frozen[key], labels, "flowcad", evaluate_scores)
        if is_qcad:
            qcad_case = frozen_qcad[key]
            if (int(qcad_case["rows"]), int(qcad_case["n"])) != all_counts:
                raise ValueError(f"Frozen QCAD row or anomaly count differs from the input: {key}")
            if key in comparison:
                validated["qcad"] = validate_qcad_scores(comparison[key], key, qcad_case, labels, evaluate_scores)
            elif not in_range_eligible:
                # No within-range metric can be calculated for these cases.
                validated["qcad"] = {"metrics": qcad_case["metrics"], "source": "archived_replay"}
        elif key in comparison:
            validated[comparator] = validate_scores(comparison[key], key, frozen[key], labels, comparator, evaluate_scores)
        if len(validated) == 2:
            pairs.extend(paired_rows(key, frozen[key], labels, inside, counts, validated, evaluate_scores, comparator))

    complete = flow.keys() == cases.keys() and required_comparison_scores <= comparison.keys()
    summaries = per_dataset_metrics(datasets, pairs, coverage, comparator)
    status = {
        "complete": complete, "n_datasets": len(datasets), "expected_runs_per_method": len(cases),
        "validated_records": {"flowcad": len(flow), comparator: len(comparison)},
        "matched_runs": sum(row["scope"] == "all" for row in pairs),
        "matched_in_range_eligible_runs": sum(row["scope"] == "in_range" and row["defined"] for row in pairs),
        "expected_in_range_eligible_runs": len(cases) - len(undefined),
        "in_range_bounds": [-TOLERANCE, 1 + TOLERANCE],
        "macro_policy": "Average paired eligible runs within dataset, then equally weight datasets.",
        "in_range_undefined_cases": undefined, "dataset_coverage": coverage,
    }
    if is_qcad:
        status["expected_qcad_score_records"] = len(required_comparison_scores)
        status["frozen_qcad_records"] = len(frozen_qcad)
    output.mkdir(parents=True, exist_ok=True)
    run_fields = ["dataset", "injection_seed", "scope", "sha256", "training_seed", "n_rows", "n_anomalies", "defined"]
    for method in ("flowcad", comparator):
        metadata_field = "qcad_full_table_source" if method == "qcad" else f"{method}_epoch"
        run_fields.extend([metadata_field, *(f"{method}_{metric}" for metric in METRICS)])
    run_fields.extend(f"delta_{metric}" for metric in METRICS)
    write_csv(output / f"{prefix}_per_run.csv", run_fields, pairs)
    write_csv(output / f"{prefix}_per_dataset.csv", list(summaries[0]), summaries)
    macro_path = output / f"{prefix}_macro.csv"
    if complete:
        macro = macro_metrics(summaries, comparator)
        write_csv(macro_path, list(macro[0]), macro)
    else:
        macro_path.unlink(missing_ok=True)
    temporary = output / f"{prefix}_coverage.tmp"
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(output / f"{prefix}_coverage.json")
    print(json.dumps({key: value for key, value in status.items() if key not in
                      {"dataset_coverage", "in_range_undefined_cases"}}), flush=True)


if __name__ == "__main__":
    main()
