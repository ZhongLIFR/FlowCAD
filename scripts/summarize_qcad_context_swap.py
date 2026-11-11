"""Validate the paired context-swap results and summarize per-dataset wins."""

import argparse
import csv
from hashlib import file_digest
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import mean


METHODS = ("flowcad", "qcad", "conditional_mean")
METRICS = ("pr_auc", "roc_auc", "p_at_n")
TIE_TOLERANCE = 1e-12
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "qcad_context_swap"


def outcome(flowcad, comparator):
    if isclose(flowcad, comparator, rel_tol=0, abs_tol=TIE_TOLERANCE):
        return "tie"
    return "win" if flowcad > comparator else "loss"


def check_result(path, expected):
    record = json.loads(path.read_bytes())
    for field, value in expected.items():
        if record[field] != value:
            raise ValueError(f"{path}: {field} is {record[field]!r}, expected {value!r}")
    for metric in METRICS:
        value = record[metric]
        if not isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{path}: invalid {metric} value {value!r}")
    if expected["method"] != "qcad":
        epoch = record["selected_epoch"]
        if type(epoch) is not int or epoch < 1:
            raise ValueError(f"{path}: invalid selected epoch {epoch!r}")
    return record


def read_paired(root):
    manifest_path = root / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    protocol = json.loads((root / "runs" / "protocol.json").read_bytes())
    with manifest_path.open("rb") as stream:
        manifest_sha256 = file_digest(stream, "sha256").hexdigest()
    if protocol["manifest_sha256"] != manifest_sha256 or tuple(protocol["methods"]) != METHODS:
        raise ValueError("Recorded run protocol differs from the input manifest or methods")
    datasets = manifest["datasets"]
    expected_datasets = manifest["expected_datasets"]
    if not datasets or sorted(datasets) != sorted(expected_datasets):
        raise ValueError(f"{manifest_path}: dataset keys differ from expected_datasets")

    expected_names = set()
    for dataset, spec in datasets.items():
        seeds = spec["injection_seeds"]
        if not seeds or sorted(spec["files"]) != sorted(map(str, seeds)):
            raise ValueError(f"{manifest_path}: files and seeds disagree for {dataset}")
        expected_names.update(f"{dataset}_seed_{seed}.json" for seed in seeds)

    for method in METHODS:
        directory = root / "runs" / method
        names = {path.name for path in directory.glob("*.json")}
        missing = sorted(expected_names - names)
        extra = sorted(names - expected_names)
        if missing or extra:
            raise ValueError(
                f"{method}: {len(missing)} missing and {len(extra)} unexpected records. "
                f"Missing examples: {missing[:5]}. Unexpected examples: {extra[:5]}."
            )

    paired = {}
    for dataset in expected_datasets:
        spec = datasets[dataset]
        for seed in spec["injection_seeds"]:
            info = spec["files"][str(seed)]
            input_path = manifest_path.parent / info["path"]
            with input_path.open("rb") as stream:
                checksum = file_digest(stream, "sha256").hexdigest()
            if checksum != info["sha256"]:
                raise ValueError(f"{input_path}: input hash differs from the manifest")
            expected = {
                "dataset": dataset,
                "injection_seed": seed,
                "sha256": checksum,
                "n_rows": spec["n_rows"],
                "n_anomalies": spec["n_anomalies"],
            }
            if not 0 < expected["n_anomalies"] < expected["n_rows"]:
                raise ValueError(f"{input_path}: metrics require both normal and anomaly rows")
            paired[dataset, seed] = {
                method: check_result(
                    root / "runs" / method / f"{dataset}_seed_{seed}.json",
                    {
                        **expected, "method": method,
                        "flowcad_commit": protocol["flowcad_commit"],
                        "training_seed": None if method == "qcad" else 201 + seed - 42,
                    },
                )
                for method in METHODS
            }
            flow_seed = paired[dataset, seed]["flowcad"]["training_seed"]
            mean_seed = paired[dataset, seed]["conditional_mean"]["training_seed"]
            if type(flow_seed) is not int or flow_seed != mean_seed:
                raise ValueError(f"{dataset}/{seed}: neural methods have different training seeds")
            for method in ("flowcad", "conditional_mean"):
                if paired[dataset, seed][method]["selected_epoch"] not in protocol["checkpoint_epochs"]:
                    raise ValueError(f"{dataset}/{seed}: selected epoch falls outside the declared grid")
    return datasets, paired


def summarize(datasets, paired):
    per_dataset = []
    for dataset, spec in datasets.items():
        seeds = spec["injection_seeds"]
        row = {"dataset": dataset, "n_seeds": len(seeds), "n_anomalies": spec["n_anomalies"]}
        for metric in METRICS:
            means = {
                method: mean(paired[dataset, seed][method][metric] for seed in seeds)
                for method in METHODS
            }
            for method, value in means.items():
                row[f"{method}_{metric}"] = value
            for comparator in METHODS[1:]:
                row[f"vs_{comparator}_{metric}"] = outcome(means["flowcad"], means[comparator])
        per_dataset.append(row)

    summary = []
    for comparator in METHODS[1:]:
        for metric in METRICS:
            decisions = [row[f"vs_{comparator}_{metric}"] for row in per_dataset]
            summary.append({
                "comparator": comparator,
                "metric": metric,
                "n_datasets": len(per_dataset),
                "n_trials": len(paired),
                "wins": decisions.count("win"),
                "ties": decisions.count("tie"),
                "losses": decisions.count("loss"),
            })
    return per_dataset, summary


def write_csv(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    datasets, paired = read_paired(args.root)
    per_dataset, summary = summarize(datasets, paired)
    write_csv(args.root / "per_dataset.csv", per_dataset)
    write_csv(args.root / "win_tie_loss.csv", summary)
    print(f"Validated {len(paired)} paired cases on {len(datasets)} datasets")
    for row in summary:
        print(f"{row['comparator']} {row['metric']}: {row['wins']}/{row['ties']}/{row['losses']}")


if __name__ == "__main__":
    main()
