"""Validate paired QCAD-paper reruns and summarize the completed methods."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


from replay import MAIN_MANIFEST as MANIFEST, MAIN_REPLAY as REPLAY, read_manifest


METRICS = ("pr_auc", "roc_auc", "p_at_n")
METHODS = ("Flow-CAD", "LUNAR", "DTE-C", "DTE-NP", "TCCM")
QCAD_METHODS = ("QCAD", "LoPAD", "ROCOD", "CAD", "IForest", "LOF", "k-NN", "SOD", "HBOS")
PAPER_STATS = dict(zip(METRICS, ("PR AUC", "ROC AUC", "P@n"), strict=True))
RANK_TOLERANCE = 1e-12


def read_records(path, method):
    if not path.exists():
        return []
    if path.is_dir():
        values = [json.loads(item.read_text()) for item in sorted(path.glob("*.json"))]
    else:
        with path.open(newline="") as stream:
            values = list(csv.DictReader(stream))
    return [row for row in values if "dataset" in row and row.get("method", method) == method]


def validate(records, expected, method):
    results = {}
    for row in records:
        key = (row["dataset"], int(row["injection_seed"]))
        if key not in expected or key in results:
            raise ValueError(f"{method}: unexpected or duplicate case {key}")
        sha256 = expected[key]
        if row["sha256"] != sha256:
            raise ValueError(f"{method}: input hash mismatch for {key}")
        values = {metric: float(row[metric]) for metric in METRICS}
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values.values()):
            raise ValueError(f"{method}: invalid metrics for {key}")
        results[key] = {"method": method, "dataset": key[0], "injection_seed": key[1],
                        "sha256": sha256, **values}
    return results


def write_csv(path, records):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=REPLAY)
    parser.add_argument("--source", action="append", required=True, metavar="METHOD=PATH",
                        help="Per-case CSV or directory of per-case JSON files, one source for each method")
    args = parser.parse_args()
    sources = {name: Path(path) for name, path in (entry.split("=", 1) for entry in args.source)}
    required_methods = set(METHODS + QCAD_METHODS)
    if len(args.source) != len(sources) or sources.keys() != required_methods:
        parser.error("Provide exactly one per-case source for each of the 14 benchmark methods")
    datasets = read_manifest(args.manifest)["datasets"]
    expected = {
        (dataset, int(seed)): spec["files"][str(seed)]["sha256"]
        for dataset, spec in datasets.items() for seed in spec["injection_seeds"]
    }
    results = {method: validate(read_records(path, method), expected, method)
               for method, path in sources.items()}
    status = {"expected_per_method": len(expected), "completed": {
        method: len(table) for method, table in results.items()
    }}
    status["complete"] = all(len(table) == len(expected) for table in results.values())
    args.output.mkdir(parents=True, exist_ok=True)
    if status["complete"]:
        summarize(args.output, datasets, results)
    temporary = args.output / "full_coverage.tmp"
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(args.output / "full_coverage.json")
    print(json.dumps(status), flush=True)


def summarize(output, datasets, results):
    per_case, per_dataset, macro = [], [], []
    for method, table in results.items():
        per_case.extend(table[key] for key in sorted(table))
        summaries = []
        for dataset, spec in datasets.items():
            rows = [table[dataset, int(seed)] for seed in spec["injection_seeds"]]
            summary = {"method": method, "dataset": dataset, "n_injections": len(rows)}
            for metric in METRICS:
                values = [row[metric] for row in rows]
                summary[f"{metric}_mean"] = statistics.mean(values)
                summary[f"{metric}_std"] = statistics.stdev(values)
            summaries.append(summary)
        per_dataset.extend(summaries)
    by_dataset = {dataset: [row for row in per_dataset if row["dataset"] == dataset]
                  for dataset in datasets}
    for rows in by_dataset.values():
        for metric in METRICS:
            key = f"{metric}_mean"
            ranks = descending_ranks([row[key] for row in rows])
            for row, rank in zip(rows, ranks, strict=True):
                row[f"{metric}_rank"] = rank
    for method in results:
        rows = [row for row in per_dataset if row["method"] == method]
        summary = {"method": method, "n_datasets": len(rows), "n_methods_ranked": len(results)}
        for metric in METRICS:
            summary[metric] = statistics.mean(row[f"{metric}_mean"] for row in rows)
            summary[f"{metric}_avg_rank"] = statistics.mean(row[f"{metric}_rank"] for row in rows)
        macro.append(summary)
    for metric in METRICS:
        key = f"{metric}_avg_rank"
        for row in macro:
            row[f"{metric}_place"] = average_rank(row[key], [peer[key] for peer in macro], False)
    write_csv(output / "full_paired_metrics.csv", per_case)
    write_csv(output / "full_per_dataset.csv", per_dataset)
    write_csv(output / "full_macro.csv", macro)
    write_main_table(output, datasets, per_dataset)


def descending_ranks(values):
    """Average ranks at full precision, grouping numerical ties within 1e-12."""
    order = sorted(range(len(values)), key=values.__getitem__, reverse=True)
    ranks = [None] * len(order)
    first = 0
    while first < len(order):
        stop = first + 1
        while stop < len(order) and values[order[first]] - values[order[stop]] <= RANK_TOLERANCE:
            stop += 1
        rank = (first + 1 + stop) / 2
        for index in order[first:stop]:
            ranks[index] = rank
        first = stop
    return ranks


def write_main_table(output, datasets, per_dataset):
    names = [spec["display_name"] for spec in datasets.values()]
    stat_names = tuple(PAPER_STATS.values())
    replay_methods = {row["method"] for row in per_dataset}
    baselines = QCAD_METHODS
    expected_methods = set(METHODS + QCAD_METHODS)
    if replay_methods != expected_methods:
        missing = sorted(expected_methods - replay_methods)
        extra = sorted(replay_methods - expected_methods)
        raise ValueError(f"Expected all 14 rerun methods; missing={missing}, extra={extra}")
    scores = {}
    for row in per_dataset:
        dataset = datasets[row["dataset"]]["display_name"]
        for metric, stat in PAPER_STATS.items():
            key = dataset, stat, row["method"]
            scores[key] = (row[f"{metric}_mean"], row[f"{metric}_std"], "replay")

    rows = []
    groups = {}
    method_order = ("Flow-CAD", *baselines, *METHODS[1:])
    for name in names:
        for stat in stat_names:
            means = [float(scores[name, stat, method][0]) for method in method_order]
            for method, rank in zip(method_order, descending_ranks(means), strict=True):
                value, std, source = scores[name, stat, method]
                row = {"dataset": name, "stat": stat, "method": method,
                       "mean": value, "std": std, "rank": rank,
                       "source": source}
                rows.append(row)
                groups.setdefault((stat, method), []).append(row)

    summary = []
    for (stat, method), group in groups.items():
        summary.append({"stat": stat, "method": method, "n_datasets": len(group),
                        "mean": statistics.mean(float(row["mean"]) for row in group),
                        "avg_rank": statistics.mean(row["rank"] for row in group),
                        "source": group[0]["source"]})
    for row in summary:
        peers = [other["avg_rank"] for other in summary if other["stat"] == row["stat"]]
        row["place"] = 1 + sum(rank < row["avg_rank"] for rank in peers)
    write_csv(output / "main_results.csv", rows)
    write_csv(output / "main_summary.csv", summary)


def average_rank(value, peers, descending):
    ahead = sum(peer > value if descending else peer < value for peer in peers)
    tied = sum(peer == value for peer in peers)
    return 1 + ahead + (tied - 1) / 2


if __name__ == "__main__":
    main()
