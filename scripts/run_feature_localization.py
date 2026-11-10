"""Measure single-coordinate attribution on in-range QCAD-derived perturbations."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import io
import json
import os
from pathlib import Path
import random
import statistics
import sys


from replay import REFERENCE, REPLAY, ROOT, read_manifest, verify_model
DATASETS = ("bodyfat", "heartfailure", "forestfires", "hepatitis", "indianliver", "energy")
SEEDS = (42, 43, 44)
METRICS = ("pr_auc", "roc_auc", "p_at_n", "top1", "mrr", "joint_top1_at_n")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def prepare_case(raw, spec, seed, output):
    import numpy as np
    from generate_qcad_paper_replay import digest, inject

    behavior = spec["behavior_columns"]
    if len(behavior) < 2 or not set(behavior).issubset(spec["scaled_columns"]):
        raise ValueError("Localization requires at least two min–max-scaled behavior columns")
    clean = inject(raw, {"AllCols": spec["benchmark_columns"], "NumCols": spec["scaled_columns"],
                         "BehaveCols": behavior, "anomaly_value": 0}, seed).drop(columns="ground_truth")
    changed = clean.copy()
    if len(clean) != spec["n_rows"]:
        raise ValueError("Clean raw data do not match the frozen benchmark row count")
    rng = random.Random(seed)
    truth = []
    for row in rng.sample(range(1, len(clean)), spec["n_anomalies"]):
        index = rng.randrange(len(behavior))
        feature = behavior[index]
        before = float(clean.at[row, feature])
        magnitude = rng.uniform(.1, .5)
        candidates = [before + sign * magnitude for sign in (-1, 1)
                      if 0 <= before + sign * magnitude <= 1]
        after = rng.choice(candidates)
        changed.at[row, feature] = after
        truth.append({"row": row, "behavior_index": index, "feature": feature,
                      "before": before, "after": after})
    differences = np.not_equal(clean.to_numpy(), changed.to_numpy())
    if differences.sum() != len(truth) or differences.sum(axis=1).max() != 1:
        raise ValueError("Single-coordinate injection check failed")
    for item in truth:
        if not differences[item["row"], clean.columns.get_loc(item["feature"])]:
            raise ValueError("Localization ground truth does not identify the changed coordinate")
    directory = output / "data"
    directory.mkdir(exist_ok=True)
    name = f"{spec['key']}_seed_{seed}"
    table_path = directory / f"{name}.csv"
    truth_path = directory / f"{name}.targets.json"
    temporary = table_path.with_suffix(".tmp")
    changed.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(table_path)
    write_json(truth_path, truth)
    return table_path, truth_path, digest(table_path.read_bytes()), digest(truth_path.read_bytes())


def localization(scores, contributions, truth, table_size):
    import numpy as np
    import pandas as pd
    from flowcad.metrics import evaluate_scores

    rows = np.array([target["row"] for target in truth])
    columns = np.array([target["behavior_index"] for target in truth])
    positive = np.zeros(table_size, dtype=int)
    positive[rows] = 1
    target_scores = contributions[rows, columns, None]
    all_scores = contributions[rows]
    ahead = (all_scores > target_scores).sum(axis=1)
    ties = (all_scores == target_scores).sum(axis=1)
    harmonic = np.concatenate(([0.], np.cumsum(1. / np.arange(1, contributions.shape[1] + 1))))
    top1 = (ahead == 0) / ties
    reciprocal = (harmonic[ahead + ties] - harmonic[ahead]) / ties
    # Same top-n convention as the shared row-level evaluator.
    top_rows = pd.Series(scores).sort_values(ascending=False).head(len(rows)).index
    detected = np.isin(rows, top_rows)
    return {**{key: value for key, value in evaluate_scores(positive, scores).items() if key != "n"},
            "top1": float(top1.mean()), "mrr": float(reciprocal.mean()),
            "joint_top1_at_n": float((top1 * detected).mean())}


def evaluate(job):
    spec, seed, release, path, truth_path, table_sha, truth_sha = job
    sys.path.insert(0, str(release))
    import numpy as np
    import torch
    from flowcad.checkpoints import select_checkpoint
    from flowcad.data import load_benchmark
    from generate_qcad_paper_replay import digest

    torch.set_num_threads(1)
    data = load_benchmark({**spec, "file": path.name}, path.parent, require_labels=False)
    if data.sha256 != table_sha or len(data.frame) != spec["n_rows"] or data.dropped_rows:
        raise ValueError("Model input differs from the planned feature-only table")
    training_seed = seed + 159
    selected = select_checkpoint(data.context, data.behavior, training_seed)
    if not np.allclose(selected["scores"], selected["contributions"].sum(axis=1), rtol=1e-6, atol=1e-7):
        raise ValueError("Flow-CAD feature contributions do not reconstruct the row scores")
    truth_bytes = truth_path.read_bytes()
    if digest(truth_bytes) != truth_sha:
        raise ValueError("Localization target sidecar differs from its frozen hash")
    truth = json.loads(truth_bytes)
    if len(truth) != spec["n_anomalies"]:
        raise ValueError("Unexpected localization target count")
    q = data.behavior.shape[1]
    evaluations = {
        "random_chance": {"top1": 1 / q, "mrr": sum(1 / j for j in range(1, q + 1)) / q},
        "flowcad_csm05": localization(selected["scores"], selected["contributions"], truth, len(data.frame)),
    }
    return {"dataset": spec["key"], "injection_seed": seed, "training_seed": training_seed,
            "n_rows": len(data.frame), "n_anomalies": len(truth), "n_behavior": q,
            "feature_sha256": table_sha, "target_sha256": truth_sha, "epoch": selected["epoch"],
            "candidates": selected["candidates"],
            "evaluations": evaluations}


def summarize(output, cases):
    records = []
    for row in sorted(cases, key=lambda item: (item["dataset"], item["injection_seed"])):
        keys = ("dataset", "injection_seed", "training_seed", "n_rows", "n_anomalies", "n_behavior",
                "feature_sha256", "target_sha256", "epoch")
        for method, metrics in row["evaluations"].items():
            records.append({**{k: row[k] for k in keys}, "method": method,
                            **{k: metrics.get(k, "") for k in METRICS}})
    write_csv(output / "per_run.csv", records)
    per_dataset = []
    for dataset in sorted({row["dataset"] for row in records}):
        for method in cases[0]["evaluations"]:
            group = [r for r in records if (r["dataset"], r["method"]) == (dataset, method)]
            means = {key: statistics.mean(r[key] for r in group) if group[0][key] != "" else ""
                     for key in METRICS}
            per_dataset.append({"dataset": dataset, "method": method, "n_injections": len(group),
                                "n_localization_targets": sum(r["n_anomalies"] for r in group), **means})
    write_csv(output / "per_dataset.csv", per_dataset)
    macro = []
    for method in cases[0]["evaluations"]:
        group = [row for row in per_dataset if row["method"] == method]
        macro.append({"method": method, "n_datasets": len(group),
                      **{key: statistics.mean(r[key] for r in group) if group[0][key] != "" else ""
                         for key in METRICS}})
    write_csv(output / "macro.csv", macro)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qcad-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=REPLAY / "feature_localization")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=3)
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    import pandas as pd
    from generate_qcad_paper_replay import SOURCE_COMMIT, digest, git

    args.output.mkdir(parents=True, exist_ok=True)
    release = ROOT
    commit = verify_model()
    manifest = read_manifest(args.manifest)
    if manifest["source_commit"] != SOURCE_COMMIT:
        raise ValueError("Unexpected original QCAD source commit")
    settings = {"flowcad_commit": commit,
                "qcad_commit": SOURCE_COMMIT, "runner_sha256": digest(Path(__file__).read_bytes()),
                "datasets": args.datasets.split(","), "injection_seeds": list(map(int, args.seeds.split(","))),
                "injection": "Original preprocessing and anomaly counts. Uniform random coordinate, magnitude U(.1,.5), uniformly choose a feasible sign so perturbed value is in [0,1].",
                "checkpoint": "Public full-table CSM protocol, training seed = injection seed + 159",
                "comparator": "uniform feature rank",
                "localization": "All injected rows; uniform tie-breaking expectation for top-1 and reciprocal rank."}
    settings_path = args.output / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Existing output folder uses different settings")
    write_json(settings_path, settings)
    jobs, records = [], []
    for key in settings["datasets"]:
        spec = {**manifest["datasets"][key], "key": key}
        source = git(args.qcad_repo, "show", f"{SOURCE_COMMIT}:{spec['raw_path']}")
        if digest(source) != spec["raw_sha256"]:
            raise ValueError(f"{key}: raw table hash mismatch")
        raw = pd.read_csv(io.BytesIO(source)).dropna()
        for seed in settings["injection_seeds"]:
            path, truth, sha, truth_sha = prepare_case(raw, spec, seed, args.output)
            result_path = args.output / f"{key}_seed_{seed}.json"
            if result_path.exists():
                record = json.loads(result_path.read_text())
                if (record["feature_sha256"], record["target_sha256"]) != (sha, truth_sha):
                    raise ValueError(f"{result_path} does not match the planned inputs")
                records.append(record)
            else:
                jobs.append((result_path, (spec, seed, release, path, truth, sha, truth_sha)))
    if records:
        summarize(args.output, records)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(evaluate, job): path for path, job in jobs}
        for future in as_completed(pending):
            result = future.result()
            write_json(pending[future], result)
            records.append(result)
            summarize(args.output, records)
            print(result["dataset"], result["injection_seed"], result["epoch"],
                  result["evaluations"], flush=True)


if __name__ == "__main__":
    main()
