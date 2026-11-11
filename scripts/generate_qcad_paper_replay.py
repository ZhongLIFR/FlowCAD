#!/usr/bin/env python3
"""Regenerate the fixed seven-seed benchmark from the pinned QCAD source tables."""

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import random
import re
import subprocess

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


from replay import MAIN_REFERENCE, MAIN_REPLAY, ROOT
SOURCE_COMMIT = "f3b887f3d4466607a9d7147e9715f6b0b27a5954"
BENCHMARK_SCRIPT = "Code/MultipleRunningAverage/AverageTest.py"
INJECTION_SCRIPT = "Code/Utilities/ContextualAnomalyInjectFinal.py"
DATASET_KEYS = (
    "abalone", "airfoil", "bodyfat", "boston", "concrete", "elnino", "energy",
    "fish", "forestfires", "gasemission", "heartfailure", "hepatitis", "indianliver",
    "maintenance", "parkinson", "power", "qsranking", "synmachine", "toxicity", "yacht",
)
TABLE_ANOMALIES = (100, 70, 20, 40, 50, 200, 50, 15, 50, 100, 30, 30, 30,
                   100, 100, 100, 40, 50, 50, 30)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def literal(block, field):
    match = re.search(rf"(?m)^{field}\s*=\s*(\[[^\]]*\]|\d+)", block)
    if match is None:
        raise ValueError(f"No {field} in benchmark configuration")
    return ast.literal_eval(match.group(1))


def benchmark_configs(source):
    blocks = re.split(r"(?m)^##DataSet\d+:[^\n]*\n", source)[1:]
    if len(blocks) != len(DATASET_KEYS):
        raise ValueError("Unexpected number of benchmark configurations")
    for key, anomalies, block in zip(DATASET_KEYS, TABLE_ANOMALIES, blocks):
        block = re.sub(r"(?m)^# ?", "", block)
        config = {name: literal(block, name) for name in (
            "AllCols", "ContextCols", "BehaveCols", "NumCols", "anomaly_value",
            "sample_value", "neighbour_value", "num_dataset",
        )}
        if config["anomaly_value"] != anomalies or config["sample_value"] != anomalies:
            raise ValueError(f"{key}: benchmark injection count conflicts with Table 3")
        yield key, config


def inject(raw, config, seed):
    # Match the released generator's random-number streams and draw order.
    if len(raw) > 20_000:
        raw = raw.sample(n=20_000, random_state=np.random.RandomState(seed))
    frame = raw[config["AllCols"]].reset_index(drop=True).copy()
    numeric = config["NumCols"]
    behavior = config["BehaveCols"]
    if numeric:
        frame[numeric] = MinMaxScaler().fit_transform(frame[numeric])
    rng = random.Random(seed)
    indices = rng.sample(range(1, len(frame)), config["anomaly_value"])
    # Match the benchmark generator's row/column loop and RNG draw order.
    positions = [frame.columns.get_loc(name) for name in behavior]
    for row in indices:
        for col in positions:
            frame.iat[row, col] += rng.choice([-1, 1]) * rng.uniform(0.1, 0.5)
    labels = np.zeros(len(frame), dtype=np.int8)
    labels[indices] = 1
    return frame.assign(ground_truth=labels)


def generate(repo, output):
    config_path = ROOT / "configs" / "datasets.json"
    reference_bytes = MAIN_REFERENCE.read_bytes()
    reference = json.loads(reference_bytes)
    flow = json.loads(config_path.read_text())["datasets"]
    benchmark = git(repo, "show", f"{SOURCE_COMMIT}:{BENCHMARK_SCRIPT}")
    injection = git(repo, "show", f"{SOURCE_COMMIT}:{INJECTION_SCRIPT}")
    raw_tree = git(repo, "ls-tree", "-r", "--name-only", SOURCE_COMMIT, "Data/RawData").decode()
    raw_paths = {Path(path).name.lower(): path for path in raw_tree.splitlines()}
    manifest = {
        "source_repository": "https://github.com/ZhongLIFR/QCAD",
        "source_commit": SOURCE_COMMIT,
        "benchmark_script": BENCHMARK_SCRIPT,
        "benchmark_script_sha256": digest(benchmark),
        "injection_script": INJECTION_SCRIPT,
        "injection_script_sha256": digest(injection),
        "flowcad_config_sha256": digest(config_path.read_bytes()),
        "protocol": reference["protocol"],
        "expected_datasets": list(DATASET_KEYS),
        "datasets": {},
    }
    jobs = sorted(
        benchmark_configs(benchmark.decode()),
        key=lambda item: reference["datasets"][item[0]]["n_rows"],
    )
    for key, config in jobs:
        spec = flow[key]
        raw_path = raw_paths[spec["file"].removesuffix("Gene.csv").lower() + ".csv"]
        data = git(repo, "show", f"{SOURCE_COMMIT}:{raw_path}")
        expected = reference["datasets"][key]
        if raw_path != expected["raw_path"] or digest(data) != expected["raw_sha256"]:
            raise ValueError(f"{key}: source file differs from the pinned replay")
        raw = pd.read_csv(io.BytesIO(data))
        complete = raw.dropna()
        n_rows = min(len(complete), 20_000)
        context, behavior = config["ContextCols"], config["BehaveCols"]
        if len(set(config["AllCols"])) != len(config["AllCols"]):
            raise ValueError(f"{key}: duplicate feature names")
        if set(context) & set(behavior) or not set(context + behavior) <= set(config["AllCols"]):
            raise ValueError(f"{key}: invalid context/behavior partition")
        seeds = list(expected["injection_seeds"])
        runs = len(seeds)
        info = {
            "display_name": spec["display_name"],
            "raw_path": raw_path,
            "raw_sha256": digest(data),
            "n_rows_before_downsampling": len(complete),
            "n_rows": n_rows,
            "n_anomalies": config["anomaly_value"],
            "benchmark_runs": runs,
            "benchmark_k": config["neighbour_value"],
            "injection_seeds": seeds,
            "context_columns": context,
            "behavior_columns": behavior,
            "categorical_columns": [name for name in spec["categorical_columns"] if name in context],
            "benchmark_columns": config["AllCols"],
            "scaled_columns": config["NumCols"],
            "unused_by_conditional_models": [name for name in config["AllCols"] if name not in context + behavior],
            "files": {},
        }
        dataset_dir = output / key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            frame = inject(complete, config, seed)
            if len(frame) != n_rows or int(frame["ground_truth"].sum()) != info["n_anomalies"]:
                raise ValueError(f"{key} seed {seed}: generated counts do not match the plan")
            target = dataset_dir / f"seed_{seed}.csv"
            temp = target.with_suffix(".tmp")
            content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
            sha = digest(content)
            if sha != expected["files"][str(seed)]["sha256"]:
                raise ValueError(f"{key} seed {seed}: regenerated CSV checksum does not match the pinned replay")
            temp.write_bytes(content)
            temp.replace(target)
            info["files"][str(seed)] = {"path": str(target.relative_to(output)), "sha256": sha}
        manifest["datasets"][key] = info
        print(f"{key}: verified {len(seeds)} injections", flush=True)
    if manifest != reference:
        raise ValueError("Regenerated metadata differs from the pinned replay")
    temp = output / "manifest.tmp"
    temp.write_bytes(reference_bytes)
    temp.replace(output / "manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qcad-repo", required=True, type=Path, help="QCAD clone containing the source commit")
    parser.add_argument("--output", type=Path, help="Destination for the generated tables and manifest")
    args = parser.parse_args()
    generate(args.qcad_repo, args.output or MAIN_REPLAY / "data")
