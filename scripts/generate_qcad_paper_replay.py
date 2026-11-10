#!/usr/bin/env python3
"""Regenerate the QCAD Table 3 injection settings on the historical raw files."""

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


from replay import REFERENCE, REPLAY, ROOT
SOURCE_COMMIT = "f3b887f3d4466607a9d7147e9715f6b0b27a5954"
BENCHMARK_SCRIPT = "Code/MultipleRunningAverage/AverageTest.py"
INJECTION_SCRIPT = "Code/Utilities/ContextualAnomalyInjectFinal.py"
DATASET_KEYS = (
    "abalone", "airfoil", "bodyfat", "boston", "concrete", "elnino", "energy",
    "fish", "forestfires", "gasemission", "heartfailure", "hepatitis", "indianliver",
    "maintenance", "parkinson", "power", "qsranking", "synmachine", "toxicity", "yacht",
)
TABLE_ROWS = (4177, 1503, 252, 506, 1030, 20000, 768, 157, 517, 7384, 299,
              615, 579, 11934, 5875, 9568, 475, 557, 908, 308)
TABLE_ANOMALIES = (100, 70, 20, 40, 50, 200, 50, 15, 50, 100, 30, 30, 30,
                   100, 100, 100, 40, 50, 50, 30)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def literal(block, field):
    match = re.search(rf"(?m)^{field}\s*=\s*(\[[^\]]*\]|\d+)", block)
    if match is None:
        raise ValueError(f"No {field} in historical benchmark configuration")
    return ast.literal_eval(match.group(1))


def benchmark_configs(source):
    blocks = re.split(r"(?m)^##DataSet\d+:[^\n]*\n", source)[1:]
    if len(blocks) != len(DATASET_KEYS):
        raise ValueError("Unexpected number of historical benchmark configurations")
    for key, rows, anomalies, block in zip(DATASET_KEYS, TABLE_ROWS, TABLE_ANOMALIES, blocks):
        block = re.sub(r"(?m)^# ?", "", block)
        config = {name: literal(block, name) for name in (
            "AllCols", "ContextCols", "BehaveCols", "NumCols", "anomaly_value",
            "sample_value", "neighbour_value", "num_dataset",
        )}
        if config["anomaly_value"] != anomalies or config["sample_value"] != anomalies:
            raise ValueError(f"{key}: benchmark injection count conflicts with Table 3")
        yield key, rows, config


def inject(raw, config, seed):
    # The historical generator downsamples with NumPy's legacy seeded RNG,
    # then uses Python's random module for row selection and every feature shift.
    if len(raw) > 20_000:
        raw = raw.sample(n=20_000, random_state=np.random.RandomState(seed))
    frame = raw[config["AllCols"]].reset_index(drop=True).copy()
    numeric = config["NumCols"]
    behavior = config["BehaveCols"]
    if numeric:
        frame[numeric] = MinMaxScaler().fit_transform(frame[numeric])
    rng = random.Random(seed)
    indices = rng.sample(range(1, len(frame)), config["anomaly_value"])
    # Preserve the historical generator's row/column loop and RNG draw order.
    positions = [frame.columns.get_loc(name) for name in behavior]
    for row in indices:
        for col in positions:
            frame.iat[row, col] += rng.choice([-1, 1]) * rng.uniform(0.1, 0.5)
    labels = np.zeros(len(frame), dtype=np.int8)
    labels[indices] = 1
    return frame.assign(ground_truth=labels)


def verify_abalone(repo, frame):
    published_path = "Data/TempFiles/abaloneGene.csv"
    ref_commit = "c8a93c86244078bec51e7b9b4460f135acda78e5"
    data = git(repo, "show", f"{ref_commit}:{published_path}")
    reference = pd.read_csv(io.BytesIO(data))[frame.columns]
    np.testing.assert_array_equal(frame["ground_truth"], reference["ground_truth"])
    np.testing.assert_allclose(frame.to_numpy(), reference.to_numpy(), rtol=0, atol=1e-12)
    return {"commit": ref_commit, "path": published_path, "sha256": digest(data),
            "verification": "identical label vector and all values agree within 1e-12"}


def generate(repo, output):
    config_path = ROOT / "configs" / "datasets.json"
    reference = json.loads(REFERENCE.read_bytes())
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
        "protocol": "historical benchmark configuration and generator; independent signed uniform shifts per injected row and behavioral feature",
        "expected_datasets": list(DATASET_KEYS),
        "datasets": {},
    }
    jobs = sorted(benchmark_configs(benchmark.decode()), key=lambda item: item[1])
    for key, published_rows, config in jobs:
        spec = flow[key]
        raw_path = raw_paths[spec["file"].removesuffix("Gene.csv").lower() + ".csv"]
        data = git(repo, "show", f"{SOURCE_COMMIT}:{raw_path}")
        expected = reference["datasets"][key]
        if raw_path != expected["raw_path"] or digest(data) != expected["raw_sha256"]:
            raise ValueError(f"{key}: historical raw file differs from the published replay")
        raw = pd.read_csv(io.BytesIO(data))
        complete = raw.dropna()
        n_rows = min(len(complete), 20_000)
        context, behavior = config["ContextCols"], config["BehaveCols"]
        if len(set(config["AllCols"])) != len(config["AllCols"]):
            raise ValueError(f"{key}: duplicate historical feature names")
        if set(context) & set(behavior) or not set(context + behavior) <= set(config["AllCols"]):
            raise ValueError(f"{key}: invalid context/behavior partition")
        seeds = list(range(42, 42 + config["num_dataset"]))
        info = {
            "display_name": spec["display_name"],
            "raw_path": raw_path,
            "raw_sha256": digest(data),
            "raw_rows": len(raw),
            "n_rows_before_downsampling": len(complete),
            "published_n_rows": published_rows,
            "n_rows": n_rows,
            "n_anomalies": config["anomaly_value"],
            "benchmark_runs": config["num_dataset"],
            "benchmark_k": config["neighbour_value"],
            "injection_seeds": seeds,
            "context_columns": context,
            "behavior_columns": behavior,
            "categorical_columns": [name for name in spec["categorical_columns"] if name in context],
            "benchmark_columns": config["AllCols"],
            "scaled_columns": config["NumCols"],
            "unused_by_conditional_models": [name for name in config["AllCols"] if name not in context + behavior],
            "flowcad_context_columns_match": context == spec["context_columns"],
            "flowcad_behavior_columns_match": behavior == spec["behavior_columns"],
            "files": {},
        }
        if n_rows != published_rows:
            info["sample_count_note"] = (
                "The released source and missing-value processing yield 589 rows; Appendix B records removing 26 of 615."
                if key == "hepatitis" else
                "The released historical raw CSV has 159 complete rows; no rows were removed to force the published count of 157."
                if key == "fish" else "The historical raw file and released preprocessing yield a different count from Table 3."
            )
        dataset_dir = output / key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            frame = inject(complete, config, seed)
            if len(frame) != n_rows or int(frame["ground_truth"].sum()) != info["n_anomalies"]:
                raise ValueError(f"{key} seed {seed}: generated counts do not match the plan")
            if key == "abalone" and seed == 42:
                info["abalone_reference_check"] = verify_abalone(repo, frame)
            target = dataset_dir / f"seed_{seed}.csv"
            temp = target.with_suffix(".tmp")
            content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
            sha = digest(content)
            if sha != expected["files"][str(seed)]["sha256"]:
                raise ValueError(f"{key} seed {seed}: regenerated CSV checksum does not match the published replay")
            temp.write_bytes(content)
            temp.replace(target)
            info["files"][str(seed)] = {"path": str(target.relative_to(output)), "sha256": sha}
        manifest["datasets"][key] = info
        print(f"{key}: verified {len(seeds)} injections", flush=True)
    if manifest != reference:
        raise ValueError("Regenerated metadata differs from the published replay")
    temp = output / "manifest.tmp"
    temp.write_bytes(REFERENCE.read_bytes())
    temp.replace(output / "manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qcad-repo", required=True, type=Path, help="QCAD clone containing the source commit")
    parser.add_argument("--output", type=Path, default=REPLAY / "data")
    args = parser.parse_args()
    generate(args.qcad_repo, args.output)
