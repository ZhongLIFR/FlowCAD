"""Run PyOD 2.0.3 LUNAR on matched QCAD-paper replay tables.

The historical Flow-CAD environment pins PyOD 2.0.3 and the related TCCM
benchmark instantiates LUNAR with defaults. Fit and score on the same unlabeled
table using the package's decision_scores_ interface.

Install the pinned environment with python -m pip install -r requirements-baselines.txt.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
from importlib.metadata import distributions, version
from importlib.util import find_spec
import json
import os
from pathlib import Path
import sys
import time


from replay import MANIFEST, REPLAY, ROOT, read_manifest
PYOD_VERSION = "2.0.3"
RELATED_PROTOCOL = "https://github.com/ZhongLIFR/TCCM-NIPS/blob/363cfa61fd9e786bcb0b905241106dd4e7f52e6f/FullExperiments.py"
METRICS = ("pr_auc", "roc_auc", "p_at_n")
PROTOCOL = {
    "configuration": "Unmodified PyOD LUNAR() defaults",
    "model_type": "WEIGHT",
    "neighbors": 5,
    "epochs": 200,
    "validation_fraction": 0.10,
    "negative_sampling": "MIXED",
    "input": "manifest context and behavior columns, with Flow-CAD categorical one-hot encoding",
    "scoring": "decision_scores_ for the entire input table",
    "labels": "Ground truth is read after scoring and used only to evaluate.",
}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(job):
    dataset, seed, spec, path, expected_hash, release, dependencies, run_hash = job
    sys.path.insert(0, str(release))
    if dependencies is not None:
        sys.path.insert(0, str(dependencies))
    import numpy as np
    import pandas as pd
    from pyod.models.lunar import LUNAR
    import sklearn
    import torch
    from flowcad.data import load_benchmark
    from flowcad.metrics import evaluate_scores

    torch.set_num_threads(1)
    data = load_benchmark({**spec, "file": path.name}, path.parent, require_labels=False)
    if data.sha256 != expected_hash or len(data.frame) != spec["n_rows"] or data.dropped_rows:
        raise ValueError(f"Unexpected input rows or SHA-256: {dataset}/{seed}")
    features = np.column_stack([data.context, data.behavior])
    training_seed = 201 + seed - 42
    started = time.perf_counter()
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)
    detector = LUNAR().fit(features)
    scores = detector.decision_scores_
    elapsed = time.perf_counter() - started
    labels = pd.read_csv(path, usecols=["ground_truth"]).loc[data.frame.index, "ground_truth"]
    metrics = evaluate_scores(labels.to_numpy(), scores)
    if metrics["n"] != spec["n_anomalies"]:
        raise ValueError(f"Unexpected number of anomalies: {dataset}/{seed}")
    return {
        "method": "LUNAR", "dataset": dataset, "injection_seed": seed, "training_seed": training_seed,
        "n_rows": len(features), "n_anomalies": metrics["n"], "n_features": features.shape[1],
        "sha256": data.sha256, "pyod_version": PYOD_VERSION, "run_sha256": run_hash,
        **{key: metrics[key] for key in METRICS}, "fit_score_seconds": elapsed,
        "torch_version": torch.__version__, "sklearn_version": sklearn.__version__,
    }


def write_json(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_summary(output, records):
    rows = sorted(records, key=lambda row: (row["dataset"], row["injection_seed"]))
    temporary = output / "summary.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "summary.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--dependencies", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=REPLAY / "lunar")
    parser.add_argument("--datasets", default="all", help="all or comma-separated dataset keys")
    parser.add_argument("--injection-seeds", default="all", help="all or comma-separated seeds")
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = "1"

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    release = ROOT
    code_hashes = {name: file_hash(release / "flowcad" / name) for name in ("data.py", "metrics.py")}
    dependencies = args.dependencies.resolve() if args.dependencies else None
    expected = {"pyod": PYOD_VERSION, "numba": "0.60.0", "llvmlite": "0.43.0"}
    if dependencies is None:
        installed = {name: version(name) for name in expected}
        package = Path(find_spec("pyod").origin).parent
    else:
        installed = {dist.metadata["Name"].lower(): dist.version
                     for dist in distributions(path=[str(dependencies)])}
        package = dependencies / "pyod"
    if any(installed.get(name) != required for name, required in expected.items()):
        raise ValueError("Install the versions in requirements-baselines.txt")
    lunar_source = package / "models" / "lunar.py"
    provenance = {"dependencies": expected, "pyod_lunar_sha256": file_hash(lunar_source),
                  "related_protocol": RELATED_PROTOCOL, "runner_sha256": file_hash(Path(__file__)),
                  "common_code_sha256": code_hashes, "settings": PROTOCOL}
    run_hash = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    write_json(output / "protocol.json", {"run_sha256": run_hash, **provenance})
    inputs = read_manifest(args.manifest)["datasets"]
    datasets = sorted(inputs, key=lambda name: (inputs[name]["n_rows"], name))
    if args.datasets != "all":
        datasets = args.datasets.split(",")
    records, pending_jobs = [], []
    for dataset in datasets:
        spec = inputs[dataset]
        seeds = spec["injection_seeds"] if args.injection_seeds == "all" else [
            int(seed) for seed in args.injection_seeds.split(",")
        ]
        for seed in seeds:
            entry = spec["files"][str(seed)]
            source = args.manifest.parent / entry["path"]
            record_path = output / f"{dataset}_seed_{seed}.json"
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if record["sha256"] != entry["sha256"] or record["run_sha256"] != run_hash:
                    raise ValueError(f"Previous result has different data or protocol: {record_path}")
                records.append(record)
            else:
                job = (dataset, seed, spec, source, entry["sha256"], release, dependencies, run_hash)
                pending_jobs.append((record_path, job))
    if records:
        write_summary(output, records)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(evaluate, job): path for path, job in pending_jobs}
        for future in as_completed(pending):
            record = future.result()
            write_json(pending[future], record)
            records.append(record)
            write_summary(output, records)
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
