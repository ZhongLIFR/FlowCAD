"""Run three paired detectors on the frozen within-context swap benchmark."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import fcntl
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import time

from replay import ROOT, verify_model


METHODS = ("flowcad", "qcad", "conditional_mean")
METRICS = ("pr_auc", "roc_auc", "p_at_n")
DEFAULT_MANIFEST = ROOT / "outputs" / "qcad_context_swap" / "data" / "manifest.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def evaluate(job):
    method, dataset, seed, spec, path, sha256, revision, target = job
    training_seed = 201 + seed - 42
    started = time.perf_counter()
    if method == "flowcad":
        from run_qcad_paper_flowcad import evaluate as run_flowcad

        result = run_flowcad((ROOT, revision, dataset, seed, spec, path, sha256, None))
    elif method == "conditional_mean":
        from run_qcad_conditional_mean import evaluate as run_conditional_mean

        with tempfile.TemporaryDirectory(prefix="flowcad-mean-") as temporary:
            result = run_conditional_mean(
                (dataset, spec, seed, path, sha256, training_seed, Path(temporary) / "scores.npz")
            )
    else:
        from run_paired_qcad import run_one

        with tempfile.TemporaryDirectory(prefix="flowcad-qcad-") as temporary:
            result = run_one((dataset, seed, path, {**spec, "sha256": sha256},
                              spec, "context_swap", Path(temporary)))
        result.update(n_rows=result["rows"], n_anomalies=result["n"],
                      selected_epoch=None, training_seed=None)
    result.update(method=method, flowcad_commit=revision,
                  wall_seconds=time.perf_counter() - started)
    if any(result[name] != expected for name, expected in (
        ("dataset", dataset), ("injection_seed", seed), ("sha256", sha256),
        ("n_rows", spec["n_rows"]), ("n_anomalies", spec["n_anomalies"]),
    )):
        raise ValueError(f"Detector returned a different input case: {dataset}/{seed}/{method}")
    write_json(target, result)
    return result


def read_completed(target, method, dataset, seed, spec, sha256, revision):
    if not target.exists():
        return None
    result = json.loads(target.read_text())
    expected = {
        "method": method, "dataset": dataset, "injection_seed": seed,
        "sha256": sha256, "flowcad_commit": revision, "n_rows": spec["n_rows"],
        "n_anomalies": spec["n_anomalies"],
        "training_seed": None if method == "qcad" else 201 + seed - 42,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Previous result belongs to a different case: {target}")
    if any(not 0.0 <= result[metric] <= 1.0 for metric in METRICS):
        raise ValueError(f"Previous result has invalid metrics: {target}")
    return result


def write_summary(output):
    records = [json.loads(path.read_text()) for method in METHODS
               for path in (output / method).glob("*.json")]
    records.sort(key=lambda row: (row["dataset"], row["injection_seed"], row["method"]))
    fields = ("dataset", "injection_seed", "method", "sha256", "n_rows", "n_anomalies",
              "training_seed", "selected_epoch", *METRICS, "wall_seconds")
    temporary = output / "per_run.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(output / "per_run.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dataset", action="append", help="Restrict a smoke run to dataset keys")
    parser.add_argument("--seed", action="append", type=int, help="Restrict a smoke run to injection seeds")
    parser.add_argument("--method", action="append", choices=METHODS)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    manifest = json.loads(args.manifest.read_bytes())
    datasets = manifest["datasets"]
    if sorted(datasets) != sorted(manifest["expected_datasets"]):
        raise ValueError("Manifest dataset coverage differs from the frozen protocol")
    if args.dataset and set(args.dataset) - datasets.keys():
        parser.error("--dataset contains unknown or ineligible keys")
    revision = verify_model()
    output = args.output or args.manifest.parent.parent / "runs"
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "run.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dependencies = ("run_qcad_paper_flowcad.py", "run_qcad_conditional_mean.py",
                    "run_paired_qcad.py", "replay.py")
    protocol = {
        "manifest_sha256": digest(args.manifest), "runner_sha256": digest(Path(__file__)),
        "implementation_sha256": {name: digest(Path(__file__).parent / name) for name in dependencies},
        "flowcad_commit": revision,
        "versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch")},
        "methods": list(METHODS),
        "checkpoint_epochs": [50, 100, 200, 500, 1000, 2000, 5000, 10000],
        "selection": "Independent label-free full-table 5% CSM for Flow-CAD and conditional-mean MLP",
        "training_seeds": "201 + injection_seed - 42",
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError(f"Run implementation or inputs changed: {protocol_path}")
    write_json(protocol_path, protocol)
    jobs, completed = [], 0
    for method in METHODS:
        (output / method).mkdir(exist_ok=True)
    for dataset in args.dataset or sorted(datasets):
        spec = datasets[dataset]
        if len(spec["behavior_columns"]) < 2:
            raise ValueError(f"Joint-dependence diagnostic requires multiple behavioral columns: {dataset}")
        seeds = args.seed or spec["injection_seeds"]
        for seed in seeds:
            info = spec["files"][str(seed)]
            path = args.manifest.parent / info["path"]
            if digest(path) != info["sha256"]:
                raise ValueError(f"Frozen input changed: {path}")
            for method in args.method or METHODS:
                target = output / method / f"{dataset}_seed_{seed}.json"
                if read_completed(target, method, dataset, seed, spec, info["sha256"], revision):
                    completed += 1
                else:
                    jobs.append((method, dataset, seed, spec, path, info["sha256"], revision, target))
    jobs.sort(key=lambda job: (-job[3]["n_rows"], job[0], job[1], job[2]))
    expected = completed + len(jobs)
    write_summary(output)
    print(json.dumps({"expected": expected, "completed": completed, "workers": args.workers}), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            write_summary(output)
            print(json.dumps({"completed": completed, "expected": expected,
                              **{key: result[key] for key in (
                                  "dataset", "injection_seed", "method", "selected_epoch",
                                  *METRICS, "wall_seconds")}}), flush=True)


if __name__ == "__main__":
    main()
