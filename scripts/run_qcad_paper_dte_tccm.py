"""Replay official point detectors on the unlabeled QCAD benchmark snapshots."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback


from replay import MAIN_MANIFEST as MANIFEST, MAIN_REPLAY as REPLAY, ROOT, read_manifest
METRICS = ("pr_auc", "roc_auc", "p_at_n")
METHODS = ("DTE-NP", "TCCM", "DTE-C")
SOURCES = {
    "dte": {
        "url": "https://github.com/vicliv/DTE",
        "commit": "2c205204cd9bd21d1ffd3dd180a9f4a94bc6b279",
        "files": ("diffusion/dte.py", "diffusion/non_param_dte.py"),
    },
    "tccm": {
        "url": "https://github.com/ZhongLIFR/TCCM-NIPS",
        "commit": "363cfa61fd9e786bcb0b905241106dd4e7f52e6f",
        "files": ("FMAD/FlowMatchingAD.py", "FMAD/functions.py"),
    },
}
SETTINGS = {
    "DTE-C": {"epochs": 400, "batch_size": 64, "lr": 1e-4, "weight_decay": 5e-4,
              "hidden_size": [256, 512, 256], "T": 400, "num_bins": 7},
    "DTE-NP": {"K": 5, "T": 1000},
    "TCCM": {"epochs": 100, "batch_size": 64, "learning_rate": 0.001},
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_provenance(paths):
    provenance = {}
    for name, path in paths.items():
        spec = SOURCES[name]
        commit = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
        if commit != spec["commit"]:
            raise ValueError(f"Expected {name} commit {spec['commit']}, got {commit}")
        files = {}
        for file in spec["files"]:
            committed = subprocess.check_output(["git", "-C", path, "show", f"{commit}:{file}"])
            current = (path / file).read_bytes()
            if current != committed:
                raise ValueError(f"Upstream source differs from commit: {path / file}")
            files[file] = sha256(current)
        provenance[name] = {"url": spec["url"], "commit": commit, "files_sha256": files}
    return provenance


def features_from(raw, spec):
    import numpy as np
    import pandas as pd

    columns = spec["context_columns"] + spec["behavior_columns"]
    frame = pd.read_csv(io.BytesIO(raw), usecols=columns)[columns]
    for column in spec["categorical_columns"]:
        if column in frame and not pd.api.types.is_numeric_dtype(frame[column]):
            if frame[column].isna().any():
                raise ValueError(f"Missing categorical values in {column}")
            frame[column] = pd.Categorical(frame[column]).codes
    values = frame.to_numpy(dtype=np.float64)
    if len(values) != spec["n_rows"] or not np.isfinite(values).all():
        raise ValueError("Unexpected row count or non-finite features")
    return values


def batched_dte_np(model, features):
    """Batch the pinned estimator's tree queries and inverse-gamma posterior."""
    import numpy as np
    from scipy.stats import invgamma
    import torch

    mean_distances = model.tree.query(features.astype(np.float32), k=model.K)[0].mean(axis=1)
    variance = (1.0 - model.alphas_cumprod).numpy()
    shape = 0.5 * features.shape[1] - 1.0
    scores = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), 256):
        block = slice(start, start + 256)
        scale = mean_distances[None, block] / 2.0
        log_pdf = invgamma.logpdf(variance[:, None], a=shape, loc=0, scale=scale)
        density = torch.from_numpy(log_pdf).float()
        density = density - density.logsumexp(0, keepdim=True)
        scores[block] = density.exp().argmax(0).float().numpy()
    return scores


def fit_score(method, dataset, seed, features):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    settings = SETTINGS[method]
    if method == "DTE-NP":
        from diffusion.non_param_dte import DTENonParametric

        model = DTENonParametric(seed=seed, **settings).fit(features)
        scores = batched_dte_np(model, features)
        if dataset == "fish" and seed == 201:
            reference = model.predict_score(features)
            if not np.array_equal(reference, scores):
                raise ValueError("Batched DTE-NP disagrees with the reference implementation")
        return scores
    if method == "DTE-C":
        from diffusion.dte import DTECategorical

        model = DTECategorical(seed=seed, device=torch.device("cpu"), **settings).fit(features)
        with torch.no_grad():
            return model.predict_score(features)

    from FMAD.FlowMatchingAD import TCCM

    model = TCCM(n_features=features.shape[1], **settings)
    model.fit(features)
    return model.decision_function(features)


def evaluate(job):
    method, dataset, injection_seed, spec, path, expected_hash, run_hash, paths = job
    sys.path[:0] = [str(paths["dte"]), str(paths["tccm"]), str(ROOT)]
    import pandas as pd
    import sklearn
    import torch
    from flowcad.metrics import evaluate_scores

    torch.set_num_threads(1)
    raw = path.read_bytes()
    if sha256(raw) != expected_hash:
        raise ValueError(f"Unexpected input SHA-256: {path}")
    features = features_from(raw, spec)
    model_seed = 201 + injection_seed - 42
    start = time.perf_counter()
    scores = fit_score(method, dataset, model_seed, features)
    seconds = time.perf_counter() - start
    labels = pd.read_csv(io.BytesIO(raw), usecols=["ground_truth"])["ground_truth"].to_numpy()
    metrics = evaluate_scores(labels, scores)
    if metrics["n"] != spec["n_anomalies"]:
        raise ValueError(f"Unexpected number of anomalies: {dataset}/{injection_seed}")
    return {
        "method": method, "dataset": dataset, "injection_seed": injection_seed,
        "sha256": expected_hash, **{name: metrics[name] for name in METRICS},
        "model_seed": model_seed, "n_rows": len(features), "n_anomalies": metrics["n"],
        "n_features": features.shape[1], "fit_score_seconds": seconds,
        "run_sha256": run_hash, "torch_version": torch.__version__, "sklearn_version": sklearn.__version__,
    }


def write_summary(output, records):
    ordered = sorted(records, key=lambda record: (record["method"], record["dataset"], record["injection_seed"]))
    temporary = output / "per_run.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    temporary.replace(output / "per_run.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dte-repo", required=True, type=Path)
    parser.add_argument("--tccm-repo", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=REPLAY / "baselines_dte_tccm")
    parser.add_argument("--datasets", default="all", help="all or comma-separated dataset keys")
    parser.add_argument("--injection-seeds", default="all", help="all or comma-separated injection seeds")
    parser.add_argument("--methods", default="all", help="all or comma-separated method names")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = "1"

    paths = {"dte": args.dte_repo.resolve(), "tccm": args.tccm_repo.resolve()}
    manifest = args.manifest.read_bytes()
    provenance = {
        "manifest_sha256": sha256(manifest), "runner_sha256": sha256(Path(__file__).read_bytes()),
        "metrics_sha256": sha256((ROOT / "flowcad/metrics.py").read_bytes()),
        "upstream": source_provenance(paths), "settings": SETTINGS,
        "feature_preprocessing": "Raw C+B columns; no scaling; numeric values kept; text categories ordinal-encoded.",
        "fit": "Unlabeled full table.",
        "score": "Same full table. DTE-NP includes the query among neighbors as in the source. Labels read after scoring.",
        "model_seed": "201 + injection_seed - 42",
        "metrics": "Trapezoidal PR AUC, ROC AUC, P@n with n equal to the number of anomalies.",
    }
    run_hash = sha256(json.dumps(provenance, sort_keys=True).encode())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text())["run_sha256"] != run_hash:
        raise ValueError(f"Output directory has a different protocol: {output}")
    write_json(protocol_path, {"run_sha256": run_hash, **provenance})

    inputs = read_manifest(args.manifest)["datasets"]
    datasets = (sorted(inputs, key=lambda name: (inputs[name]["n_rows"], name)) if args.datasets == "all"
                else args.datasets.split(","))
    methods = METHODS if args.methods == "all" else args.methods.split(",")
    if set(methods) - set(METHODS):
        parser.error("Unknown method")
    records, jobs = [], {}
    for dataset in datasets:
        spec = inputs[dataset]
        seeds = (spec["injection_seeds"] if args.injection_seeds == "all" else
                 [int(seed) for seed in args.injection_seeds.split(",")])
        for seed in seeds:
            entry = spec["files"][str(seed)]
            path = args.manifest.parent / entry["path"]
            for method in methods:
                target = output / "records" / method / dataset / f"seed_{seed}.json"
                if target.exists():
                    record = json.loads(target.read_text())
                    key = (record["method"], record["dataset"], record["injection_seed"], record["sha256"],
                           record["run_sha256"])
                    if key != (method, dataset, seed, entry["sha256"], run_hash):
                        raise ValueError(f"Mismatched prior result: {target}")
                    records.append(record)
                else:
                    jobs[target] = (method, dataset, seed, spec, path, entry["sha256"], run_hash, paths)

    print(f"Completed records: {len(records)}; pending: {len(jobs)}", flush=True)
    if records:
        write_summary(output, records)
    failed = 0
    prioritized = sorted(jobs.items(), key=lambda item: item[1][0] != "DTE-NP")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate, job): target for target, job in prioritized}
        for future in as_completed(futures):
            target = futures[future]
            try:
                record = future.result()
            except Exception:
                failed += 1
                print(f"Failed case {target}:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            write_json(target, record)
            records.append(record)
            write_summary(output, records)
            print(json.dumps(record), flush=True)
    print(f"Completed records: {len(records)}; failed: {failed}", flush=True)
    return bool(failed)


if __name__ == "__main__":
    sys.exit(main())
