"""Run a conditional-mean MLP control on every frozen paper-replay table."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import time

from replay import MANIFEST, ROOT, read_manifest, verify_model

METRICS = ("pr_auc", "roc_auc", "p_at_n")


def fit_and_select(context, behavior, seed, checkpoints, np, torch):
    checkpoints.set_seed(seed)
    loader = torch.utils.data.DataLoader(
        checkpoints._BatchedDataset(context, behavior),
        batch_size=128, shuffle=True, drop_last=False, collate_fn=lambda batch: batch,
    )
    model = checkpoints.ConditionalVectorField(context.shape[1], behavior.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    c = torch.as_tensor(context)
    b = torch.as_tensor(behavior)
    median = np.median(behavior, axis=0)
    scale = torch.as_tensor(np.maximum(np.median(np.abs(behavior - median), axis=0), 1e-6))
    zero_state = torch.zeros_like(b)
    zero_time = torch.zeros((len(b), 1), dtype=b.dtype)
    k = min(len(b) - 2, max(2, math.floor(checkpoints.TAIL_FRACTION * len(b))))
    selected_epoch, best_csm, best_scores = None, None, None
    candidates = []

    for epoch in range(1, checkpoints.CANDIDATE_EPOCHS[-1] + 1):
        model.train()
        for batch_context, batch_behavior in loader:
            # Advance the same RNG stream as Flow-CAD so shuffle orders match.
            # The samples never enter the direct-regression model or its loss.
            torch.randn_like(batch_behavior)
            torch.rand((len(batch_behavior), 1), device="cpu")
            prediction = model(zero_state[:len(batch_behavior)], batch_context, zero_time[:len(batch_behavior)])
            loss = torch.mean((prediction - batch_behavior) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch not in checkpoints.CANDIDATE_EPOCHS:
            continue
        model.eval()
        with torch.no_grad():
            prediction = model(zero_state, c, zero_time)
            scores = ((prediction - b).abs() / scale).clamp(max=10.0).sum(dim=1).numpy().astype(np.float64)
        csm = checkpoints._contrast_score_margin(scores, k)
        candidates.append(csm)
        if csm is not None and (best_csm is None or csm > best_csm):
            selected_epoch, best_csm, best_scores = epoch, csm, scores
    if best_scores is None:
        raise ValueError("All conditional-mean checkpoints have undefined CSM")
    return selected_epoch, best_csm, best_scores, candidates


def evaluate(job):
    dataset, spec, seed, input_path, sha, training_seed, target = job
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import pandas as pd
    import torch
    from flowcad import checkpoints
    from flowcad.data import load_benchmark
    from flowcad.metrics import evaluate_scores

    torch.set_num_threads(1)
    data = load_benchmark({**spec, "file": input_path.name}, input_path.parent, require_labels=False)
    if data.sha256 != sha or len(data.frame) != spec["n_rows"] or data.dropped_rows:
        raise ValueError(f"Input differs from manifest: {dataset}/{seed}")
    started = time.perf_counter()
    epoch, csm, scores, candidates = fit_and_select(data.context, data.behavior, training_seed, checkpoints, np, torch)
    labels = pd.read_csv(input_path, usecols=["ground_truth"]).loc[data.frame.index, "ground_truth"].to_numpy()
    metrics = evaluate_scores(labels, scores)
    if metrics["n"] != spec["n_anomalies"]:
        raise ValueError(f"Label count differs from manifest: {dataset}/{seed}")
    result = {
        "dataset": dataset, "injection_seed": seed, "sha256": sha, "training_seed": training_seed,
        "selected_epoch": epoch, "csm": csm, "n_rows": len(scores), "n_anomalies": metrics["n"],
        **{metric: metrics[metric] for metric in METRICS}, "fit_seconds": time.perf_counter() - started,
    }
    temporary = target.with_suffix(".tmp")
    with temporary.open("wb") as output:
        np.savez(output, **result, scores=scores, candidate_epochs=checkpoints.CANDIDATE_EPOCHS,
                 candidate_csm=np.asarray(candidates, dtype=np.float64))
    temporary.replace(target)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--flow-records", type=Path, help="Directory of CSM-selected Flow-CAD JSON records")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dataset", action="append", help="Restrict a smoke run to these datasets")
    parser.add_argument("--seed", type=int, action="append", help="Restrict a smoke run to these injection seeds")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    import numpy as np

    revision = verify_model()
    datasets = read_manifest(args.manifest)["datasets"]
    replay_dir = args.manifest.parent.parent
    flow_records = args.flow_records or replay_dir / "flowcad_csm05"
    output = args.output or replay_dir / "qcad_conditional_mean"
    records = output / "records"
    records.mkdir(parents=True, exist_ok=True)
    settings = {
        "input_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "flowcad_commit": revision,
        "versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch")},
        "model": "Same context encoder and MLP widths as Flow-CAD; state/time inputs fixed at zero",
        "training": "Direct behavioral MSE; Adam 0.001, batch 128, seed and shuffles matched to Flow-CAD",
        "score": "L1 absolute residual from conditional mean, full-table MAD floor 1e-6, per-coordinate cap 10",
        "selection": "Same label-free 5% CSM and candidate epochs as Flow-CAD, independently selected",
    }
    protocol = output / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != settings:
        raise ValueError(f"Run protocol changed: {protocol}")
    protocol.write_text(json.dumps(settings, indent=2) + "\n")

    jobs, completed = [], 0
    for dataset in args.dataset or datasets:
        spec = datasets[dataset]
        for seed in args.seed or spec["injection_seeds"]:
            input_spec = spec["files"][str(seed)]
            reference_name = f"{dataset}_seed_{seed}"
            reference = json.loads((flow_records / f"{reference_name}.json").read_text())
            if (reference["sha256"] != input_spec["sha256"] or reference["injection_seed"] != seed
                    or reference["dataset"] != dataset or reference["mode"] != "csm_05"
                    or reference["flowcad_commit"] != revision or reference["training_seed"] != 201 + seed - 42):
                raise ValueError(f"Flow-CAD reference differs from manifest: {reference_name}")
            target = records / f"{reference_name}.npz"
            if target.exists():
                with np.load(target, allow_pickle=False) as previous:
                    if (previous["sha256"].item() != input_spec["sha256"]
                            or previous["dataset"].item() != dataset or previous["injection_seed"].item() != seed
                            or previous["training_seed"].item() != reference["training_seed"]
                            or previous["scores"].shape != (spec["n_rows"],)
                            or not np.isfinite(previous["scores"]).all()):
                        raise ValueError(f"Invalid saved record: {target}")
                completed += 1
                continue
            jobs.append((dataset, spec, seed, args.manifest.parent / input_spec["path"],
                         input_spec["sha256"], reference["training_seed"], target))
    jobs.sort(key=lambda job: (-job[1]["n_rows"], job[0], job[2]))
    expected = completed + len(jobs)
    print(json.dumps({"expected": expected, "resuming_from": completed, "workers": args.workers}), flush=True)
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(evaluate, job) for job in jobs]):
                record = future.result()
                completed += 1
                print(json.dumps({"completed": completed, "expected": expected, **record}), flush=True)


if __name__ == "__main__":
    main()
