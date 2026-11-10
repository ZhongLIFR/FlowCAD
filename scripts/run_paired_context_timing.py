"""Compare Flow-CAD and a QCAD-compatible Gower port on frozen paired inputs."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter, process_time


from replay import MANIFEST, REPLAY, ROOT, read_manifest, verify_model


PORT = ROOT / "scripts" / "run_paired_qcad.py"
METRICS = ("pr_auc", "roc_auc", "p_at_n")


def flowcad_timing(path, spec, reference, seed):
    import numpy as np
    import pandas as pd
    from flowcad import checkpoints
    from flowcad.data import load_benchmark
    from flowcad.metrics import evaluate_scores

    original = checkpoints._SharedDraws
    captured = []

    class CaptureDraws(original):
        def contributions(self, model):
            captured.append(model)
            return super().contributions(model)

    wall, cpu = perf_counter(), process_time()
    data = load_benchmark({**spec, "file": path.name}, path.parent, require_labels=False)
    checkpoints._SharedDraws = CaptureDraws
    try:
        result = checkpoints.select_checkpoint(data.context, data.behavior, seed)
    finally:
        checkpoints._SharedDraws = original
    labels = pd.read_csv(path, usecols=["ground_truth"]).loc[data.frame.index, "ground_truth"]
    metrics = evaluate_scores(labels.to_numpy(), result["scores"])
    total_cpu, total_wall = process_time() - cpu, perf_counter() - wall
    if result["epoch"] != reference["selected_epoch"] or any(metrics[m] != reference[m] for m in METRICS):
        raise ValueError("Timing run did not reproduce the frozen Flow-CAD result")

    wall, cpu = perf_counter(), process_time()
    draws = original(data.context, data.behavior, seed)
    scores = draws.scores(captured[0])
    draws.contributions(captured[0])
    scoring_cpu, scoring_wall = process_time() - cpu, perf_counter() - wall
    if not np.array_equal(scores, result["scores"]):
        raise ValueError("Retimed scoring did not reproduce the selected scores")
    return {
        "total_cpu_s": total_cpu, "total_wall_s": total_wall,
        "fitted_score_cpu_s": scoring_cpu, "fitted_score_wall_s": scoring_wall,
        "selected_epoch": result["epoch"], **{m: metrics[m] for m in METRICS},
    }


def write_csv(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--out", type=Path, default=REPLAY / "paired_timing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--qcad-results", type=Path, default=REPLAY / "paired_qcad")
    args = parser.parse_args()

    verify_model()
    sys.path.insert(0, str(ROOT))
    import flowcad  # Bind the requested release before importing the port.
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    port_spec = importlib.util.spec_from_file_location("matched_qcad", PORT)
    port = importlib.util.module_from_spec(port_spec)
    port_spec.loader.exec_module(port)
    manifest = read_manifest(args.manifest)["datasets"]
    chosen = sorted(manifest, key=lambda name: (manifest[name]["n_rows"], name))[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "protocol.json").write_text(json.dumps({
        "datasets": chosen, "injection_seed": args.seed,
        "architecture": platform.machine(), "system": platform.platform(), "threads": 1,
        "pytorch": torch.__version__, "sklearn": port.sklearn.__version__,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "port_sha256": hashlib.sha256(PORT.read_bytes()).hexdigest(),
        "design": "Serial per table, alternate method order. Includes feature loading, fitting, selection, scoring, attribution and evaluation. The fitted Flow-CAD timer excludes labels and includes fresh draws, scores and contributions.",
    }, indent=2) + "\n")

    rows = []
    with threadpool_limits(limits=1):
        for i, dataset in enumerate(chosen):
            spec = manifest[dataset]
            file = spec["files"][str(args.seed)]
            path = args.manifest.parent / file["path"]
            reference = json.loads((args.manifest.parent.parent / "flowcad_csm05" / f"{dataset}_seed_{args.seed}.json").read_text())
            qcad_reference = json.loads((args.qcad_results / f"paper_{dataset}_seed_{args.seed}.json").read_text())
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if len({actual_hash, file["sha256"], reference["sha256"], qcad_reference["sha256"]}) != 1:
                raise ValueError(f"Input hash mismatch for {dataset}")
            methods = ("Flow-CAD", "QCAD-Gower port") if i % 2 == 0 else ("QCAD-Gower port", "Flow-CAD")
            for method in methods:
                if method == "Flow-CAD":
                    timed = flowcad_timing(path, spec, reference, reference["training_seed"])
                else:
                    port_result = args.out / f"paper_{dataset}_seed_{args.seed}.json"
                    port_result.unlink(missing_ok=True)
                    wall, cpu = perf_counter(), process_time()
                    qcad = port.run_one((dataset, args.seed, path, {**spec, **file}, spec, "paper", args.out))
                    total_cpu, total_wall = process_time() - cpu, perf_counter() - wall
                    if any(qcad[m] != qcad_reference[m] for m in METRICS):
                        raise ValueError("Timing run did not reproduce the frozen QCAD port result")
                    timed = {
                        "total_cpu_s": total_cpu, "total_wall_s": total_wall,
                        "fitted_score_cpu_s": "", "fitted_score_wall_s": "", "selected_epoch": "",
                        **{m: qcad[m] for m in METRICS},
                    }
                row = {"dataset": dataset, "injection_seed": args.seed, "sha256": actual_hash,
                       "n_rows": spec["n_rows"], "method": method, **timed}
                rows.append(row)
                write_csv(args.out / "per_run.csv", rows)
                print(json.dumps(row), flush=True)
    pairs = {(r["dataset"], r["method"]): r for r in rows}
    for column in ("total_cpu_s", "total_wall_s"):
        ratios = [pairs[d, "QCAD-Gower port"][column] / pairs[d, "Flow-CAD"][column] for d in chosen]
        print(column, "QCAD/Flow min, median, max:", min(ratios), statistics.median(ratios), max(ratios))


if __name__ == "__main__":
    main()
