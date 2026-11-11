"""Fit Flow-CAD for 1,000 epochs and vary scoring on the main benchmark inputs."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time


from replay import MAIN_MANIFEST as MANIFEST, ROOT, read_manifest, verify_model
METRICS = ("pr_auc", "roc_auc", "p_at_n")
TAUS = (2, 5, 10, 20, 30, 40, 50, 80, 100)
BUDGETS = (2, 4, 8, 16, 32, 64, 128)
SETTINGS = {
    "normalization": ("default", "no_mad", "no_trunc"),
    "tau": TAUS,
    "norm": ("L1", "L2", "max"),
    "kt": BUDGETS,
    "kz": BUDGETS,
}
DEFAULT_SETTINGS = {"normalization": "default", "tau": "10", "norm": "L1", "kt": "16", "kz": "16"}


def absolute_residuals(model, context, behavior, noise, times, chunk_size, torch):
    n_rows, n_z, n_t = times.shape
    n_draws = n_z * n_t
    flat_times = times.reshape(-1, 1)
    output = torch.empty((n_rows * n_draws, behavior.shape[1]), dtype=behavior.dtype)
    for start in range(0, len(output), chunk_size):
        stop = min(start + chunk_size, len(output))
        flat = torch.arange(start, stop)
        rows = torch.div(flat, n_draws, rounding_mode="floor")
        noise_index = torch.div(flat, n_t, rounding_mode="floor") % n_z
        t = flat_times[start:stop]
        z, b = noise[rows, noise_index], behavior[rows]
        xt, target = (1.0 - t) * z + t * b, b - z
        output[start:stop] = torch.abs(model(xt, context[rows], t) - target)
    return output


def score_sweeps(model, draws, seed, checkpoints, np, torch):
    rows_per_block = checkpoints.CHUNK_SIZE // checkpoints.N_DRAWS
    base_z, base_t = checkpoints.N_Z, checkpoints.N_T
    extra_z, extra_t = BUDGETS[-1] - base_z, BUDGETS[-1] - base_t
    z_generator = torch.Generator(device="cpu").manual_seed(seed + 3_000_003)
    t_generator = torch.Generator(device="cpu").manual_seed(seed + 4_000_003)
    output = {
        (study, str(setting)): np.empty(draws.n_rows, dtype=np.float64)
        for study, settings in SETTINGS.items() for setting in settings
    }

    def score_residuals(context, behavior, noise, times):
        return absolute_residuals(
            model, context, behavior, noise, times, checkpoints.CHUNK_SIZE, torch,
        )

    def reduce_rows(values, n):
        # The default is accumulated in the exact order used by _SharedDraws.scores.
        totals = np.zeros(n, dtype=np.float64)
        np.add.at(totals, np.repeat(np.arange(n), checkpoints.N_DRAWS), values.numpy())
        return totals / checkpoints.N_DRAWS

    model.eval()
    with torch.no_grad():
        for start in range(0, draws.n_rows, rows_per_block):
            block = slice(start, start + rows_per_block)
            context, behavior = draws.context[block], draws.behavior[block]
            noise, times = draws.noise[block], draws.time[block]
            n = len(behavior)
            raw = score_residuals(context, behavior, noise, times)
            normalized = raw / draws.scale
            clipped = torch.clamp(normalized, max=10.0)
            l1 = clipped.sum(dim=1)
            base = reduce_rows(l1, n)
            output["normalization", "default"][block] = base
            output["normalization", "no_mad"][block] = reduce_rows(raw.clamp(max=10.0).sum(dim=1), n)
            output["normalization", "no_trunc"][block] = reduce_rows(normalized.sum(dim=1), n)
            output["norm", "L1"][block] = base
            if draws.n_features == 1:
                output["norm", "L2"][block] = base
                output["norm", "max"][block] = base
            else:
                output["norm", "L2"][block] = reduce_rows(torch.linalg.vector_norm(clipped, dim=1), n)
                output["norm", "max"][block] = reduce_rows(clipped.amax(dim=1), n)
            for tau in TAUS:
                output["tau", str(tau)][block] = base if tau == 10 else reduce_rows(
                    normalized.clamp(max=tau).sum(dim=1), n,
                )

            # The exact main 16×16 draws are prefixes on both axes. Each
            # axis adds an independent extension while holding the other fixed.
            new_times = torch.rand((n, base_z, extra_t), generator=t_generator)
            t_extra = score_residuals(context, behavior, noise, new_times)
            t_scores = (t_extra / draws.scale).clamp(max=10.0).sum(dim=1).numpy().reshape(n, base_z, extra_t)
            new_noise = torch.randn((n, extra_z, draws.n_features), generator=z_generator)
            new_z_times = torch.rand((n, extra_z, base_t), generator=z_generator)
            z_extra = score_residuals(context, behavior, new_noise, new_z_times)
            z_scores = (z_extra / draws.scale).clamp(max=10.0).sum(dim=1).numpy().reshape(n, extra_z, base_t)
            base_draw_scores = l1.numpy().reshape(n, base_z, base_t)
            base_total = base * checkpoints.N_DRAWS
            for k in BUDGETS:
                if k < base_t:
                    t_value = base_draw_scores[:, :, :k].sum(axis=(1, 2), dtype=np.float64) / (base_z * k)
                    z_value = base_draw_scores[:, :k, :].sum(axis=(1, 2), dtype=np.float64) / (base_t * k)
                elif k == base_t:
                    t_value = z_value = base
                else:
                    t_total = t_scores[:, :, :k - base_t].sum(axis=(1, 2), dtype=np.float64)
                    z_total = z_scores[:, :k - base_z, :].sum(axis=(1, 2), dtype=np.float64)
                    t_value = (base_total + t_total) / (base_z * k)
                    z_value = (base_total + z_total) / (base_t * k)
                output["kt", str(k)][block] = t_value
                output["kz", str(k)][block] = z_value
    return output


def evaluate(job):
    release, spec, data_path, record = job
    sys.path.insert(0, str(release))
    import numpy as np
    import pandas as pd
    import torch
    from flowcad import checkpoints
    from flowcad.data import load_benchmark
    from flowcad.fixed import ScoringDraws, fit_fixed
    from flowcad.metrics import evaluate_scores

    torch.set_num_threads(1)
    data = load_benchmark({**spec, "file": data_path.name}, data_path.parent, require_labels=False)
    if data.sha256 != record["sha256"] or len(data.frame) != record["n_rows"] or data.dropped_rows:
        raise ValueError("Input differs from the matched fixed-epoch replay")
    started = time.perf_counter()
    model = fit_fixed(data.context, data.behavior, record["training_seed"])
    fit_seconds = time.perf_counter() - started
    draws = ScoringDraws(data.context, data.behavior, record["training_seed"])
    scores = score_sweeps(model, draws, record["training_seed"], checkpoints, np, torch)
    if not np.array_equal(scores["normalization", "default"], draws.scores(model)):
        raise ValueError("Ablation default raw scores differ from the fixed-epoch scorer")
    labels = pd.read_csv(data_path, usecols=["ground_truth"]).loc[data.frame.index, "ground_truth"].to_numpy()
    results = []
    for (study, setting), values in scores.items():
        metrics = evaluate_scores(labels, values)
        results.append({"study": study, "setting": setting, **{name: metrics[name] for name in METRICS}})
    verify_defaults(results, record)
    if draws.n_features == 1 and any(
        not np.array_equal(scores["norm", norm], scores["norm", "L1"]) for norm in ("L2", "max")
    ):
        raise ValueError("Scalar residual norms must coincide")
    fields = ("dataset", "injection_seed", "training_seed", "selected_epoch", "sha256", "flowcad_commit")
    return {
        **{name: record[name] for name in fields},
        "fit_seconds": fit_seconds, "sweep_seconds": time.perf_counter() - started - fit_seconds,
        "results": results,
    }


def verify_defaults(results, primary):
    defaults = {row["study"]: row for row in results if DEFAULT_SETTINGS[row["study"]] == row["setting"]}
    if defaults.keys() != DEFAULT_SETTINGS.keys() or any(
        row[name] != primary[name] for row in defaults.values() for name in METRICS
    ):
        raise ValueError(f"Default metrics differ from the fixed-epoch replay for {primary['dataset']}")


def write_csv(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_results(output, records):
    rows, grouped = [], defaultdict(list)
    for record in sorted(records, key=lambda row: (row["dataset"], row["injection_seed"])):
        fields = ("dataset", "injection_seed", "training_seed", "selected_epoch", "sha256")
        for result in record["results"]:
            rows.append({**{field: record[field] for field in fields}, **result})
            grouped[record["dataset"], result["study"], result["setting"]].append(result)
    write_csv(output / "per_run.csv", rows)
    summary = []
    for (dataset, study, setting), group in sorted(grouped.items()):
        row = {"dataset": dataset, "study": study, "setting": setting, "n_runs": len(group)}
        for metric in METRICS:
            values = [item[metric] for item in group]
            row[metric + "_mean"] = statistics.mean(values)
            row[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(row)
    write_csv(output / "summary.csv", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--injection-seeds", default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        parser.error("workers must be between 1 and 6")
    release = ROOT
    revision = verify_model()
    data = args.manifest.parent
    datasets = read_manifest(args.manifest)["datasets"]
    replay = data.parent
    output = replay / "ablation_sensitivity_fixed_1000"
    output.mkdir(exist_ok=True)
    records, jobs = [], []
    for dataset in datasets if args.datasets == "all" else args.datasets.split(","):
        spec = datasets[dataset]
        seeds = spec["injection_seeds"] if args.injection_seeds == "all" else map(int, args.injection_seeds.split(","))
        for seed in seeds:
            filename = f"{dataset}_seed_{seed}.json"
            record = json.loads((replay / "flowcad_fixed_1000" / filename).read_text())
            info = spec["files"][str(seed)]
            if (record["sha256"] != info["sha256"] or record["flowcad_commit"] != revision
                    or record["selected_epoch"] != 1000 or record["training_seed"] != 201 + seed - 42):
                raise ValueError(f"Primary run does not match code or input: {filename}")
            path = output / filename
            if path.is_file():
                existing = json.loads(path.read_text())
                if any(existing[key] != record[key] for key in (
                    "sha256", "injection_seed", "training_seed", "selected_epoch", "flowcad_commit",
                )):
                    raise ValueError(f"Ablation does not match primary run: {filename}")
                verify_defaults(existing["results"], record)
                records.append(existing)
            else:
                estimated_steps = math.ceil(spec["n_rows"] / 128) * record["selected_epoch"]
                jobs.append((estimated_steps, path, (release, spec, data / info["path"], record)))
    if records:
        write_results(output, records)
    # Start the longest jobs first so they do not hold up the end of the run.
    jobs.sort(key=lambda item: (-item[0], item[1].name))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(evaluate, job): path for _, path, job in jobs}
        for completed, future in enumerate(as_completed(pending), 1):
            row = future.result()
            path = pending[future]
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
            temporary.replace(path)
            records.append(row)
            if completed % 5 == 0 or completed == len(pending):
                write_results(output, records)
            print(json.dumps({key: value for key, value in row.items() if key != "results"}), flush=True)


if __name__ == "__main__":
    main()
