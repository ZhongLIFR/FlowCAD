"""Compare Flow-CAD to conditional means on marginal-preserving synthetic data."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import statistics
import sys


from replay import REPLAY, ROOT, verify_model
FAMILIES = ("multimodal_equal_moments", "unimodal_mean_shift")
SEEDS = (42, 43, 44, 45, 46)
ROWS = 800
METRICS = ("pr_auc", "roc_auc", "p_at_n")


def synthetic_table(family, seed, np):
    rng = np.random.default_rng(seed)
    if family == "multimodal_equal_moments":
        # A block contains a noisy point rotated to each of eight modes.
        # Opposite pairs and quarter rotations fix both conditional means
        # and covariance matrices, including after flipping whole blocks.
        angles = np.concatenate((np.arange(4) * np.pi / 2, np.arange(4) * np.pi / 2 + np.pi / 4))
        cosine, sine = np.cos(angles), np.sin(angles)
        rotations = np.array([[cosine, -sine], [sine, cosine]]).transpose(2, 0, 1)
        sigma = 0.08
        latent = rng.normal(0.0, sigma, size=(ROWS // 8, 2))
        latent[:, 0] += 1.0
        behavior = np.einsum("aij,bj->bai", rotations, latent)
        pattern = np.repeat([0, 1], 4)
    else:
        sigma = 0.25
        latent = rng.normal(0.0, sigma, size=(ROWS // 2, 2))
        latent[:, 0] -= 1.0
        behavior = np.stack([latent, -latent], axis=1)
        pattern = np.array([0, 1])

    # Swapping equally represented contexts within randomly selected blocks
    # leaves the complete behavior table and exact context counts unchanged.
    flips = np.zeros(len(behavior), dtype=bool)
    flips[rng.choice(len(behavior), len(behavior) // 20, replace=False)] = True
    labels = np.repeat(flips, len(pattern)).astype(np.int8)
    clean = np.tile(pattern, len(behavior))
    context = np.bitwise_xor(clean, labels)
    behavior = behavior.reshape(-1, 2)
    np.testing.assert_array_equal(np.bincount(context), np.bincount(clean))
    np.testing.assert_array_equal(np.bincount(context[labels == 1]), [20, 20])
    if family == "multimodal_equal_moments":
        np.testing.assert_allclose(behavior[context == 0].mean(0), [0, 0], atol=1e-12)
        np.testing.assert_allclose(behavior[context == 1].mean(0), [0, 0], atol=1e-12)
        np.testing.assert_allclose(np.cov(behavior[context == 0].T), np.cov(behavior[context == 1].T), atol=1e-12)
    order = rng.permutation(ROWS)
    return context[order], behavior[order].astype(np.float32), labels[order]


def evaluate(job):
    release, family, seed = job
    sys.path.insert(0, str(release))
    import numpy as np
    import torch
    from flowcad import checkpoints
    from flowcad.metrics import evaluate_scores

    torch.set_num_threads(1)
    context, behavior, truth = synthetic_table(family, seed, np)
    training_seed = 201 + seed - SEEDS[0]
    learned = {}
    for method, features in (("conditional_flow", context), ("no_context_flow", np.zeros_like(context))):
        learned[method] = checkpoints.select_checkpoint(features[:, None].astype(np.float32), behavior, training_seed)

    # Fitted means and MADs use all feature rows, without the injected labels.
    fit_mean = np.stack([behavior[context == group].mean(axis=0) for group in (0, 1)])
    scale = np.maximum(np.median(np.abs(behavior - np.median(behavior, axis=0)), axis=0), 1e-6)

    scored = {
        "fitted_conditional_mean": (
            None, np.minimum(np.abs(behavior - fit_mean[context]) / scale, 10.0).sum(axis=1)
        ),
        **{method + "_csm": (selected["epoch"], selected["scores"])
           for method, selected in learned.items()},
    }

    records = []
    for method, (epoch, scores) in scored.items():
        metrics = evaluate_scores(truth, scores)
        records.append({"family": family, "data_seed": seed, "training_seed": training_seed,
                        "method": method, "epoch": epoch, **{name: metrics[name] for name in METRICS}})
    return records


def write_csv(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(records):
    summary = []
    for family, method in sorted({(row["family"], row["method"]) for row in records}):
        group = [row for row in records if row["family"] == family and row["method"] == method]
        row = {"family": family, "method": method, "n_seeds": len(group)}
        for metric in METRICS:
            values = [r[metric] for r in group]
            row[metric + "_mean"] = statistics.mean(values)
            row[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(row)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPLAY / "conditional_distribution")
    parser.add_argument("--families", choices=FAMILIES, nargs="+", default=FAMILIES)
    parser.add_argument("--seeds", type=int, choices=SEEDS, nargs="+", default=SEEDS)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between 1 and 4")
    release = ROOT
    commit = verify_model()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "flowcad_commit": commit, "rows_per_trial": ROWS, "anomaly_fraction": 0.05,
        "families": args.families, "data_seeds": args.seeds,
        "model_seeds": [201 + seed - SEEDS[0] for seed in args.seeds],
        "injection": "Swap binary contexts within randomly selected balanced blocks; preserves complete behavior and context marginals.",
        "multimodal": "Four axial vs four diagonal modes at radius 1; Gaussian noise SD .08; quarter-rotation blocks match first two conditional moments exactly.",
        "unimodal": "Reflected Gaussian pairs centered on (-1, 0) and (1, 0); per-dimension SD .25.",
        "no_context": "Keep architecture and initialization identical; replace context input with zeros.",
        "fit_and_score": "Same full table; public CSM with 5% tail and its complete checkpoint grid; one CPU thread per process.",
        "labels": "Used for metrics only, after checkpoint selection.",
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    records, jobs = [], []
    for family in args.families:
        for seed in args.seeds:
            path = output / f"{family}_{seed}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if record["flowcad_commit"] != commit:
                    raise ValueError(f"Existing run uses different public code: {path}")
                records.extend(record["results"])
            else:
                jobs.append((path, (release, family, seed)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(evaluate, job): path for path, job in jobs}
        for future in as_completed(pending):
            results = future.result()
            path = pending[future]
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"flowcad_commit": commit, "results": results},
                                            indent=2, allow_nan=False) + "\n")
            temporary.replace(path)
            records.extend(results)
            print(path.name, flush=True)
    order = lambda row: (row["family"], row["data_seed"], row["method"], row["epoch"] or 0)
    write_csv(output / "per_run.csv", sorted(records, key=order))
    write_csv(output / "summary.csv", summarize(records))


if __name__ == "__main__":
    main()
