"""Run the held-out sign-parity conditional-dependence experiment."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations, product
from pathlib import Path
import sys

import numpy as np

from replay import MAIN_REPLAY, ROOT, verify_model

sys.path.insert(0, str(ROOT))


ROWS = 800
SEEDS = tuple(range(42, 52))
METHODS = ("joint_flow", "conditional_mean_mlp", "coordinatewise_flow", "zero_context_flow")
TIE_TOLERANCE = 1e-6


def synthetic_table(seed):
    """Create balanced sign-parity contexts with five percent context swaps."""
    rng = np.random.default_rng(seed)
    signs = np.asarray(list(product((-1, 1), repeat=3)))
    magnitudes = rng.normal(1.0, 0.08, size=(ROWS // len(signs), signs.shape[1]))
    behavior = (magnitudes[:, None, :] * signs[None, :, :]).reshape(ROWS, -1).astype(np.float32)
    clean_context = np.tile((signs.prod(axis=1) < 0).astype(np.int8), len(magnitudes))
    swapped_blocks = np.zeros(len(magnitudes), dtype=bool)
    swapped_blocks[rng.choice(len(magnitudes), len(magnitudes) // 20, replace=False)] = True
    swaps = np.repeat(swapped_blocks, len(signs))
    observed_context = np.bitwise_xor(clean_context, swaps)

    for assignments in (clean_context, observed_context):
        left, right = behavior[assignments == 0], behavior[assignments == 1]
        np.testing.assert_array_equal(np.sort(left, axis=0), np.sort(right, axis=0))
        for columns in combinations(range(behavior.shape[1]), 2):
            expected = np.unique(left[:, columns], axis=0, return_counts=True)
            actual = np.unique(right[:, columns], axis=0, return_counts=True)
            for actual_part, expected_part in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(actual_part, expected_part)
    order = rng.permutation(ROWS)
    return observed_context[order], behavior[order], swaps[order]


def test_pairs(seed):
    observed, behavior, swaps = synthetic_table(seed)
    correct = np.bitwise_xor(observed, swaps)
    wrong = 1 - correct
    _, blocks, inverse, sizes = np.unique(
        np.abs(behavior), axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    np.testing.assert_array_equal(sizes, 8)
    return (correct[:, None].astype(np.float32), wrong[:, None].astype(np.float32),
            behavior, blocks, inverse)


def flow_pair_scores(train_context, train_behavior, correct, wrong, behavior,
                     blocks, inverse, seed):
    import torch
    from flowcad.fixed import ScoringDraws, fit_fixed

    model = fit_fixed(train_context, train_behavior, seed)
    draws = ScoringDraws(correct, behavior, seed)
    block_draws = ScoringDraws(correct[blocks], behavior[blocks], seed)
    draws.noise = block_draws.noise[inverse]
    draws.time = block_draws.time[inverse]
    median = np.median(train_behavior, axis=0)
    scale = np.maximum(np.median(np.abs(train_behavior - median), axis=0), 1e-6)
    draws.scale = torch.as_tensor(scale, dtype=torch.float32).view(1, -1)
    correct_scores = draws.scores(model)
    draws.context = torch.as_tensor(wrong, dtype=torch.float32)
    return correct_scores, draws.scores(model)


def conditional_mean_scores(train_context, train_behavior, correct, wrong, behavior, seed):
    import torch
    from torch.utils.data import DataLoader
    from flowcad.checkpoints import _BatchedDataset
    from flowcad.fixed import DEFAULT_EPOCHS
    from flowcad.reference import ConditionalVectorField, set_seed

    set_seed(seed)
    loader = DataLoader(
        _BatchedDataset(train_context, train_behavior), batch_size=128, shuffle=True,
        drop_last=False, collate_fn=lambda batch: batch,
    )
    model = ConditionalVectorField(train_context.shape[1], train_behavior.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(DEFAULT_EPOCHS):
        model.train()
        for batch_context, batch_behavior in loader:
            torch.randn_like(batch_behavior)
            torch.rand((len(batch_behavior), 1), device="cpu")
            state = torch.zeros_like(batch_behavior)
            time = torch.zeros((len(batch_behavior), 1), dtype=batch_behavior.dtype)
            loss = torch.mean((model(state, batch_context, time) - batch_behavior) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    contexts = torch.as_tensor(np.vstack((correct, wrong)), dtype=torch.float32)
    targets = torch.as_tensor(np.tile(behavior, (2, 1)), dtype=torch.float32)
    state = torch.zeros_like(targets)
    time = torch.zeros((len(targets), 1), dtype=targets.dtype)
    scale = np.maximum(
        np.median(np.abs(train_behavior - np.median(train_behavior, axis=0)), axis=0), 1e-6
    )
    model.eval()
    with torch.no_grad():
        scores = ((model(state, contexts, time) - targets).abs()
                  / torch.as_tensor(scale, dtype=torch.float32)).clamp(max=10).sum(dim=1).numpy()
    return np.split(scores, 2)


def evaluate(data_seed):
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    training_seed = 201 + data_seed - SEEDS[0]
    train_context, train_behavior, _ = synthetic_table(data_seed)
    train_context = train_context[:, None].astype(np.float32)
    correct, wrong, behavior, blocks, inverse = test_pairs(data_seed + 10_000)

    with threadpool_limits(limits=1):
        joint = flow_pair_scores(
            train_context, train_behavior, correct, wrong, behavior, blocks, inverse, training_seed
        )
        mean = conditional_mean_scores(
            train_context, train_behavior, correct, wrong, behavior, training_seed
        )
        zero = np.zeros_like(correct)
        zero_context = flow_pair_scores(
            np.zeros_like(train_context), train_behavior, zero, zero, behavior,
            blocks, inverse, training_seed,
        )
        coordinatewise = np.zeros_like(joint)
        for coordinate in range(behavior.shape[1]):
            column = slice(coordinate, coordinate + 1)
            coordinatewise += flow_pair_scores(
                train_context, train_behavior[:, column], correct, wrong, behavior[:, column],
                blocks, inverse, training_seed + 1_000 * coordinate,
            )

    rows = []
    for method, (normal, anomaly) in zip(
            METHODS, (joint, mean, coordinatewise, zero_context), strict=True):
        difference = anomaly - normal
        credits = np.where(np.abs(difference) <= TIE_TOLERANCE, 0.5, difference > 0)
        rows.append({
            "family": "sign_parity", "data_seed": data_seed,
            "training_seed": training_seed, "method": method,
            "epochs": 1000, "training_rows": ROWS, "test_pairs": len(behavior),
            "pair_accuracy": float(credits.mean()),
            "tie_fraction": float((credits == 0.5).mean()),
        })
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(records, seeds):
    by_case = {(row["method"], row["data_seed"]): row["pair_accuracy"] for row in records}
    resamples = np.random.default_rng(5720).integers(0, len(seeds), size=(10_000, len(seeds)))
    joint = np.array([by_case["joint_flow", seed] for seed in seeds])
    rows = []
    for method in METHODS:
        values = np.array([by_case[method, seed] for seed in seeds])
        differences = joint - values
        low, high = np.quantile(differences[resamples].mean(axis=1), [0.025, 0.975])
        rows.append({
            "family": "sign_parity", "method": method, "seeds": len(seeds),
            "mean_accuracy": values.mean(), "min_accuracy": values.min(),
            "max_accuracy": values.max(), "seeds_above_75pct": (values > 0.75).sum(),
            "joint_minus_method": differences.mean(), "paired_ci_low": low,
            "paired_ci_high": high,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MAIN_REPLAY / "heldout_dependence")
    parser.add_argument("--seeds", choices=SEEDS, type=int, nargs="+", default=SEEDS)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct")
    verify_model()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(evaluate, seed): seed for seed in args.seeds}
        for future in as_completed(pending):
            records.extend(future.result())
            print(f"completed seed {pending[future]}", flush=True)
    records.sort(key=lambda row: (row["data_seed"], METHODS.index(row["method"])))
    write_csv(args.output / "per_run.csv", records)
    write_csv(args.output / "summary.csv", summarize(records, args.seeds))


if __name__ == "__main__":
    main()
