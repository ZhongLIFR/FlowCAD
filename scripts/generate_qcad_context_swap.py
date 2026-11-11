#!/usr/bin/env python3
"""Generate within-stratum behavioral swaps from the historical QCAD raw data."""

import argparse
from copy import deepcopy
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from generate_qcad_paper_replay import digest, git
from replay import REFERENCE, ROOT


MIN_CHILD_ROWS = 8
TARGET_STRATUM_ROWS = 32
PROTOCOL = "within_context_stratum_behavior_swap"


def context_strata(frame, context, categorical):
    numeric = [name for name in context if name not in categorical]
    coordinates = (
        MinMaxScaler().fit_transform(frame[numeric])
        if numeric
        else np.empty((len(frame), 0))
    )
    if categorical:
        buckets = frame.groupby(categorical, sort=False, observed=True).indices.values()
    else:
        buckets = [np.arange(len(frame))]

    def split(indices):
        if len(indices) > TARGET_STRATUM_ROWS:
            values = coordinates[indices]
            widths = np.ptp(values, axis=0)
            for axis in np.argsort(-widths, kind="stable"):
                order = np.argsort(values[:, axis], kind="stable")
                sorted_values = values[order, axis]
                cuts = np.flatnonzero(sorted_values[1:] > sorted_values[:-1]) + 1
                cuts = cuts[(cuts >= MIN_CHILD_ROWS) & (cuts <= len(indices) - MIN_CHILD_ROWS)]
                if len(cuts):
                    cut = cuts[np.argmin(np.abs(2 * cuts - len(indices)))]
                    yield from split(indices[order[:cut]])
                    yield from split(indices[order[cut:]])
                    return
        yield indices

    return [stratum for indices in buckets for stratum in split(indices)]


def swap_behavior(frame, info, seed):
    behavior = info["behavior_columns"]
    strata = context_strata(frame, info["context_columns"], info["categorical_columns"])
    original = frame[behavior].to_numpy(copy=True)
    swapped = original.copy()
    rng = np.random.default_rng(seed)
    pairs = []
    for stratum in strata:
        order = rng.permutation(stratum)
        candidates = order[:len(order) - len(order) % 2].reshape(-1, 2)
        unequal = (original[candidates[:, 0]] != original[candidates[:, 1]]).sum(axis=1) >= 2
        pairs.extend(candidates[unequal])

    target = info["n_anomalies"]
    n_pairs = target // 2
    if len(pairs) < n_pairs:
        raise ValueError(f"Seed {seed}: need {n_pairs} nontrivial disjoint pairs, have {len(pairs)}")
    selected = np.asarray(pairs)[rng.choice(len(pairs), size=n_pairs, replace=False)]
    for left, right in selected:
        candidates = np.flatnonzero(original[left] != original[right])
        column = rng.choice(candidates)
        swapped[[left, right], column] = original[[right, left], column]

    changed = (original != swapped).sum(axis=1)
    labels = (changed == 1).astype(np.int8)
    if changed.max() > 1 or int(labels.sum()) != target or len(np.unique(selected)) != target:
        raise ValueError(f"Seed {seed}: invalid swapped-row labels")
    for stratum in strata:
        np.testing.assert_array_equal(
            np.sort(swapped[stratum], axis=0), np.sort(original[stratum], axis=0)
        )

    result = frame.copy()
    result[behavior] = swapped
    unchanged_columns = frame.columns.difference(behavior)
    pd.testing.assert_frame_equal(result[unchanged_columns], frame[unchanged_columns])
    return result.assign(ground_truth=labels), {
        "stratum_count": len(strata),
        "stratum_max_rows": max(map(len, strata)),
        "n_candidate_pairs": len(pairs),
    }


def prepare(raw, info, seed):
    if len(raw) > 20_000:
        raw = raw.sample(n=20_000, random_state=np.random.RandomState(seed))
    frame = raw[info["benchmark_columns"]].reset_index(drop=True).copy()
    numeric = info["scaled_columns"]
    if numeric:
        frame[numeric] = MinMaxScaler().fit_transform(frame[numeric])
    return frame


def write_atomic(path, content):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(content)
    temp.replace(path)


def generate(qcad_repo, output):
    reference = json.loads(REFERENCE.read_bytes())
    manifest = {
        "protocol": PROTOCOL,
        "source_repository": reference["source_repository"],
        "source_commit": reference["source_commit"],
        "source_manifest_sha256": digest(REFERENCE.read_bytes()),
        "context_strata": {
            "categorical": "exact match on the replay's categorical context columns",
            "numeric": "recursive median splits on the widest min-max-scaled context dimension with a valid split",
            "minimum_child_rows": MIN_CHILD_ROWS,
            "target_maximum_rows": TARGET_STRATUM_ROWS,
        },
        "pair_sampling": "permute rows within strata, form disjoint consecutive pairs, retain pairs differing in at least two behavioral coordinates, sample pairs uniformly, and exchange one uniformly sampled unequal behavioral coordinate per pair",
        "target_rounding": "nearest even count, rounding ties down",
        "expected_datasets": [],
        "excluded_datasets": {},
        "datasets": {},
    }
    for key in reference["expected_datasets"]:
        original = reference["datasets"][key]
        if len(original["behavior_columns"]) < 2:
            manifest["excluded_datasets"][key] = "fewer than two behavioral columns"
            continue
        info = deepcopy(original)
        info.pop("abalone_reference_check", None)
        info["original_n_anomalies"] = info["n_anomalies"]
        info["n_anomalies"] -= info["n_anomalies"] % 2
        info["files"] = {}
        if (qcad_repo / ".git").exists():
            raw_bytes = git(qcad_repo, "show", f"{reference['source_commit']}:{info['raw_path']}")
        else:
            raw_bytes = (qcad_repo / info["raw_path"]).read_bytes()
        if digest(raw_bytes) != info["raw_sha256"]:
            raise ValueError(f"{key}: raw data checksum differs from the historical replay")
        raw = pd.read_csv(io.BytesIO(raw_bytes)).dropna()
        dataset_dir = output / key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for seed in info["injection_seeds"]:
            frame = prepare(raw, info, seed)
            if len(frame) != info["n_rows"]:
                raise ValueError(f"{key} seed {seed}: row count differs from the historical replay")
            result, diagnostics = swap_behavior(frame, info, seed)
            path = dataset_dir / f"seed_{seed}.csv"
            content = result.to_csv(index=False, lineterminator="\n").encode()
            write_atomic(path, content)
            info["files"][str(seed)] = {
                "path": str(path.relative_to(output)),
                "sha256": digest(content),
                **diagnostics,
            }
        manifest["expected_datasets"].append(key)
        manifest["datasets"][key] = info
        print(f"{key}: verified {len(info['files'])} trials with {info['n_anomalies']} swapped rows each", flush=True)

    write_atomic(output / "manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qcad-repo", type=Path, required=True, help="QCAD clone or tree with historical raw CSVs")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "qcad_context_swap" / "data")
    args = parser.parse_args()
    generate(args.qcad_repo, args.output)
