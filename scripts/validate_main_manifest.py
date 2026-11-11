#!/usr/bin/env python3
"""Validate the pinned 20-dataset, seven-seed main benchmark manifest."""

import argparse
from hashlib import sha256
from pathlib import Path

from replay import MAIN_REFERENCE, read_manifest


EXPECTED_SEEDS = tuple(range(43, 50))
EXPECTED_DATASETS = 20
EXPECTED_CASES = EXPECTED_DATASETS * len(EXPECTED_SEEDS)


def validate(path, check_files=False, data_root=None):
    manifest = read_manifest(path)
    data_root = data_root or path.parent
    datasets = manifest["datasets"]
    if len(datasets) != EXPECTED_DATASETS:
        raise ValueError(f"Expected {EXPECTED_DATASETS} datasets, found {len(datasets)}")
    cases = 0
    for dataset, spec in datasets.items():
        seeds = tuple(spec["injection_seeds"])
        if seeds != EXPECTED_SEEDS or spec["benchmark_runs"] != len(EXPECTED_SEEDS):
            raise ValueError(f"{dataset}: expected injection seeds 43 through 49")
        if tuple(sorted(map(int, spec["files"]))) != EXPECTED_SEEDS:
            raise ValueError(f"{dataset}: file entries do not match injection seeds")
        for seed in seeds:
            entry = spec["files"][str(seed)]
            if Path(entry["path"]).as_posix() != f"{dataset}/seed_{seed}.csv":
                raise ValueError(f"{dataset}/{seed}: unexpected relative path")
            if check_files:
                source = data_root / entry["path"]
                if not source.is_file():
                    raise FileNotFoundError(source)
                if sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
                    raise ValueError(f"{dataset}/{seed}: CSV checksum mismatch")
            cases += 1
    if cases != EXPECTED_CASES:
        raise ValueError(f"Expected {EXPECTED_CASES} cases, found {cases}")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MAIN_REFERENCE)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--data-root", type=Path,
                        help="Directory containing the dataset subdirectories (default: manifest directory)")
    args = parser.parse_args()
    data_root = args.data_root.resolve() if args.data_root else None
    cases = validate(args.manifest.resolve(), args.check_files, data_root)
    print(f"Validated {len(EXPECTED_SEEDS)} seeds on {EXPECTED_DATASETS} datasets ({cases} cases).")


if __name__ == "__main__":
    main()
