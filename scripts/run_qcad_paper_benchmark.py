"""Run and aggregate the paired QCAD shift benchmark."""

import argparse
import json
import os
from pathlib import Path
import sys

from benchmark_utils import CLASSICAL_METHODS, run_groups
from replay import MAIN_MANIFEST, ROOT, read_manifest
from summarize_qcad_paper_full import read_records, summarize, validate


def case_hashes(datasets):
    return {
        (dataset, int(seed)): spec["files"][str(seed)]["sha256"]
        for dataset, spec in datasets.items()
        for seed in spec["injection_seeds"]
    }


def run_detectors(args, manifest, work, datasets, pending):
    csv_scope = ("--datasets", "all", "--injection-seeds", "all")
    qcad_scope = ("--all-injections",)
    classical_scope = ()

    def command(script, *options):
        return [sys.executable, "-Bu", str(ROOT / "scripts" / script),
                "--manifest", str(manifest), *(str(option) for option in options)]

    commands = {
        "qcad": command("run_paired_qcad.py", *qcad_scope, "--jobs", args.qcad_jobs,
                        "--output", work / "all_qcad"),
        "dte-tccm": command("run_qcad_paper_dte_tccm.py", *csv_scope,
                           "--dte-repo", args.dte_repo.resolve(),
                           "--tccm-repo", args.tccm_repo.resolve(), "--workers", args.dte_workers,
                           "--output", work / "baselines_dte_tccm"),
        "flowcad": command("run_qcad_paper_flowcad.py", *csv_scope,
                          "--fixed-epochs", 1000, "--workers", args.flowcad_workers),
        "lunar": command("run_qcad_replay_lunar.py", *csv_scope,
                        "--workers", args.lunar_workers, "--output", work / "lunar"),
        "classical": command("run_qcad_baselines.py", *classical_scope,
                            "--qcad-repo", args.qcad_repo.resolve(), "--jobs", args.classical_jobs,
                            "--output", work / "classical_baselines",
                            "--methods", *CLASSICAL_METHODS),
    }
    env = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        env[variable] = "1"
    run_groups(commands, work, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MAIN_MANIFEST)
    parser.add_argument("--qcad-repo", type=Path, help="QCAD checkout supplying LoPAD's Markov blankets")
    parser.add_argument("--dte-repo", type=Path, help="Pinned DTE checkout")
    parser.add_argument("--tccm-repo", type=Path, help="Pinned TCCM checkout")
    parser.add_argument("--summary-only", action="store_true", help="Aggregate results already on disk")
    parser.add_argument("--qcad-jobs", type=int, default=4)
    parser.add_argument("--dte-workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--flowcad-workers", type=int, default=2)
    parser.add_argument("--lunar-workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--classical-jobs", type=int, default=1)
    args = parser.parse_args()
    if min(args.qcad_jobs, args.flowcad_workers, args.classical_jobs) < 1:
        parser.error("Worker counts must be positive")
    if not args.summary_only and any(path is None for path in
                                     (args.qcad_repo, args.dte_repo, args.tccm_repo)):
        parser.error("Provide --qcad-repo, --dte-repo, and --tccm-repo")

    manifest = args.manifest.resolve()
    config = read_manifest(manifest)
    datasets = config["datasets"]
    expected = case_hashes(datasets)
    work = manifest.parent.parent
    dte_tccm = work / "baselines_dte_tccm" / "per_run.csv"
    sources = {
        "Flow-CAD": work / "flowcad_fixed_1000" / "summary.csv",
        "LUNAR": work / "lunar" / "summary.csv",
        "QCAD": work / "all_qcad" / "per_run.csv",
        "DTE-C": dte_tccm,
        "DTE-NP": dte_tccm,
        "TCCM": dte_tccm,
        **{method: work / "classical_baselines" / method for method in CLASSICAL_METHODS},
    }
    pending = dict(expected)
    print(f"{len(pending)} inputs per method", flush=True)
    if pending and not args.summary_only:
        run_detectors(args, manifest, work, datasets, pending)

    results = {}
    for method, source in sources.items():
        results[method] = validate(read_records(source, method), expected, method)
    status = {"expected_per_method": len(expected),
              "completed": {method: len(records) for method, records in results.items()}}
    status["complete"] = all(records.keys() == expected.keys() for records in results.values())
    if status["complete"]:
        summarize(work, datasets, results)
    temporary = work / "full_coverage.tmp"
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(work / "full_coverage.json")
    print(json.dumps(status), flush=True)
    if not status["complete"]:
        raise SystemExit("The benchmark is incomplete. See full_coverage.json for run counts.")


if __name__ == "__main__":
    main()
