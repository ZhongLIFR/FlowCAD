# Flow-CAD

**Flow Matching for Contextual Anomaly Detection and Explanation in Tabular Data**

Zhong Li¹², Qi Huang², Yuhang Wang³, Yang Cao¹, Thomas Bäck², Niki van Stein², Matthijs van Leeuwen², Lincen Yang²

¹ Great Bay University · ² Leiden University · ³ KU Leuven  
Corresponding author: Lincen Yang ([l.yang@liacs.leidenuniv.nl](mailto:l.yang@liacs.leidenuniv.nl))

Flow-CAD learns a conditional velocity field for behavioral features given their context. It assigns row-level anomaly scores using median absolute deviation (MAD)-normalized and truncated velocity residuals, and decomposes each score into additive behavioral-feature contributions.

## Main benchmark

The main experiment contains 20 tabular datasets and seven fixed injection seeds (43–49), giving 140 input tables. All 14 methods are evaluated on the same tables and paired by dataset and injection seed. Flow-CAD uses 1,000 training epochs for every run; injection seed `s` is paired with training seed `s + 159` (202–208).

The input specification and checksums are fixed in [`configs/qcad_paper_7_seed_manifest.json`](configs/qcad_paper_7_seed_manifest.json). The benchmark is reconstructed from the pinned [QCAD repository](https://github.com/ZhongLIFR/QCAD/tree/f3b887f3d4466607a9d7147e9715f6b0b27a5954).

## Installation

Python 3.12 is recommended.
Run the commands below from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

mkdir -p .repro
git clone --filter=blob:none --no-checkout https://github.com/ZhongLIFR/QCAD.git .repro/QCAD
git -C .repro/QCAD checkout --detach f3b887f3d4466607a9d7147e9715f6b0b27a5954

python scripts/generate_qcad_paper_replay.py --qcad-repo .repro/QCAD
python scripts/validate_main_manifest.py \
  --manifest outputs/qcad_paper_7_seed_replay/data/manifest.json \
  --check-files
```

## Run Flow-CAD

A single example:

```bash
python scripts/run_qcad_paper_flowcad.py \
  --datasets fish --injection-seeds 43 \
  --fixed-epochs 1000 --workers 1
```

All 140 main-benchmark runs:

```bash
python scripts/run_qcad_paper_flowcad.py \
  --datasets all --injection-seeds all \
  --fixed-epochs 1000 --workers 2
```

Completed cases are saved under `outputs/qcad_paper_7_seed_replay/`, so interrupted runs can resume.

## Run the complete comparison

Install the baseline dependencies and pinned external implementations:

```bash
python -m pip install -r requirements-baselines.txt

git clone --no-checkout https://github.com/vicliv/DTE.git .repro/DTE
git -C .repro/DTE checkout --detach 2c205204cd9bd21d1ffd3dd180a9f4a94bc6b279

git clone --no-checkout https://github.com/ZhongLIFR/TCCM-NIPS.git .repro/TCCM-NIPS
git -C .repro/TCCM-NIPS checkout --detach 363cfa61fd9e786bcb0b905241106dd4e7f52e6f

python scripts/run_qcad_paper_benchmark.py \
  --qcad-repo .repro/QCAD \
  --dte-repo .repro/DTE \
  --tccm-repo .repro/TCCM-NIPS
```

The benchmark runner verifies input hashes, executes the missing cases, and produces the aggregate summary only after all 14 methods cover all 140 inputs.

## Additional experiments

The scoring ablations reuse the same fixed 1,000-epoch models:

```bash
python scripts/run_qcad_paper_ablation.py \
  --datasets all --injection-seeds all --workers 4
```

After the main Flow-CAD runs finish, the categorical-context comparison can be reproduced with:

```bash
python scripts/run_context_encoding_ablation.py --workers 4
```

The held-out sign-parity experiment is independent of the tabular benchmark inputs:

```bash
python scripts/run_heldout_dependence.py --workers 2
```

## Fixed Flow-CAD configuration

| Component | Value |
| --- | --- |
| Context encoder | 2 × 32 MLP layers |
| Velocity network | 3 × 64 MLP layers, ReLU |
| Optimizer | Adam, learning rate 0.001 |
| Batch size | 128 |
| Training epochs | 1,000 |
| Score aggregation | L1 |
| Clipping threshold \(\tau\) | 10 |
| Monte Carlo budget | 16 Gaussian × 16 time draws |

Labels are not used for training or model selection. They are used only to compute PR AUC, ROC AUC, and P@n after scoring, where `n` is the number of injected anomalies in the corresponding table.

## Results

Aggregate results are provided in:

- [`results/main_results.csv`](results/main_results.csv): per-dataset means, standard deviations, and ranks for all methods.
- [`results/ablation_sensitivity.csv`](results/ablation_sensitivity.csv): aggregate scoring ablations and sensitivity results.
- [`results/context_encoding_ablation_summary.csv`](results/context_encoding_ablation_summary.csv): categorical-context encoding comparison.
- [`results/heldout_dependence_summary.csv`](results/heldout_dependence_summary.csv): aggregate results for the held-out sign-parity experiment.

Across the 20 datasets, Flow-CAD obtains average ranks of **1.900** for PR AUC, **2.050** for ROC AUC, and **1.825** for P@n, ranking first among the 14 methods for all three metrics.

## Repository structure

```text
flowcad/      Flow-CAD model, scoring, preprocessing, and metrics
configs/      dataset settings and pinned input manifests
scripts/      benchmark generation, runners, and summarization
results/      aggregate results reported with the paper
```

## License

Released under the [MIT License](LICENSE).
