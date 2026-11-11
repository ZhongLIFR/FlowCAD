# Flow-CAD

**Flow Matching for Contextual Anomaly Detection and Explanation in Tabular Data**

Zhong Li¹², Qi Huang², Yuhang Wang³, Yang Cao¹, Thomas Bäck², Niki van Stein², Matthijs van Leeuwen², Lincen Yang²

¹ Great Bay University · ² Leiden University · ³ KU Leuven  
Corresponding author: Lincen Yang ([l.yang@liacs.leidenuniv.nl](mailto:l.yang@liacs.leidenuniv.nl))

Flow-CAD learns a conditional velocity field for behavioral features given their context. It scores each row with MAD-normalized, truncated residuals and reports additive feature contributions.

## Quick start

Use Python 3.12 and Git. Run these commands from this repository:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p .repro
git clone --filter=blob:none --no-checkout https://github.com/ZhongLIFR/QCAD.git .repro/QCAD
python scripts/generate_qcad_paper_replay.py --qcad-repo .repro/QCAD
python scripts/run_qcad_paper_flowcad.py --datasets fish --injection-seeds 42 --workers 1
```

On Windows, use `.venv\Scripts\activate`. The generator reconstructs the 170 input tables used by every method across 20 datasets from pinned QCAD commit `f3b887f3d4466607a9d7147e9715f6b0b27a5954` and verifies each CSV against the input manifest. Data retain their original terms. [Input manifest](configs/qcad_paper_manifest.json) · [QCAD source](https://github.com/ZhongLIFR/QCAD/tree/f3b887f3d4466607a9d7147e9715f6b0b27a5954) · [QCAD paper](https://doi.org/10.1007/s10618-023-00967-z)

## Reproduce the main results

```bash
python scripts/run_qcad_paper_flowcad.py --datasets all --injection-seeds all --fixed-epochs 1000 --workers 2

python -m pip install -r requirements-baselines.txt
git clone --no-checkout https://github.com/vicliv/DTE.git .repro/DTE
git -C .repro/DTE checkout --detach 2c205204cd9bd21d1ffd3dd180a9f4a94bc6b279
git clone --no-checkout https://github.com/ZhongLIFR/TCCM-NIPS.git .repro/TCCM-NIPS
git -C .repro/TCCM-NIPS checkout --detach 363cfa61fd9e786bcb0b905241106dd4e7f52e6f

python scripts/run_qcad_replay_lunar.py --workers 2
python scripts/run_qcad_paper_dte_tccm.py --dte-repo .repro/DTE --tccm-repo .repro/TCCM-NIPS --workers 2

replay=outputs/qcad_paper_replay
python scripts/run_qcad_baselines.py --qcad-repo .repro/QCAD
python scripts/run_qcad_baselines.py --methods CAD-Full
python scripts/run_paired_qcad.py --all-injections --jobs 4 --output "$replay/all_qcad" \
  --score-output "$replay/qcad_in_range/records"

sources=()
for method in LoPAD ROCOD CAD-Full IForest LOF k-NN SOD HBOS; do
  sources+=(--source "$method=$replay/historical_baselines/$method")
done
python scripts/summarize_qcad_paper_full.py \
  --source "Flow-CAD=$replay/flowcad_fixed_1000/summary.csv" \
  --source "LUNAR=$replay/lunar/summary.csv" \
  --source "DTE-C=$replay/baselines_dte_tccm/per_run.csv" \
  --source "DTE-NP=$replay/baselines_dte_tccm/per_run.csv" \
  --source "TCCM=$replay/baselines_dte_tccm/per_run.csv" \
  --source "QCAD=$replay/all_qcad/per_run.csv" \
  "${sources[@]}"
```

The runners save completed cases in `outputs/qcad_paper_replay/` and resume from those records. The summarizer checks the input hashes and writes the run counts to `full_coverage.json` in that directory. Full runs take several hours on CPU. Results may vary across hardware and software platforms.

Each method receives and scores the corresponding perturbed input table. Flow-CAD trains for exactly 1,000 epochs on every table and scores all rows with the resulting model. Injection labels are used only to compute PR AUC, ROC AUC, and P@n. Flow-CAD uses one CPU thread, two width-32 context layers, three width-64 velocity hidden layers, ReLU, Adam with learning rate 0.001, and batch size 128. The raw score uses a MAD floor of 1e-6, clipping at 10, L1 aggregation, and 16 Gaussian draws with 16 time draws each.

The baselines use the fixed settings below. Dataset-specific QCAD neighborhood sizes and the input columns for each run are recorded in the [input manifest](configs/qcad_paper_manifest.json).

| Method | Configuration used |
| --- | --- |
| QCAD | Released per-dataset neighborhood sizes (80–500), mixed-type Gower distance, and weighted leaf quantiles from 10 trees with minimum split size 10 and forest seed 0. The port floors a zero lower-tail IQR at 1e-9. |
| LoPAD | Released Markov blankets and 20 CART fits on 80% subsamples drawn without replacement. |
| ROCOD | Categorical Gower distance cutoff 0.9, with the global linear predictor used for isolated rows. |
| CAD | Five full-covariance GMM components with seed 0. The conditional mapping uses at most 100 EM iterations with tolerance 1e-6. |
| IForest, LOF, k-NN, SOD, HBOS | PyOD 2.0.3 defaults. IForest uses 100 trees and automatic sample size. LOF uses 20 neighbors, k-NN uses the distance to the fifth neighbor, SOD uses 20 neighbors, a 10-row reference set, and alpha 0.8, and HBOS uses 10 bins, alpha 0.1, and tolerance 0.5. |
| LUNAR | PyOD 2.0.3 WEIGHT model, five neighbors, and 200 epochs. The internal 10% validation split selects an epoch using generated MIXED negatives. |
| DTE-C | 400 epochs, batch size 64, learning rate 1e-4, weight decay 5e-4, hidden widths 256/512/256, 400 diffusion steps, and seven bins. |
| DTE-NP | Five neighbors and 1,000 diffusion steps, retaining the released same-table neighbor convention. |
| TCCM | 100 epochs, batch size 64, and learning rate 0.001. |

Flow-CAD and LUNAR one-hot encode designated categorical context columns. DTE and TCCM use stored numeric values and ordinal codes for text categories. LoPAD and the five PyOD point baselines above use the released benchmark feature columns. The stochastic Flow-CAD, LUNAR, DTE, TCCM, LoPAD and IForest runs use training seed 201 plus the injection index.

All fourteen methods are evaluated on the same 170 reconstructed tables, paired by dataset and injection seed. Abalone, ElNino, GasEmission, Maintenance, Parkinson, and PowerPlant use seeds 42–46. The remaining datasets use seeds 42–51. CAD uses Song et al.'s full GMM variant with five mixture components, labeled `CAD-Full` in the scripts and results. Per-dataset ranks use the displayed two-decimal means and average ties. [CAD paper](https://doi.org/10.1109/TKDE.2007.1009) · [Main results](results/main_results.csv) · [Per-run results](results/main_per_run.csv)

## Reproduce the scoring sweeps

After the full Flow-CAD run:

```bash
python scripts/run_qcad_paper_ablation.py --workers 2
```

The sweeps vary normalization, truncation, aggregation norms, and Monte Carlo sampling budgets using the same 1,000-epoch model for each input. The script verifies that the default scoring configuration reproduces the main experiment. [Scoring sweep results](results/ablation_sensitivity.csv)

After generating the benchmark inputs, run the categorical context comparison:

```bash
python scripts/run_context_encoding_ablation.py --workers 4
```

FishWeight and QSRanking each use ten paired injections. Both one-hot encoding and sorted category IDs scaled to [0, 1] use 1,000 training epochs. The runner checks the one-hot results against the main replay and reports top-n recovery on QSRanking for countries with at most five rows and for the other countries. [Per-run encoding results](results/context_encoding_ablation_per_run.csv) · [Per-dataset summary](results/context_encoding_ablation_summary.csv)

## Paired diagnostics

After regenerating the input tables:

```bash
python scripts/run_paired_qcad.py
python scripts/run_conditional_distribution_diagnostic.py --workers 2
python scripts/run_feature_localization.py --qcad-repo .repro/QCAD --workers 2
```

The paired accuracy results are the subset of the main replay for the eight smallest tables and injection seeds 42–44. [Paired accuracy results](results/paired_qcad_accuracy.csv)

The synthetic experiment compares conditional Flow-CAD, constant-context Flow-CAD, and a conditional sample-mean residual. The sign-parity setting matches the empirical distributions of all single features and all feature pairs across contexts. The mean-shift control changes conditional means. Both settings swap context labels while preserving the behavior rows and context counts. The learned flow models select epochs using the label-free 5% contrast score margin from [AutoGAD](https://doi.org/10.1007/s10618-025-01115-5). [Synthetic results](results/conditional_distribution_per_run.csv)

After the main Flow-CAD replay, reproduce the conditional-mean MLP comparison on all 170 benchmark tables:

```bash
python scripts/run_qcad_conditional_mean.py --workers 2
python scripts/run_qcad_in_range_flow.py --workers 2
python scripts/summarize_conditional_mean_comparison.py
python scripts/summarize_conditional_mean_comparison.py --comparator qcad
```

The MLP uses the same context encoder and hidden widths, with its state and time inputs fixed at zero. Both models use the same training seeds and run for 1,000 epochs. The range-restricted evaluation keeps rows with all scaled behavioral values in `[0, 1]` within a tolerance of `1e-8`. A trial must have both classes after filtering. The QCAD comparison checks full-table metrics against the main replay before filtering. [Per-dataset results](results/conditional_mean_per_dataset.csv) · [Paired runs](results/conditional_mean_per_run.csv)

For the QCAD range comparison, see the [per-dataset results](results/qcad_in_range_per_dataset.csv) and [paired runs](results/qcad_in_range_per_run.csv).

Reproduce the within-context behavioral-swap comparison:

```bash
python scripts/generate_qcad_context_swap.py --qcad-repo .repro/QCAD
python scripts/run_qcad_context_swap.py --workers 4
python scripts/summarize_qcad_context_swap.py
```

This diagnostic covers all 12 benchmarks with at least two behavioral features and all 95 original dataset/seed cases. It groups rows using context only, forms disjoint pairs, and swaps one differing behavioral coordinate. Each behavioral column retains exactly the same empirical distribution within its stratum. Flow-CAD, QCAD, and the capacity-matched conditional-mean MLP use identical swapped tables. Both neural methods select checkpoints with the label-free 5% CSM rule. The summary compares dataset-level seed means and reports wins, ties, and losses. [Per-dataset results](results/context_swap_per_dataset.csv) · [Paired runs](results/context_swap_per_run.csv) · [Win/tie/loss counts](results/context_swap_win_tie_loss.csv)

The localization experiment changes one behavioral coordinate per injected row across six datasets. Flow-CAD uses the 5% CSM rule, and we compare its attributions with uniform feature ranking. [Feature localization results](results/feature_localization_per_run.csv)

After the main Flow-CAD and QCAD runs, use `python scripts/run_paired_context_timing.py --qcad-results outputs/qcad_paper_replay/all_qcad` for the serial seed-42 comparison. The run alternates method order on one Apple M4 CPU thread with macOS 15.7.3, PyTorch 2.6.0, and scikit-learn 1.5.2. End-to-end timing includes the 1,000 training epochs for Flow-CAD. Fitted Flow-CAD timing covers fresh draws, scores, and contributions. [Paired timing results](results/paired_qcad_timing.csv)

## Python API

`from flowcad import FlowCAD` exposes `fit(context, behavior)` and `score_samples(context, behavior, return_contributions=True)`. The original `run.py` CLI also accepts the earlier prepared inputs via `--data-dir`. [Earlier release and inputs](https://github.com/ZhongLIFR/FlowCAD/tree/20e3d74da1809771b865ce05f97ef908b34f5483)

## Citation

```bibtex
@misc{li2026flowcad,
  title = {Flow-CAD: Flow Matching for Contextual Anomaly Detection and Explanation in Tabular Data},
  author = {Li, Zhong and Huang, Qi and Wang, Yuhang and Cao, Yang and
            B{\"a}ck, Thomas and van Stein, Niki and van Leeuwen, Matthijs and Yang, Lincen},
  year = {2026},
  note = {Research manuscript},
  url = {https://github.com/ZhongLIFR/FlowCAD}
}
```

## Acknowledgments

This publication is part of the project Digital Twin with project number P18-03 of the research programme TTW Perspective, which is (partly) financed by the Dutch Research Council (NWO).

## License

The Flow-CAD code is released under the [MIT License](LICENSE).
