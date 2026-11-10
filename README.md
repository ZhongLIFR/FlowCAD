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

On Windows, use `.venv\Scripts\activate`. The generator reconstructs 170 input tables across 20 datasets from pinned QCAD commit `f3b887f3d4466607a9d7147e9715f6b0b27a5954` and verifies each CSV against the input manifest. Data retain their original terms. [Input manifest](configs/qcad_paper_manifest.json) · [QCAD source](https://github.com/ZhongLIFR/QCAD/tree/f3b887f3d4466607a9d7147e9715f6b0b27a5954) · [QCAD paper](https://doi.org/10.1007/s10618-023-00967-z)

## Reproduce the main results

```bash
python scripts/run_qcad_paper_flowcad.py --datasets all --injection-seeds all --workers 2

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
  --source "Flow-CAD=$replay/flowcad_csm05/summary.csv" \
  --source "LUNAR=$replay/lunar/summary.csv" \
  --source "DTE-C=$replay/baselines_dte_tccm/per_run.csv" \
  --source "DTE-NP=$replay/baselines_dte_tccm/per_run.csv" \
  --source "TCCM=$replay/baselines_dte_tccm/per_run.csv" \
  --source "QCAD=$replay/all_qcad/per_run.csv" \
  "${sources[@]}"
```

The runners save completed cases in `outputs/qcad_paper_replay/` and resume from those records. Full runs take several hours on CPU. Results may vary across hardware and software platforms.

Each replay uses the same perturbed inputs and scores every row. Flow-CAD trains on the full table and selects one of eight training checkpoints from 50 to 10,000 epochs using the contrast score margin between the highest-scoring 5% of rows and the remainder. The scores reuse Monte Carlo draws across checkpoints. The earliest checkpoint wins ties. Ground-truth anomaly labels enter only the final PR AUC, ROC AUC and P@n calculations. [AutoGAD](https://doi.org/10.1007/s10618-025-01115-5)

LUNAR uses the PyOD defaults and their internal split. Flow-CAD and LUNAR use one-hot categorical inputs. DTE and TCCM use the stored numeric values and ordinal codes for text categories. DTE-NP retains the same-table neighbor convention of the released implementation.

IForest, LOF, k-NN, SOD, and HBOS use PyOD 2.0.3 and the benchmark feature columns. IForest uses seed 201 plus the injection index. LoPAD uses the released Markov blankets and 20 CART subsamples. ROCOD uses the benchmark's categorical Gower neighborhoods and the global predictor for rows without neighbors. QCAD uses mixed-type Gower neighborhoods and weighted leaf quantiles.

Flow-CAD and all thirteen baselines use the same 170 reconstructed tables. CAD-Full implements Song et al.'s GMM-CAD-Full with five mixture components and the benchmark context and behavior columns. Per-dataset ranks use the two-decimal displayed means and average ties. [CAD-Full paper](https://doi.org/10.1109/TKDE.2007.1009) · [Main results](results/main_results.csv)

## Reproduce the scoring sweeps

After the full Flow-CAD run:

```bash
python scripts/run_qcad_paper_ablation.py --workers 2
```

The sweeps cover normalization and truncation, aggregation norms, and Monte Carlo sampling budgets. The script uses the same selected checkpoints and verifies that default scores match the main experiment. [Scoring sweep results](results/ablation_sensitivity.csv)

## Paired diagnostics

After regenerating the input tables:

```bash
python scripts/run_paired_qcad.py
python scripts/run_conditional_distribution_diagnostic.py --workers 2
python scripts/run_feature_localization.py --qcad-repo .repro/QCAD --workers 2
```

The paired accuracy comparison covers the eight smallest tables and injection seeds 42–44. The QCAD port uses a mixed-type Gower distance and the maintained scikit-learn forest backend. [Paired accuracy results](results/paired_qcad_accuracy.csv)

The synthetic experiment compares conditional Flow-CAD, constant-context Flow-CAD, and a conditional sample-mean residual. The sign-parity setting matches the empirical distributions of all single features and all feature pairs across contexts. The mean-shift control changes conditional means. Both settings swap context labels while preserving the behavior rows and context counts. [Synthetic results](results/conditional_distribution_per_run.csv)

After the main Flow-CAD replay, reproduce the conditional-mean MLP comparison on all 170 benchmark tables:

```bash
python scripts/run_qcad_conditional_mean.py --workers 2
python scripts/run_qcad_in_range_flow.py --workers 2
python scripts/summarize_conditional_mean_comparison.py
python scripts/summarize_conditional_mean_comparison.py --comparator qcad
```

The MLP uses the same context encoder and hidden widths, with its state and time inputs fixed at zero. Both models use the same training seeds and choose their checkpoints independently with the 5% CSM rule. The range-restricted evaluation reuses the same scores and keeps rows with all scaled behavioral values in `[0, 1]` within a tolerance of `1e-8`. A trial must have both classes after filtering. The QCAD comparison uses the row scores saved by the main replay and checks their full-table metrics before filtering. [Macro results](results/conditional_mean_macro.csv) · [Per-dataset results](results/conditional_mean_per_dataset.csv) · [Paired runs](results/conditional_mean_per_run.csv)

[QCAD range comparison](results/qcad_in_range_macro.csv) · [Per-dataset results](results/qcad_in_range_per_dataset.csv) · [Paired runs](results/qcad_in_range_per_run.csv)

The localization experiment changes one behavioral coordinate per injected row across six datasets. It compares Flow-CAD with uniform feature ranking and reports joint row-and-feature recovery for Flow-CAD. [Feature localization results](results/feature_localization_per_run.csv)

After the Flow-CAD and paired QCAD runs, use `python scripts/run_paired_context_timing.py` for the serial seed-42 timing comparison. The recorded run alternated method order and used one CPU thread on an Apple M4 with macOS 15.7.3, PyTorch 2.6.0, and scikit-learn 1.5.2. End-to-end timing includes CSM for Flow-CAD. Fitted Flow-CAD timing covers fresh draws, scores, and contributions. [Paired timing results](results/paired_qcad_timing.csv)

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
