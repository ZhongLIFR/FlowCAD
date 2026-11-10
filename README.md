# Flow-CAD

**Flow Matching for Contextual Anomaly Detection and Explanation in Tabular Data**

Zhong Li¹², Qi Huang², Yuhang Wang³, Yang Cao¹, Thomas Bäck², Niki van Stein², Matthijs van Leeuwen², Lincen Yang²

¹ Great Bay University · ² Leiden University · ³ KU Leuven  
Corresponding author: Lincen Yang ([l.yang@liacs.leidenuniv.nl](mailto:l.yang@liacs.leidenuniv.nl))

Flow-CAD learns a conditional velocity field for behavioral features given their context. Anomaly scores use MAD-normalized, truncated velocity residuals and provide additive contributions for each behavioral feature.

## Installation

Use Python 3.12 and run commands from this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run

```bash
python run.py --datasets energy --seeds 201
python run.py --datasets all --seeds 201,202,203,204,205
```

The default implements the paper's full-table protocol. For each seed and dataset, train once and score checkpoints at 50, 100, 200, 500, 1,000, 2,000, 5,000, and 10,000 epochs using shared Monte Carlo draws. Select the largest [contrast score margin (AutoGAD, Eq. 3)](https://link.springer.com/article/10.1007/s10618-025-01115-5) between the highest-scoring 5% of rows and the remaining rows. Ties select the earliest epoch. Training and selection use only features and raw scores. Evaluation reads anomaly labels after selection.

To run a fixed budget through the standalone Python API and its scoring stream, pass `--epochs`:

```bash
python run.py --datasets energy --seeds 201 --epochs 1
```

Results are saved under `outputs/` as per-run metrics, mean and sample standard deviation, settings, CSM records in `selections/`, and per-instance scores and contributions in `scores/`. Use `--output-dir` to choose a fresh destination.

## Files and settings

```text
flowcad/              Model, data loading, and evaluation
run.py                Benchmark entry point
configs/datasets.json Feature partitions
data/GenData/         20 prepared benchmark datasets
results/main_results.csv  Aggregate comparison results
```

The CSM protocol uses a single CPU thread, a two-layer context encoder of width 32, three velocity-network hidden layers of width 64, ReLU, Adam with learning rate 0.001, batch size 128, MAD floor 1e-6, truncation threshold 10, and Monte Carlo budgets Kz=Kt=16. Each noise vector is reused across the inner time draws.

Metrics are trapezoidal PR AUC, ROC AUC, and Precision@n. Larger scores indicate more anomalous behavior. The Python API is available as `from flowcad import FlowCAD`; `fit(context, behavior)` trains the model and `score_samples(context, behavior, return_contributions=True)` returns raw scores and feature contributions.

The prepared data follow the [QCAD benchmark](https://github.com/ZhongLIFR/QCAD) and retain their original data terms. In `results/main_results.csv`, Flow-CAD uses the five-seed CSM protocol and the 13 baseline rows per metric and dataset come from archived runs.

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
