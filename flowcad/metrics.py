"""Within-batch percentile ranks and anomaly detection metrics."""

import numbers
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def _scores(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("scores must be a nonempty finite one-dimensional array")
    return array


def percentile_ranks(scores):
    """Rank scores within this batch on (0, 100], averaging tied ranks.

    Matches scipy.stats.percentileofscore(..., kind='rank') on these same
    observations. These values are ranks, not anomaly probabilities.
    """
    scores = _scores(scores)
    return rankdata(scores, method="average") * (100.0 / len(scores))


def evaluate_scores(labels, scores, n=None):
    """Return trapezoidal PR AUC, ROC AUC and precision at n anomalies.

    The default cutoff is the number of positive labels. This label-dependent
    cutoff is used for evaluation only. PR AUC is not average precision.
    Tied top-n scores follow the original pandas sort_values convention.
    """
    scores = _scores(scores)
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(scores) or not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be a 0/1 vector with one label per score")
    if len(np.unique(labels)) != 2:
        raise ValueError("Both normal and anomalous labels are needed for evaluation")
    if n is None:
        n = int(np.sum(labels == 1))
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or not 1 <= n <= len(scores):
        raise ValueError("n must be an integer between 1 and the number of rows")
    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    top = pd.DataFrame({"label": labels, "score": scores}).sort_values(
        "score", ascending=False,
    ).head(n)
    return {"pr_auc": float(auc(recall, precision)),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "p_at_n": float((top["label"] == 1).sum() / n), "n": int(n)}
