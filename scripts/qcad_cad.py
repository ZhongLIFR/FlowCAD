"""Replay the Python CAD baseline in QCAD, including its legacy scoring rules.

The released implementation fits one full-covariance joint GMM and then
updates a mapping after every row. It uses the last context component's
density, a fixed responsibility denominator, and transposed mapping updates.
Its final score mixes behavioral densities using the prior component weights,
without evaluating the query's context density. These choices differ from
the conditional scoring and normalized EM in Song et al. (2007).
"""

from pathlib import Path

from numba import njit
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


COMPONENTS = 5
ALPHA = 0.001
SWEEPS = 100


@njit
def _legacy_mapping(context_density, behavior_densities, weights):
    n_rows, n_components = behavior_densities.shape
    mapping = np.full((n_components, n_components), 1.0 / n_components)
    responsibilities = np.zeros((n_rows, n_components, n_components))
    suffix = np.zeros((n_rows + 1, n_components, n_components))
    prefix = np.zeros((n_components, n_components))

    for _ in range(SWEEPS):
        # The legacy row-wise update combines fresh values from rows already
        # visited with stored values from the preceding sweep for later rows.
        for row in range(n_rows - 1, -1, -1):
            for context in range(n_components):
                for behavior in range(n_components):
                    suffix[row, context, behavior] = (
                        responsibilities[row, context, behavior] + suffix[row + 1, context, behavior]
                    )
        prefix.fill(0.0)
        for row in range(n_rows):
            for context in range(n_components):
                prior_density = context_density[row] * weights[context]
                for behavior in range(n_components):
                    value = prior_density * behavior_densities[row, behavior] * mapping[context, behavior] / ALPHA
                    responsibilities[row, context, behavior] = value
                    prefix[context, behavior] += value
            for context in range(n_components):
                denominator = 0.0
                for behavior in range(n_components):
                    denominator += prefix[context, behavior] + suffix[row + 1, context, behavior]
                for behavior in range(n_components):
                    numerator = prefix[context, behavior] + suffix[row + 1, context, behavior]
                    mapping[behavior, context] = numerator / (denominator + ALPHA)
    return mapping


def joint_columns(spec):
    columns = list(spec["benchmark_columns"])
    # AverageTestCOD.py excludes this field, unlike the other baseline runner.
    # Use named features so no real field is mistaken for a saved row index.
    if Path(spec["raw_path"]).stem.casefold() == "heartfailure":
        columns.remove("DEATH_EVENT")
    return columns


def score_cad(features, spec):
    """Return label-free raw scores in original table order."""
    columns = joint_columns(spec)
    model = GaussianMixture(n_components=COMPONENTS, random_state=0, covariance_type="full")
    model.fit(features.loc[:, columns].to_numpy(dtype=np.float64))
    context_columns, behavior_columns = spec["context_columns"], spec["behavior_columns"]
    context_index = [columns.index(column) for column in context_columns]
    behavior_index = [columns.index(column) for column in behavior_columns]
    context = features.loc[:, context_columns].to_numpy(dtype=np.float64)
    behavior = features.loc[:, behavior_columns].to_numpy(dtype=np.float64)

    last = COMPONENTS - 1
    context_density = multivariate_normal.pdf(
        context, mean=model.means_[last, context_index],
        cov=model.covariances_[last][np.ix_(context_index, context_index)], allow_singular=True,
    )
    behavior_densities = np.column_stack([
        multivariate_normal.pdf(
            behavior, mean=mean[behavior_index], cov=covariance[np.ix_(behavior_index, behavior_index)],
        )
        for mean, covariance in zip(model.means_, model.covariances_, strict=True)
    ])
    mapping = _legacy_mapping(context_density, behavior_densities, model.weights_)
    scores = -(behavior_densities @ mapping @ model.weights_)
    if not np.isfinite(scores).all():
        raise ValueError("Legacy CAD produced non-finite anomaly scores")
    return scores
