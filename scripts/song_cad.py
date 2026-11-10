"""GMM-CAD-Full following Song et al. (2007), Sections II-B and III-C."""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture


COMPONENTS = 5
MAPPING_ITERATIONS = 100
MAPPING_TOLERANCE = 1e-6


def _marginal_log_density(values, means, covariances):
    constant = values.shape[1] * np.log(2 * np.pi)
    columns = []
    for mean, covariance in zip(means, covariances, strict=True):
        lower = np.linalg.cholesky(covariance)
        whitened = solve_triangular(lower, (values - mean).T, lower=True)
        log_determinant = 2 * np.log(lower.diagonal()).sum()
        columns.append(-0.5 * (constant + log_determinant + np.square(whitened).sum(axis=0)))
    return np.column_stack(columns)


def _fit_log_mapping(log_density_pairs):
    n_context, n_behavior = log_density_pairs.shape[1:]
    log_mapping = np.full((n_context, n_behavior), -np.log(n_behavior))
    previous_mean_log_likelihood = -np.inf
    for _ in range(MAPPING_ITERATIONS):
        log_joint = log_density_pairs + log_mapping
        log_density = logsumexp(log_joint, axis=(1, 2), keepdims=True)
        mean_log_likelihood = log_density.mean()
        if mean_log_likelihood - previous_mean_log_likelihood <= MAPPING_TOLERANCE:
            break
        # Normalize responsibilities over component pairs, then normalize the
        # resulting counts over behavioral components within each context row.
        log_counts = logsumexp(log_joint - log_density, axis=0)
        log_mapping = log_counts - logsumexp(log_counts, axis=1, keepdims=True)
        previous_mean_log_likelihood = mean_log_likelihood
    return log_mapping


def score_cad_full(features, spec):
    """Fit and score one table with the paper equations and QCAD's component count.

    This is a reimplementation. The joint Gaussian mixture uses five full
    covariance components and a fixed seed, as in the QCAD benchmark, with
    scikit-learn's default covariance regularization. Context and behavior
    follow the manifest partition; fitting and scoring use no labels.
    """
    context = features.loc[:, spec["context_columns"]].to_numpy(dtype=np.float64)
    behavior = features.loc[:, spec["behavior_columns"]].to_numpy(dtype=np.float64)
    model = GaussianMixture(n_components=COMPONENTS, covariance_type="full", random_state=0)
    model.fit(np.column_stack((context, behavior)))

    split = context.shape[1]
    log_context_joint = np.log(model.weights_) + _marginal_log_density(
        context, model.means_[:, :split], model.covariances_[:, :split, :split]
    )
    log_context_posterior = log_context_joint - logsumexp(log_context_joint, axis=1, keepdims=True)
    log_behavior_density = _marginal_log_density(
        behavior, model.means_[:, split:], model.covariances_[:, split:, split:]
    )
    log_density_pairs = log_context_posterior[:, :, None] + log_behavior_density[:, None, :]
    log_mapping = _fit_log_mapping(log_density_pairs)
    return -logsumexp(log_density_pairs + log_mapping, axis=(1, 2))
