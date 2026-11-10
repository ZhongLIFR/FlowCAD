"""Score point detectors with the settings used by the QCAD benchmark."""

import numpy as np
from numba import njit
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.sod import SOD
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits


@njit
def _shared_neighbor_references(neighbors, reference_size):
    """Count shared neighbors through inverted lists and retain PyOD's tie order."""
    n_rows, n_neighbors = neighbors.shape
    offsets = np.zeros(n_rows + 1, dtype=np.int64)
    for row in range(n_rows):
        for neighbor in neighbors[row]:
            offsets[neighbor + 1] += 1
    offsets = np.cumsum(offsets)

    rows_by_neighbor = np.empty(n_rows * n_neighbors, dtype=np.int64)
    next_position = offsets[:-1].copy()
    for row in range(n_rows):
        for neighbor in neighbors[row]:
            position = next_position[neighbor]
            rows_by_neighbor[position] = row
            next_position[neighbor] += 1

    references = np.empty((n_rows, reference_size), dtype=np.uint32)
    counts = np.empty(n_rows, dtype=np.uint32)
    for row in range(n_rows):
        counts.fill(0)
        for neighbor in neighbors[row]:
            for position in range(offsets[neighbor], offsets[neighbor + 1]):
                counts[rows_by_neighbor[position]] += 1
        # PyOD assigns self the highest count, sorts, then skips self.
        counts[row] = np.iinfo(np.uint32).max
        references[row] = np.argsort(counts)[::-1][1:reference_size + 1]
    return references


class _IndexedSOD(SOD):
    def _snn(self, features):
        search = NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=1).fit(features)
        neighbors = search.kneighbors(return_distance=False)
        return _shared_neighbor_references(neighbors, self.ref_set)


def score_point_baselines(features, seed, methods=None):
    """Yield (method, raw same-table score vector) for the requested detectors."""
    models = {
        "IForest": IForest(random_state=int(seed)),
        "LOF": LOF(),
        "k-NN": KNN(),
        "SOD": _IndexedSOD(),
        "HBOS": HBOS(),
    }
    requested = models.keys() if methods is None else set(methods)
    unknown = requested - models.keys()
    if unknown:
        raise ValueError(f"Unknown point detectors: {', '.join(sorted(unknown))}")
    with threadpool_limits(limits=1):
        for name, model in models.items():
            if name in requested:
                yield name, np.asarray(model.fit(features).decision_scores_)
