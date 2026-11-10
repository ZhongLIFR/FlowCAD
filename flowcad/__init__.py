"""Flow-CAD contextual anomaly detection and behavioral score attribution."""

from .detector import FlowCAD
from .metrics import evaluate_scores, percentile_ranks

__version__ = "0.1.0"
__all__ = ["FlowCAD", "evaluate_scores", "percentile_ranks"]
