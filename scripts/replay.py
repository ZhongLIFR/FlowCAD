"""Paths and input validation shared by the benchmark scripts."""

from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_REPLAY = ROOT / "outputs" / "qcad_paper_7_seed_replay"
MAIN_REFERENCE = ROOT / "configs" / "qcad_paper_7_seed_manifest.json"
MAIN_MANIFEST = MAIN_REPLAY / "data" / "manifest.json"
MODEL_COMMIT = "icde-2027-fixed-1000"


def read_manifest(path):
    value = json.loads(path.read_bytes())
    if value != json.loads(MAIN_REFERENCE.read_bytes()):
        raise ValueError(f"Input manifest differs from the pinned main benchmark: {path}")
    return value


def verify_model():
    checksums = json.loads((ROOT / "configs" / "flowcad_model_sha256.json").read_bytes())
    for name, expected in checksums.items():
        if sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Flow-CAD model source differs from {MODEL_COMMIT}: {name}")
    return MODEL_COMMIT
