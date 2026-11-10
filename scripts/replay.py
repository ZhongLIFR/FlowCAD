"""Paths and input validation shared by the benchmark scripts."""

from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "outputs" / "qcad_paper_replay"
REFERENCE = ROOT / "configs" / "qcad_paper_manifest.json"
MANIFEST = REPLAY / "data" / "manifest.json"
MODEL_COMMIT = "20e3d74da1809771b865ce05f97ef908b34f5483"


def read_manifest(path):
    value = json.loads(path.read_bytes())
    if value != json.loads(REFERENCE.read_bytes()):
        raise ValueError(f"Input manifest differs from the published replay: {path}")
    return value


def verify_model():
    checksums = json.loads((ROOT / "configs" / "flowcad_model_sha256.json").read_bytes())
    for name, expected in checksums.items():
        if sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Flow-CAD model source differs from {MODEL_COMMIT}: {name}")
    return MODEL_COMMIT
