"""
backend/model_loader.py
───────────────────────
Parses an uploaded Python (.py) model file and constructs
the appropriate DNN architecture from it.

Supported formats in the uploaded .py:
  - MODEL_ARCH = [16, 32, 64, 4]          (list of layer sizes)
  - MODEL_NAME = "MyNet"                   (optional name)
  - LAYERS = [(16,32,"Input"), (32,4,"Out")] (tuple format)
  - Any sequential layer definition pattern
"""

import re
import ast
import time
import numpy as np
from backend.dnn_engine import DNN


class ModelParseError(Exception):
    pass


def _extract_arch_from_source(source: str) -> tuple:
    """
    Try multiple heuristics to pull an architecture list from Python source.
    Returns (arch_list, model_name, notes).
    """
    name  = "UploadedModel"
    notes = []

    # ── Heuristic 1: MODEL_ARCH = [...]  ──────────────────────────────────────
    m = re.search(r"MODEL_ARCH\s*=\s*(\[[^\]]+\])", source)
    if m:
        try:
            arch = ast.literal_eval(m.group(1))
            if isinstance(arch, list) and all(isinstance(n, int) for n in arch):
                notes.append("Parsed from MODEL_ARCH variable.")
                mn = re.search(r'MODEL_NAME\s*=\s*["\']([^"\']+)["\']', source)
                if mn:
                    name = mn.group(1)
                return arch, name, notes
        except Exception:
            pass

    # ── Heuristic 2: LAYERS = [(in, out, name), ...] ──────────────────────────
    m = re.search(r"LAYERS\s*=\s*(\[[\s\S]*?\])", source)
    if m:
        try:
            layers = ast.literal_eval(m.group(1))
            arch   = [layers[0][0]] + [t[1] for t in layers]
            notes.append("Parsed from LAYERS list.")
            return arch, name, notes
        except Exception:
            pass

    # ── Heuristic 3: nn.Linear(a, b) patterns (PyTorch style) ────────────────
    linears = re.findall(r"nn\.Linear\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", source)
    if linears:
        sizes = [int(linears[0][0])] + [int(b) for _, b in linears]
        notes.append(f"Parsed {len(linears)} nn.Linear layers.")
        cn = re.search(r"class\s+(\w+)\s*\(", source)
        if cn:
            name = cn.group(1)
        return sizes, name, notes

    # ── Heuristic 4: Dense/Sequential (Keras style) ───────────────────────────
    dense = re.findall(r"Dense\s*\(\s*(\d+)", source)
    if dense:
        input_m = re.search(r"input_shape\s*=\s*\(?(\d+)", source)
        start   = int(input_m.group(1)) if input_m else 16
        sizes   = [start] + [int(d) for d in dense]
        notes.append(f"Parsed {len(dense)} Dense (Keras) layers.")
        return sizes, name, notes

    # ── Heuristic 5: layer sizes as plain list anywhere ───────────────────────
    lists = re.findall(r"\[\s*(\d+(?:\s*,\s*\d+){2,})\s*\]", source)
    for raw in lists:
        try:
            candidate = [int(x.strip()) for x in raw.split(",")]
            if 2 < len(candidate) < 10 and all(4 <= n <= 4096 for n in candidate):
                notes.append("Inferred architecture from numeric list.")
                return candidate, name, notes
        except Exception:
            pass

    raise ModelParseError(
        "Could not detect a valid architecture in the uploaded file.\n"
        "Please define MODEL_ARCH = [input_size, hidden..., output_size] in your .py file."
    )


def load_model_from_source(source: str, filename: str) -> DNN:
    """
    Parse source code, construct and pre-train a DNN, return it.
    """
    arch, name, notes = _extract_arch_from_source(source)

    # Sanity checks
    if len(arch) < 2:
        raise ModelParseError("Architecture must have at least 2 sizes (input, output).")
    if any(n < 1 for n in arch):
        raise ModelParseError("All layer sizes must be >= 1.")
    if any(n > 4096 for n in arch):
        raise ModelParseError("Layer sizes > 4096 not supported in the demo engine.")

    dnn             = DNN(arch=arch, name=name)
    dnn.source_file = filename

    # Pre-train a few steps so the model has a baseline
    for _ in range(40):
        loss, acc = dnn.train_step()
        dnn.history.append({"loss": round(loss, 4), "acc": round(acc * 100, 2)})

    # Append parse notes to repair log
    for note in notes:
        dnn.repair_log.append({
            "time"    : time.strftime("%H:%M:%S"),
            "message" : note,
            "strategy": "Model Loader",
            "layer"   : None,
            "before"  : None,
            "after"   : None,
            "status"  : "info",
        })

    return dnn


def validate_source(source: str) -> dict:
    """Quick validation — returns {valid, arch, name, error}"""
    try:
        arch, name, notes = _extract_arch_from_source(source)
        return {"valid": True, "arch": arch, "name": name, "notes": notes, "error": None}
    except ModelParseError as e:
        return {"valid": False, "arch": None, "name": None, "notes": [], "error": str(e)}
