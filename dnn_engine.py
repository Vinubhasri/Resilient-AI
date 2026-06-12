"""
backend/dnn_engine.py
─────────────────────
Pure-NumPy DNN: Layer, DNN classes with forward/backward pass,
error injection, health-check, and serialisation helpers.
"""

import numpy as np
import json
import time
import random


# ── Activations ────────────────────────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)

def relu_d(x):
    return (x > 0).astype(float)

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def cross_entropy(pred, y):
    eps = 1e-9
    return -np.mean(np.sum(y * np.log(pred + eps), axis=1))


# ── Layer ──────────────────────────────────────────────────────────────────────

class Layer:
    def __init__(self, n_in: int, n_out: int, name: str = "layer"):
        self.name       = name
        self.n_in       = n_in
        self.n_out      = n_out
        self.W          = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        self.b          = np.zeros(n_out)
        self.dW         = np.zeros_like(self.W)
        self.db         = np.zeros_like(self.b)
        self.healthy    = True
        self.error_type = None
        # keep a clean snapshot for diff reporting
        self._orig_W    = self.W.copy()
        self._orig_b    = self.b.copy()
        self.input      = None
        self.z          = None
        self.a          = None

    # ── Forward ──
    def forward(self, x: np.ndarray, activation: str = "relu") -> np.ndarray:
        self.input = x
        self.z     = x @ self.W + self.b
        if activation == "relu":
            self.a = relu(self.z)
        elif activation == "softmax":
            self.a = softmax(self.z)
        elif activation == "sigmoid":
            self.a = sigmoid(self.z)
        else:
            self.a = self.z
        return self.a

    # ── Backward ──
    def backward(self, delta: np.ndarray, activation: str = "relu") -> np.ndarray:
        if activation == "relu":
            delta = delta * relu_d(self.z)
        self.dW = self.input.T @ delta / len(self.input)
        self.db = delta.mean(axis=0)
        return delta @ self.W.T

    def apply_gradients(self, lr: float) -> None:
        self.W -= lr * self.dW
        self.b -= lr * self.db

    # ── Error Injection ──
    def inject_error(self, error_type: str) -> None:
        self.healthy    = False
        self.error_type = error_type
        if error_type == "weight_explosion":
            self.W *= random.uniform(50, 200)
        elif error_type == "dead_neurons":
            mask = np.random.rand(*self.W.shape) < 0.70
            self.W[mask] = 0.0
        elif error_type == "gradient_vanish":
            self.W *= 0.0001
        elif error_type == "noisy_weights":
            self.W += np.random.randn(*self.W.shape) * 10.0
        elif error_type == "sign_flip":
            self.W = -self.W
        elif error_type == "bias_shift":
            self.b += np.random.randn(*self.b.shape) * 100.0

    # ── Stats ──
    def stats(self) -> dict:
        return {
            "name"   : self.name,
            "shape"  : list(self.W.shape),
            "mean"   : float(np.mean(np.abs(self.W))),
            "std"    : float(np.std(self.W)),
            "max"    : float(np.max(np.abs(self.W))),
            "min"    : float(np.min(np.abs(self.W))),
            "zeros"  : float(np.mean(self.W == 0)),
            "norm"   : float(np.linalg.norm(self.W)),
            "healthy": self.healthy,
            "error"  : self.error_type,
        }

    # ── Serialise ──
    def to_dict(self) -> dict:
        return {
            "name"      : self.name,
            "n_in"      : self.n_in,
            "n_out"     : self.n_out,
            "W"         : self.W.tolist(),
            "b"         : self.b.tolist(),
            "healthy"   : self.healthy,
            "error_type": self.error_type,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Layer":
        layer = cls(d["n_in"], d["n_out"], d["name"])
        layer.W          = np.array(d["W"])
        layer.b          = np.array(d["b"])
        layer.healthy    = d.get("healthy", True)
        layer.error_type = d.get("error_type", None)
        layer._orig_W    = layer.W.copy()
        layer._orig_b    = layer.b.copy()
        return layer


# ── DNN ────────────────────────────────────────────────────────────────────────

class DNN:
    """
    A simple feed-forward network usable as a standalone repair target.
    Architecture is flexible — defined by `arch` list of neuron counts.
    Default: [16, 32, 64, 32, 4]
    """

    def __init__(self, arch: list = None, name: str = "DNN"):
        if arch is None:
            arch = [16, 32, 64, 32, 4]
        self.arch       = arch
        self.name       = name
        self.layers     = []
        self.history    = []          # [{loss, acc}, ...]
        self.repair_log = []          # repair events
        self.repaired   = False
        self.source_file= None        # path of uploaded .py model
        self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self._build_layers()

    def _build_layers(self) -> None:
        names = []
        for i in range(len(self.arch) - 1):
            if i == 0:
                names.append("Input→H1")
            elif i == len(self.arch) - 2:
                names.append(f"H{i}→Output")
            else:
                names.append(f"H{i}→H{i+1}")
        self.layers = [
            Layer(self.arch[i], self.arch[i + 1], names[i])
            for i in range(len(self.arch) - 1)
        ]

    # ── Inference ──
    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x
        for i, layer in enumerate(self.layers):
            act = "softmax" if i == len(self.layers) - 1 else "relu"
            h = layer.forward(h, act)
        return h

    # ── Training step ──
    def backward(self, x: np.ndarray, y: np.ndarray, lr: float = 0.005) -> float:
        pred  = self.forward(x)
        loss  = cross_entropy(pred, y)
        delta = (pred - y) / len(x)
        for i in reversed(range(len(self.layers))):
            act   = "softmax" if i == len(self.layers) - 1 else "relu"
            delta = self.layers[i].backward(delta, act)
            self.layers[i].apply_gradients(lr)
        return float(loss)

    def train_step(self, n_samples: int = 256) -> tuple:
        n_out  = self.arch[-1]
        n_in   = self.arch[0]
        X      = np.random.randn(n_samples, n_in)
        y_idx  = np.random.randint(0, n_out, n_samples)
        Y      = np.eye(n_out)[y_idx]
        loss   = self.backward(X, Y)
        pred   = self.forward(X)
        acc    = float((pred.argmax(1) == y_idx).mean())
        return loss, acc

    def evaluate(self, n_samples: int = 512) -> tuple:
        n_out = self.arch[-1]
        n_in  = self.arch[0]
        X     = np.random.randn(n_samples, n_in)
        y_idx = np.random.randint(0, n_out, n_samples)
        Y     = np.eye(n_out)[y_idx]
        pred  = self.forward(X)
        loss  = cross_entropy(pred, Y)
        acc   = float((pred.argmax(1) == y_idx).mean())
        return float(loss), acc

    # ── Health check ──
    def health_check(self) -> list:
        issues = []
        for layer in self.layers:
            s = layer.stats()
            if s["max"]   > 100:   issues.append([layer.name, "weight_explosion"])
            if s["zeros"] > 0.50:  issues.append([layer.name, "dead_neurons"])
            if s["mean"]  < 1e-4:  issues.append([layer.name, "gradient_vanish"])
            if s["std"]   > 20:    issues.append([layer.name, "noisy_weights"])
        return issues

    # ── State snapshot ──
    def get_state(self) -> dict:
        loss, acc = self.evaluate()
        return {
            "name"        : self.name,
            "arch"        : self.arch,
            "loss"        : round(loss, 4),
            "accuracy"    : round(acc * 100, 2),
            "layers"      : [l.stats() for l in self.layers],
            "layer_names" : [l.name for l in self.layers],
            "issues"      : self.health_check(),
            "history"     : self.history[-60:],
            "repair_log"  : self.repair_log[-30:],
            "repaired"    : self.repaired,
            "source_file" : self.source_file,
            "created_at"  : self.created_at,
        }

    # ── Serialise ──
    def to_dict(self) -> dict:
        return {
            "name"       : self.name,
            "arch"       : self.arch,
            "layers"     : [l.to_dict() for l in self.layers],
            "history"    : self.history,
            "repair_log" : self.repair_log,
            "repaired"   : self.repaired,
            "source_file": self.source_file,
            "created_at" : self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DNN":
        dnn             = cls.__new__(cls)
        dnn.name        = d.get("name", "DNN")
        dnn.arch        = d["arch"]
        dnn.layers      = [Layer.from_dict(ld) for ld in d["layers"]]
        dnn.history     = d.get("history", [])
        dnn.repair_log  = d.get("repair_log", [])
        dnn.repaired    = d.get("repaired", False)
        dnn.source_file = d.get("source_file", None)
        dnn.created_at  = d.get("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        return dnn
