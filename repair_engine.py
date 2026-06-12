"""
backend/repair_engine.py
────────────────────────
All 5 repair strategies + auto-repair dispatcher.
Each strategy returns a list of log-entry dicts.
"""

import numpy as np
import time
from backend.dnn_engine import DNN, Layer


class RepairEngine:
    def __init__(self, dnn: DNN):
        self.dnn = dnn

    # ── Internal logger ────────────────────────────────────────────────────────

    def _log(self, message: str, strategy: str,
             layer_name: str = None,
             before=None, after=None) -> dict:
        entry = {
            "time"    : time.strftime("%H:%M:%S"),
            "message" : message,
            "strategy": strategy,
            "layer"   : layer_name,
            "before"  : round(float(before), 6) if before is not None else None,
            "after"   : round(float(after),  6) if after  is not None else None,
            "status"  : "success",
        }
        self.dnn.repair_log.append(entry)
        return entry

    # ── Helper: get layer list ──────────────────────────────────────────────

    def _resolve_layers(self, layer_idx):
        if layer_idx is not None:
            return [self.dnn.layers[layer_idx]]
        return self.dnn.layers

    # ══════════════════════════════════════════════════════════════════════════
    #  STRATEGY 1 — Weight Clipping
    # ══════════════════════════════════════════════════════════════════════════

    def weight_clipping(self, layer_idx=None, clip_val: float = 3.0) -> list:
        """
        Clips all weight values to [-clip_val, +clip_val].
        Effective for weight_explosion and sign_flip errors.
        """
        results = []
        for layer in self._resolve_layers(layer_idx):
            before = float(np.max(np.abs(layer.W)))
            np.clip(layer.W, -clip_val, clip_val, out=layer.W)
            after  = float(np.max(np.abs(layer.W)))
            layer.healthy    = True
            layer.error_type = None
            r = self._log(
                f"Clipped {layer.name}: max |W| {before:.3f} → {after:.3f}",
                "Weight Clipping", layer.name, before, after
            )
            results.append(r)
        return results

    # ══════════════════════════════════════════════════════════════════════════
    #  STRATEGY 2 — Xavier Reinitialisation
    # ══════════════════════════════════════════════════════════════════════════

    def xavier_reinit(self, layer_idx=None, threshold: float = 1e-4) -> list:
        """
        Reinitialises dead or near-zero weights using Xavier uniform init.
        Effective for dead_neurons and gradient_vanish errors.
        """
        results = []
        for layer in self._resolve_layers(layer_idx):
            n_in, n_out = layer.W.shape
            dead_mask   = np.abs(layer.W) < threshold
            before      = float(dead_mask.mean() * 100)
            limit       = np.sqrt(6.0 / (n_in + n_out))
            layer.W[dead_mask] = np.random.uniform(-limit, limit, int(dead_mask.sum()))
            after  = float((np.abs(layer.W) < threshold).mean() * 100)
            layer.healthy    = True
            layer.error_type = None
            r = self._log(
                f"Xavier reinit {layer.name}: dead {before:.1f}% → {after:.1f}%",
                "Xavier Reinit", layer.name, before, after
            )
            results.append(r)
        return results

    # ══════════════════════════════════════════════════════════════════════════
    #  STRATEGY 3 — Gradient Norm Repair
    # ══════════════════════════════════════════════════════════════════════════

    def gradient_norm_repair(self, layer_idx=None) -> list:
        """
        Rescales the weight matrix so its Frobenius norm equals √n_in.
        Restores proper gradient magnitude flow. Effective for gradient_vanish.
        """
        results = []
        for layer in self._resolve_layers(layer_idx):
            before = float(np.mean(np.abs(layer.W)))
            norm   = np.linalg.norm(layer.W) + 1e-8
            target = np.sqrt(float(layer.W.shape[0]))
            layer.W = layer.W / norm * target
            after  = float(np.mean(np.abs(layer.W)))
            layer.healthy    = True
            layer.error_type = None
            r = self._log(
                f"Grad-norm {layer.name}: mean {before:.5f} → {after:.5f}",
                "Gradient Norm", layer.name, before, after
            )
            results.append(r)
        return results

    # ══════════════════════════════════════════════════════════════════════════
    #  STRATEGY 4 — Noise Denoising (Soft Thresholding)
    # ══════════════════════════════════════════════════════════════════════════

    def noise_denoising(self, layer_idx=None, percentile: float = 75.0) -> list:
        """
        Applies soft-thresholding at the given percentile of |W|.
        Effective for noisy_weights errors.
        """
        results = []
        for layer in self._resolve_layers(layer_idx):
            before    = float(np.std(layer.W))
            threshold = float(np.percentile(np.abs(layer.W), percentile))
            layer.W   = np.sign(layer.W) * np.maximum(np.abs(layer.W) - threshold * 0.5, 0)
            after     = float(np.std(layer.W))
            layer.healthy    = True
            layer.error_type = None
            r = self._log(
                f"Denoised {layer.name}: std {before:.4f} → {after:.4f}",
                "Noise Denoising", layer.name, before, after
            )
            results.append(r)
        return results

    # ══════════════════════════════════════════════════════════════════════════
    #  STRATEGY 5 — Bias Correction
    # ══════════════════════════════════════════════════════════════════════════

    def bias_correction(self, layer_idx=None) -> list:
        """
        Zero-centres and unit-normalises bias vectors.
        Effective for bias_shift errors.
        """
        results = []
        for layer in self._resolve_layers(layer_idx):
            before   = float(np.max(np.abs(layer.b)))
            layer.b  = layer.b - np.mean(layer.b)
            std_b    = np.std(layer.b)
            if std_b > 1.0:
                layer.b /= std_b
            after    = float(np.max(np.abs(layer.b)))
            layer.healthy    = True
            layer.error_type = None
            r = self._log(
                f"Bias-fix {layer.name}: max |b| {before:.4f} → {after:.4f}",
                "Bias Correction", layer.name, before, after
            )
            results.append(r)
        return results

    # ══════════════════════════════════════════════════════════════════════════
    #  AUTO REPAIR — detect & dispatch
    # ══════════════════════════════════════════════════════════════════════════

    def auto_repair(self) -> list:
        """
        Inspects every layer, chooses the best strategy per detected issue,
        then finishes with bias_correction on all layers.
        """
        issues  = self.dnn.health_check()
        results = []
        handled = set()

        DISPATCH = {
            "weight_explosion": self.weight_clipping,
            "dead_neurons"    : self.xavier_reinit,
            "gradient_vanish" : self.gradient_norm_repair,
            "noisy_weights"   : self.noise_denoising,
        }

        for layer_name, issue in issues:
            idx = next(
                i for i, l in enumerate(self.dnn.layers) if l.name == layer_name
            )
            if idx not in handled and issue in DISPATCH:
                results += DISPATCH[issue](idx)
                handled.add(idx)

        # Always finish with bias correction on all layers
        results += self.bias_correction()
        self.dnn.repaired = True
        return results

    # ── Dispatcher (called from routes) ────────────────────────────────────────

    def apply(self, strategy: str, layer_idx=None) -> list:
        mapping = {
            "auto"    : lambda: self.auto_repair(),
            "clip"    : lambda: self.weight_clipping(layer_idx),
            "xavier"  : lambda: self.xavier_reinit(layer_idx),
            "gradient": lambda: self.gradient_norm_repair(layer_idx),
            "denoise" : lambda: self.noise_denoising(layer_idx),
            "bias"    : lambda: self.bias_correction(layer_idx),
        }
        fn = mapping.get(strategy, lambda: self.auto_repair())
        return fn()
