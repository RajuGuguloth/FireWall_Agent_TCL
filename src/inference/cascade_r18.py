"""
R18 production cascade — single implementation shared by API and offline eval.

Import paths, thresholds, and inference logic from here (not duplicated in api/).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import onnxruntime as ort

import config as cfg


def gate_summary(seq: np.ndarray) -> np.ndarray:
    """seq: (N, window, features) or (1, window, features)."""
    return np.concatenate([seq.mean(1), seq.std(1), seq[:, -1, :]], axis=1)


def softmax_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / temperature
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def maha_score(embed: np.ndarray, t3: Dict[str, Any]) -> float:
    d = embed - t3["mu"]
    return float(np.einsum("j,jk,k->", d, t3["inv_cov"], d))


def load_gate():
    gate = joblib.load(cfg.TIER1_GATE)
    if hasattr(gate, "n_jobs"):
        gate.n_jobs = 1
    return gate


@dataclass
class CascadeRuntime:
    scaler: Any
    encoder: Any
    classes: List[str]
    benign_idx: int
    gate: Any
    gate_benign_idx: int
    ort_cls: ort.InferenceSession
    cls_in: str
    temperature: float
    embed_sess: Optional[ort.InferenceSession]
    embed_in: Optional[str]
    t3: Optional[Dict[str, Any]]

    @classmethod
    def load(cls) -> "CascadeRuntime":
        encoder = joblib.load(cfg.ENCODER_PATH)
        classes = list(encoder.classes_)
        ort_cls = ort.InferenceSession(cfg.TIER2_ONNX)
        embed_path = cfg.TIER2_EMBED
        embed_sess = ort.InferenceSession(embed_path) if os.path.isfile(embed_path) else None
        t3 = joblib.load(cfg.TIER3_ONECLASS) if os.path.isfile(cfg.TIER3_ONECLASS) else None
        if t3 and not embed_sess:
            raise FileNotFoundError(f"Tier-3 params exist but embedding ONNX missing: {embed_path}")
        with open(cfg.TIER2_TEMP) as f:
            temperature = float(json.load(f)["temperature"])
        gate = load_gate()
        return cls(
            scaler=joblib.load(cfg.SCALER_PATH),
            encoder=encoder,
            classes=classes,
            benign_idx=classes.index("BENIGN"),
            gate=gate,
            gate_benign_idx=list(gate.classes_).index(1),
            ort_cls=ort_cls,
            cls_in=ort_cls.get_inputs()[0].name,
            temperature=temperature,
            embed_sess=embed_sess,
            embed_in=embed_sess.get_inputs()[0].name if embed_sess else None,
            t3=t3,
        )

    def scale_raw(self, raw_seq: List[List[float]]) -> np.ndarray:
        if len(raw_seq) != cfg.WINDOW_SIZE:
            raise ValueError(f"Sequence length must be {cfg.WINDOW_SIZE}, got {len(raw_seq)}")
        if any(len(p) != cfg.N_FEATURES for p in raw_seq):
            raise ValueError(f"Each packet must have {cfg.N_FEATURES} features")
        arr = np.array(raw_seq, dtype=np.float32).reshape(1, cfg.WINDOW_SIZE, cfg.N_FEATURES)
        flat = self.scaler.transform(arr.reshape(-1, cfg.N_FEATURES))
        flat = np.clip(flat, -cfg.CLIP_VAL, cfg.CLIP_VAL)
        return flat.reshape(1, cfg.WINDOW_SIZE, cfg.N_FEATURES)

    def _tier3_check(self, seq: np.ndarray, trace: List[Dict[str, Any]], tiers: List[str]) -> Optional[Dict[str, Any]]:
        """Run Tier-3 on an ALLOW candidate. Returns override dict or None to pass."""
        if self.t3 is None or self.embed_sess is None:
            return None
        embed = self.embed_sess.run(None, {self.embed_in: seq})[0][0]
        tiers.append("tier3_oneclass")
        score = maha_score(embed, self.t3)
        t3_flag = score > self.t3["threshold"]
        trace.append({
            "tier": "tier3_oneclass",
            "role": "Zero-day Mahalanobis on r18 embeddings",
            "maha_score": round(score, 4),
            "threshold": round(float(self.t3["threshold"]), 4),
            "decision": "FLAG_ANOMALY" if t3_flag else "PASS",
            "explanation": (
                f"novelty score {score:.2f} > {self.t3['threshold']:.2f} → FLAG zero-day"
                if t3_flag
                else f"novelty score {score:.2f} ≤ threshold → pass"
            ),
        })
        if t3_flag:
            return {"action": "FLAG", "label": "ANOMALY", "class_id": None}
        return None

    def classify_scaled(self, seq: np.ndarray) -> Dict[str, Any]:
        """Classify one scaled window (1, window, features)."""
        tiers: List[str] = []
        trace: List[Dict[str, Any]] = []

        p_benign = float(
            self.gate.predict_proba(gate_summary(seq))[0, self.gate_benign_idx]
        )
        trace.append({
            "tier": "tier1_gate",
            "role": "Fast BENIGN vs ATTACK gate",
            "p_benign": round(p_benign, 4),
            "threshold": cfg.GATE_THRESHOLD,
            "decision": "ALLOW_FAST_PATH" if p_benign >= cfg.GATE_THRESHOLD else "ESCALATE",
            "explanation": (
                f"P(BENIGN)={p_benign:.2%} ≥ {cfg.GATE_THRESHOLD} → skip deep model"
                if p_benign >= cfg.GATE_THRESHOLD
                else f"P(BENIGN)={p_benign:.2%} < {cfg.GATE_THRESHOLD} → send to Tier-2"
            ),
        })

        if p_benign >= cfg.GATE_THRESHOLD:
            tiers.append("tier1_gate")
            t3 = self._tier3_check(seq, trace, tiers)
            if t3:
                return {
                    **t3,
                    "confidence": p_benign,
                    "tiers_used": tiers,
                    "tier_trace": trace,
                    "probabilities": {"BENIGN": p_benign},
                }
            return {
                "label": "BENIGN",
                "class_id": self.benign_idx,
                "confidence": p_benign,
                "tiers_used": tiers,
                "tier_trace": trace,
                "probabilities": {"BENIGN": p_benign},
                "action": "ALLOW",
            }

        tiers.extend(["tier1_gate_escalate", "tier2_cnn_gru"])
        logits = self.ort_cls.run(None, {self.cls_in: seq})[0][0]
        probs = softmax_logits(logits, self.temperature)
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        label = self.classes[class_id]
        prob_map = {self.classes[i]: round(float(probs[i]), 4) for i in range(len(self.classes))}

        if label == "BENIGN":
            action = "ALLOW"
            t2_reason = "Tier-2 predicted BENIGN → ALLOW"
        elif confidence > cfg.BLOCK_THRESHOLD:
            action = "BLOCK"
            t2_reason = f"attack conf {confidence:.2%} > block {cfg.BLOCK_THRESHOLD} → BLOCK"
        elif confidence >= cfg.FLAG_THRESHOLD:
            action = "FLAG"
            t2_reason = f"attack conf {confidence:.2%} ≥ flag {cfg.FLAG_THRESHOLD} → FLAG"
        else:
            action = "ALLOW"
            t2_reason = f"low attack conf {confidence:.2%} → ALLOW (leak risk)"

        trace.append({
            "tier": "tier2_cnn_gru",
            "role": "6-class CNN-GRU attack classifier",
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": prob_map,
            "block_threshold": cfg.BLOCK_THRESHOLD,
            "flag_threshold": cfg.FLAG_THRESHOLD,
            "decision": action,
            "explanation": t2_reason,
        })

        if action == "ALLOW":
            t3 = self._tier3_check(seq, trace, tiers)
            if t3:
                return {
                    **t3,
                    "confidence": confidence,
                    "tiers_used": tiers,
                    "tier_trace": trace,
                    "probabilities": prob_map,
                }

        return {
            "label": label,
            "class_id": class_id,
            "confidence": confidence,
            "tiers_used": tiers,
            "tier_trace": trace,
            "probabilities": prob_map,
            "action": action,
        }

    def classify_raw(self, raw_seq: List[List[float]]) -> Dict[str, Any]:
        return self.classify_scaled(self.scale_raw(raw_seq))
