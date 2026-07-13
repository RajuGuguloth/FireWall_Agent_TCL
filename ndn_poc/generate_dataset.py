"""
Generate the NDN-native behavioural dataset.

Runs many randomized simulation episodes across three classes
(BENIGN, INTEREST_FLOODING, CACHE_POLLUTION), extracts causal 20-packet windows,
and saves them for training.

Usage:
    python -m ndn_poc.generate_dataset --episodes 200 --out data/ndn
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import List

import numpy as np

from .features import make_windows, observations_to_matrix, FEATURE_NAMES
from .simulator import (
    BENIGN,
    CACHE_POLLUTION,
    EpisodeConfig,
    INTEREST_FLOODING,
    run_episode,
)

CLASSES = [BENIGN, INTEREST_FLOODING, CACHE_POLLUTION]


def _random_cfg(rng: np.random.Generator, attack_type: str, seed: int) -> EpisodeConfig:
    return EpisodeConfig(
        duration=float(rng.uniform(5.0, 7.0)),
        seed=seed,
        catalog_size=int(rng.integers(300, 800)),
        zipf_s=float(rng.uniform(0.9, 1.3)),
        cs_capacity=int(rng.integers(50, 150)),
        pit_lifetime=float(rng.uniform(1.5, 2.5)),
        n_benign=int(rng.integers(3, 7)),
        benign_rate_lo=20.0,
        benign_rate_hi=40.0,
        attack_type=attack_type,
        attack_start=float(rng.uniform(0.8, 1.5)),
        n_attackers=int(rng.integers(1, 3)),
        attack_rate_lo=150.0,
        attack_rate_hi=350.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200, help="episodes PER class")
    ap.add_argument("--out", type=str, default="data/ndn")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meta_rng = np.random.default_rng(args.seed)

    all_windows: List[np.ndarray] = []
    all_labels: List[str] = []
    episode_id = 0

    for attack_type in CLASSES:
        for _ in range(args.episodes):
            cfg = _random_cfg(meta_rng, attack_type, seed=int(meta_rng.integers(0, 2**31)))
            log = run_episode(cfg)
            if len(log) < 20:
                continue
            X_pkt, atk = observations_to_matrix(log)
            W, labels = make_windows(X_pkt, atk, attack_type)
            if W.shape[0]:
                all_windows.append(W)
                all_labels.extend(labels)
            episode_id += 1
        print(f"  {attack_type:<18}: episodes done")

    X = np.concatenate(all_windows, axis=0)
    y = np.array(all_labels)

    np.save(os.path.join(args.out, "ndn_windows.npy"), X)
    np.save(os.path.join(args.out, "ndn_labels.npy"), y)

    dist = Counter(y.tolist())
    meta = {
        "n_windows": int(X.shape[0]),
        "window_size": int(X.shape[1]),
        "n_features": int(X.shape[2]),
        "feature_names": FEATURE_NAMES,
        "class_distribution": dist,
        "episodes_per_class": args.episodes,
        "seed": args.seed,
    }
    with open(os.path.join(args.out, "ndn_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 60)
    print("  NDN dataset generated")
    print("=" * 60)
    print(f"  windows      : {X.shape}")
    print(f"  features     : {X.shape[2]} NDN-native ({', '.join(FEATURE_NAMES[:4])} ...)")
    print(f"  distribution : {dict(dist)}")
    print(f"  saved to     : {args.out}/")


if __name__ == "__main__":
    main()
