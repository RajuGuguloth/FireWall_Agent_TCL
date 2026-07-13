"""
NDN-native feature extraction.

Converts a simulator observation log (one row per Interest/Data packet crossing
the monitored router) into per-packet feature vectors and then into 20-packet
behavioural windows, mirroring the production IP pipeline's window contract
(config.WINDOW_SIZE=20, config.STRIDE=10) so the *same* detection approach
applies. Every feature is computed CAUSALLY (past packets only), so the extractor
is deployable on a live forwarder, not just offline.

The 17 NDN-native features deliberately parallel the 17 IP features of the
production model, but read NDN forwarder state (PIT, Content Store, content
names) instead of TCP/IP headers:

    IP feature (production)     ->  NDN-native analogue (this PoC)
    ---------------------------     ----------------------------------------
    has_tcp/has_udp/has_icmp    ->  is_interest / is_data
    dst_port / tcp_flags        ->  name_depth / name_entropy
    payload_entropy             ->  name_entropy, name_diversity_win
    is_syn/rst flood signals    ->  pit_size, pit_growth, unsatisfied_ratio_win
    flow_total_bytes            ->  interest_rate_win
    (n/a in IP)                 ->  cs_hit, cs_hit_ratio_win, cs_size  (Content Store)
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .simulator import Observation, INTEREST, DATA, BENIGN

WINDOW_SIZE = 20
STRIDE = 10
LOOKBACK = 50  # trailing packets used for windowed running features

FEATURE_NAMES = [
    "is_interest",          # 1 for Interest packets
    "is_data",              # 1 for Data packets
    "name_depth",           # number of name components
    "name_entropy",         # Shannon entropy of the content name chars
    "is_new_name",          # name unseen within LOOKBACK -> flooding/pollution
    "is_unsatisfiable",     # producer has no such content -> Interest Flooding
    "cs_hit",               # Interest satisfied from Content Store
    "pit_aggregated",       # Interest collapsed onto existing PIT entry
    "pit_size",             # Pending Interest Table occupancy
    "pit_growth",           # change in PIT size vs previous packet
    "cs_size",              # Content Store occupancy
    "expired_since_last",   # PIT timeouts since previous packet
    "interarrival",         # seconds since previous packet
    "interest_rate_win",    # Interests/sec over trailing LOOKBACK
    "cs_hit_ratio_win",     # Content Store hit ratio over trailing LOOKBACK
    "unsatisfied_ratio_win",# timeout/unsatisfied ratio over trailing LOOKBACK
    "name_diversity_win",   # unique names / packets over trailing LOOKBACK
]
N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 17, N_FEATURES


def _name_entropy(name: str) -> float:
    if not name:
        return 0.0
    counts = {}
    for ch in name:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(name)
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values()))


def observations_to_matrix(log: List[Observation]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (X_packets, is_attack) where
        X_packets: (num_packets, 17) causal feature matrix
        is_attack: (num_packets,) 1 if the packet came from an attacker
    """
    n = len(log)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)
    atk = np.zeros(n, dtype=np.int64)

    prev_t = log[0].t if log else 0.0
    prev_pit = 0
    seen_names: "dict[str, int]" = {}  # name -> last index seen

    for i, o in enumerate(log):
        is_interest = int(o.kind == INTEREST)
        is_data = int(o.kind == DATA)
        depth = o.name.count("/")
        ent = _name_entropy(o.name)
        is_new = int(o.name not in seen_names or (i - seen_names[o.name]) > LOOKBACK)
        interarrival = max(0.0, o.t - prev_t)
        pit_growth = o.pit_size - prev_pit

        # trailing-window running stats (causal: indices [lo, i])
        lo = max(0, i - LOOKBACK + 1)
        win = log[lo:i + 1]
        span = max(1e-6, win[-1].t - win[0].t)
        n_interest = sum(1 for w in win if w.kind == INTEREST)
        interest_rate = n_interest / span
        cs_lookups = sum(1 for w in win if w.kind == INTEREST)
        cs_hits = sum(w.cs_hit for w in win)
        cs_hit_ratio = cs_hits / cs_lookups if cs_lookups else 0.0
        expired = sum(w.expired_since_last for w in win)
        unsatisfied_ratio = expired / cs_lookups if cs_lookups else 0.0
        uniq = len({w.name for w in win})
        name_div = uniq / len(win)

        X[i] = (
            is_interest, is_data, depth, ent, is_new,
            int(o.satisfiable == 0), o.cs_hit, o.pit_aggregated,
            o.pit_size, pit_growth, o.cs_size, o.expired_since_last,
            interarrival, interest_rate, cs_hit_ratio, unsatisfied_ratio, name_div,
        )
        atk[i] = o.is_attack

        seen_names[o.name] = i
        prev_t = o.t
        prev_pit = o.pit_size

    return X, atk


def make_windows(
    X: np.ndarray,
    atk: np.ndarray,
    attack_type: str,
    window: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> Tuple[np.ndarray, List[str]]:
    """
    Slice per-packet features into (num_windows, window, 17) tensors.

    A window is labelled with `attack_type` if it contains >=1 attacker packet,
    otherwise BENIGN. This means warmup windows of an attack episode are
    correctly labelled BENIGN, preventing trivial episode-level separation.
    """
    N = X.shape[0]
    windows: List[np.ndarray] = []
    labels: List[str] = []
    if N < window:
        return np.empty((0, window, X.shape[1]), dtype=np.float32), labels
    for start in range(0, N - window + 1, stride):
        seg = X[start:start + window]
        seg_atk = atk[start:start + window]
        label = attack_type if seg_atk.sum() > 0 and attack_type != BENIGN else BENIGN
        windows.append(seg)
        labels.append(label)
    return np.stack(windows).astype(np.float32), labels
