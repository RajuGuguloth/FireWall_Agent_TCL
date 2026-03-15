import numpy as np

def generate_sliding_windows(
    packet_features: np.ndarray,
    window_size: int = 64,
    stride: int = 32
):
    """
    packet_features: (N, F)
    returns: (num_windows, window_size, F)
    """

    if packet_features.ndim != 2:
        raise ValueError("packet_features must be 2D (N, F)")

    N, F = packet_features.shape
    windows = []

    if N < window_size:
        return np.empty((0, window_size, F), dtype=np.float32)

    for start in range(0, N - window_size + 1, stride):
        window = packet_features[start:start + window_size]
        windows.append(window)

    return np.stack(windows).astype(np.float32)

