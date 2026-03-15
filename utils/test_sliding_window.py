import numpy as np
from sliding_window import generate_sliding_windows

# Fake packet feature array: 100 packets, 13 features
fake_packets = np.random.rand(100, 13).astype(np.float32)

windows = generate_sliding_windows(
    packet_features=fake_packets,
    window_size=32,
    stride=16
)

print("Input shape:", fake_packets.shape)
print("Output shape:", windows.shape)

assert windows.ndim == 3
assert windows.shape[1] == 32
assert windows.shape[2] == 13

print("Sliding window test PASSED")

