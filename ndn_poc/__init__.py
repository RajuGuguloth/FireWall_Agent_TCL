"""
Hybrid-Sentinel — NDN proof-of-concept module.

This package provides a controlled, discrete-event Named Data Networking (NDN)
simulation used to validate the paper's NDN-native threat claims (Interest
Flooding against the PIT, Cache Pollution against the Content Store).

It is intentionally SEPARATE from the production IP-attack pipeline (src/, api/).
The production cascade is trained and evaluated on real TCP/IP captures; this
module supplies an NDN-native behavioural dataset so the same detection approach
(behavioural windows -> classifier) can be shown to transfer to NDN forwarder
state. See ndn_poc/README.md for scope and honest limitations.
"""

__all__ = ["simulator", "features"]
