from scapy.all import rdpcap, wrpcap
import glob, random

random.seed(42)

# ── Load all benign ───────────────────────────────────────
print("Loading benign packets...")
benign = []
for f in sorted(glob.glob("data/raw/benign/*.pcap")):
    pkts = rdpcap(f)
    benign.extend(pkts)
    print(f"  {f} → {len(pkts):,}")

print(f"Total benign: {len(benign):,}")

# ── Load all attacks ──────────────────────────────────────
print("\nLoading attack packets...")
attacks = []
for f in sorted(glob.glob("data/raw/attacks/*.pcap")):
    pkts = rdpcap(f)
    attacks.extend(pkts)
    print(f"  {f} → {len(pkts):,}")

print(f"Total attacks: {len(attacks):,}")

# ── Sample attacks to match 2.4% ratio ───────────────────
n_benign  = len(benign)
n_attacks = int(n_benign * (2.4 / 97.6))
sampled_attacks = random.sample(attacks, min(n_attacks, len(attacks)))

print(f"\nSampled {len(sampled_attacks):,} attacks for 2.4% ratio")

# ── Create mixed_production_v2.pcap ──────────────────────
mixed = list(benign) + sampled_attacks
random.shuffle(mixed)
wrpcap("data/raw/mixed_production_v2.pcap", mixed)
print(f"✅ mixed_production_v2.pcap → {len(mixed):,} packets")
print(f"   Benign  : {n_benign:,} ({n_benign/len(mixed)*100:.1f}%)")
print(f"   Attacks : {len(sampled_attacks):,} ({len(sampled_attacks)/len(mixed)*100:.1f}%)")

# ── Create clean_attacks.pcap (all attacks, no benign) ───
wrpcap("data/raw/clean_attacks.pcap", attacks)
print(f"\n✅ clean_attacks.pcap → {len(attacks):,} packets (pure attacks only)")

