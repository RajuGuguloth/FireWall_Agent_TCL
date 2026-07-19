# IEEE Format Report — Hybrid-Sentinel (R18 + NDN PoC)

Dual-track paper: **Track A** = production IP cascade (R18); **Track B** = NDN forwarder PoC (Interest Flooding / Cache Pollution).

## Quick start (Overleaf)

Upload **`Hybrid_Sentinel_IEEE_Overleaf_v2.zip`** (repo root) → set `main.tex` as main document → pdfLaTeX.

Source copy also lives at `docs/ieee_overleaf_package/` and `docs/IEEE_REPORT.tex`.

## Regenerate figures

```bash
source .venv/bin/activate

# Unified dual-track architecture (NEW — Fig. 1 in rewritten paper)
python scripts/analysis/plot_unified_ndn_architecture.py

# NDN PoC IEEE figures (fixed topology labels)
python scripts/analysis/generate_ieee_figures_ndn.py

# R18 submission metrics / plots
python scripts/analysis/generate_submission_metrics_r18.py

# Optional cascade / orchestration diagrams
python scripts/analysis/generate_ieee_figures_r18.py
```

## Figure map (rewritten paper)

| Figure | File | Notes |
|--------|------|--------|
| Unified architecture | `fig_unified_system_architecture.pdf` | Dual track — keep |
| Cascade flow | `architecture_flow.pdf` | Track A detail — keep |
| Orchestration | `orchestration_pipeline.pdf` | Track A build path — keep |
| Tier metrics / CM / heatmap / E2E / latency | `docs/r18/metrics/fig_submission_*` | Accurate vs JSON — keep |
| NDN topology | `fig_ndn_architecture.pdf` | **Fixed** clipped attacker label |
| NDN results | `fig_ndn_*.pdf` | Match `ndn_metrics.json` — keep |

## Important honesty in the story

- Do **not** claim R18 was trained on NDN packets.
- NDN PoC is a **simulation** (not live NFD).
- Use E2E benign FPR **1.11%** from `submission_metrics_r18.json` (not older 0.86% notes unless you re-run that eval).

## Local compile

```bash
cd docs/ieee_overleaf_package && pdflatex main.tex && pdflatex main.tex
```
