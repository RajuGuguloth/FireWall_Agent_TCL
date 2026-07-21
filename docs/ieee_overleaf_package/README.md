# Hybrid-Sentinel IEEE Overleaf Package (section-wise figures)

## Upload
1. Upload this folder (or `HybridSentinel_final_fixed.zip` at repo root) to Overleaf.
2. Set **main.tex** as the main document.
3. Compile with **pdfLaTeX** (may need 2 passes for refs).

## What was fixed vs HybridSentinel_final (2).zip
- Wide architecture uses `figure*` + `\textwidth` (was incorrectly squeezed into one column).
- Every figure has a **body paragraph** explaining what the graph conveys (not caption-only).
- Figures are placed **section-wise** with `\FloatBarrier`:
  - §System Architecture → unified, cascade flow, orchestration, NDN topology
  - §Experimental Setup → Docker topology, class distribution
  - §Results Track A → tier bars, Tier-1 CM, Tier-2 CM, heatmap, funnel, E2E, Tier-3 ROC, latency
  - §Results Track B → NDN CM, per-class metrics, per-class accuracy, summary, ROC
- NDN results are fully explained (Interest Flooding perfect; benign↔pollution 1% swap).

## Regenerate plots (optional)
```bash
python scripts/analysis/regenerate_clear_ieee_cms.py
python scripts/analysis/plot_unified_ndn_architecture.py
python scripts/analysis/generate_ieee_figures_ndn.py
```
