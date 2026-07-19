# Hybrid-Sentinel IEEE Overleaf Package (Dual-Track: R18 + NDN PoC)

## Upload to Overleaf
1. Zip this folder (or use `Hybrid_Sentinel_IEEE_Overleaf_v2.zip` at repo root).
2. New Project → Upload Project.
3. Set `main.tex` as the main document.
4. Compile with pdfLaTeX.

## What changed vs previous Overleaf zip
- Full story rewrite with honest **dual-track** architecture (IP cascade + NDN PoC).
- New Fig. 1: `fig_unified_system_architecture.pdf`.
- NDN sections + figures (Interest Flooding, Cache Pollution).
- Snort comparison woven into abstract/intro/discussion.
- Metrics aligned to `docs/r18/metrics/submission_metrics_r18.json` (E2E FPR **1.11%**) and `docs/ndn_poc/ndn_metrics.json`.

## Regenerate figures (from repo root)
```bash
python scripts/analysis/plot_unified_ndn_architecture.py
python scripts/analysis/generate_ieee_figures_ndn.py
python scripts/analysis/generate_submission_metrics_r18.py
```
