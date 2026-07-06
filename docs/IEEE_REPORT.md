# IEEE Format Report — Hybrid-Sentinel R18

This guide explains how to produce an **IEEE conference-style paper** with figures and proof-of-work for your project.

---

## Quick start

```bash
cd firewall_ml_project
source .venv/bin/activate

# 1. Generate all IEEE figures (PDF + PNG)
python scripts/analysis/generate_ieee_figures_r18.py

# 2. Standard per-tier submission metrics & plots (Table I, ROC, latency)
python scripts/analysis/generate_submission_metrics_r18.py

# 2. Optional: legacy thesis figures (architecture, docker topology)
python scripts/analysis/plot_architecture_diagram.py
python scripts/analysis/plot_docker_topology.py
python scripts/analysis/plot_data_split_diagram.py

# 3. Compile LaTeX (requires pdflatex)
cd docs && pdflatex IEEE_REPORT.tex && pdflatex IEEE_REPORT.tex
```

Output PDF: `docs/IEEE_REPORT.pdf`

---

## Submission metrics (standard per-tier)

```bash
python scripts/analysis/generate_submission_metrics_r18.py
```

**Outputs:** `results/submission_figures/` + `results/submission_metrics_r18.json`

| Figure | Metrics shown |
|--------|----------------|
| `fig_submission_metrics_summary_table` | **Table I** — Acc/Prec/Rec/F1 per tier + E2E |
| `fig_submission_tier_metric_comparison` | Bar chart: all tiers compared |
| `fig_submission_tier2_per_class_heatmap` | Tier-2 precision/recall/F1 per class |
| `fig_submission_tier1_confusion_matrix` | Tier-1 binary confusion |
| `fig_submission_tier2_confusion_matrix` | Tier-2 normalized confusion |
| `fig_submission_tier3_roc_curve` | Tier-3 ROC + AUC |
| `fig_submission_latency_throughput` | Latency (ms) + throughput (seq/s) |
| `fig_submission_e2e_security_rates` | Attack detection, benign FPR, E2E precision/F1 |

LaTeX table: `results/submission_metrics_table.tex` (paste into IEEE paper)


All generated to `results/ieee_figures/`:

| File | IEEE label | Content |
|------|------------|---------|
| `fig_r18_orchestration_pipeline.pdf` | Fig. 2 | **Build & deploy orchestration** — PCAP → train → ONNX → API |
| `fig_r18_architecture_flow.pdf` | Fig. 1 | **Runtime 3-tier cascade** — Tier-1/2/3 decision flow |
| `fig_r18_confusion_matrix.pdf` | Fig. 3 | Tier-2 normalized confusion matrix (live test inference) |
| `fig_r18_per_class_metrics.pdf` | Fig. 4 | Precision / Recall / F1 per attack class |
| `fig_r18_cascade_funnel.pdf` | Fig. 5 | Traffic funnel (% at each cascade stage) |
| `fig_r18_latency_benchmark.pdf` | Fig. 6 | Latency avg & p99 vs G-Scaler baseline |
| `fig_r18_class_distribution.pdf` | — | Test set class balance |
| `fig_r17_vs_r18_comparison.pdf` | Fig. 7 | Why R18 replaced R17 (benign FPR fix) |

**Legacy figures** in `results/proof_of_work_visuals/`:

| File | Use for |
|------|---------|
| `fig_architecture_diagram.pdf` | Detailed vertical architecture (update labels to R18 if needed) |
| `fig_latency_benchmark.png` | Older R16 latency (use R18 version instead) |
| `fig_ce_vs_focal_confusion.png` | Focal loss ablation (R13 vs R16) |
| `fig_validation_comparison.png` | GroupShuffleSplit vs random split |

---

## IEEE paper structure (`docs/IEEE_REPORT.tex`)

| Section | Content |
|---------|---------|
| Abstract | Problem, 3-tier approach, headline metrics |
| I. Introduction | Snort vs AI vs NDN hybrid motivation |
| II. Related Work | Signature IDS, deep learning IDS |
| III. System Architecture | Figs 1–2 |
| IV. Methodology | Features, splits, tiers, classes |
| V. Experimental Setup | Docker lab, test set size |
| VI. Results | Tables + Figs 3–7 |
| VII. Discussion | Limitations, future work |
| VIII. Conclusion | R18 contribution |

**Before submitting:** Replace `[your.email@domain.com]` in the `.tex` file.

---

## Proof of work checklist

| Evidence | Location |
|----------|----------|
| Metrics JSON | `results/r18_tier_metrics.json` |
| Latency JSON | `results/r18_latency_benchmark.json` |
| Cascade eval log | Run `python scripts/eval/measure_cascade_flow.py` |
| IEEE figures | `results/ieee_figures/` |
| Design notes | `docs/DESIGN_AND_IMPLEMENTATION.md` |
| Full report | `docs/FINAL_SUBMISSION_REPORT.md` |

---

## Architecture orchestration — what to explain

### Build-time orchestration (Fig. 2)

```
PCAP capture → extract_v5_features.py (17 features)
            → prepare_v6_sequences.py (group split, scaler)
            → train_tier1_gate_v6.py
            → train_cnn_gru_v6.py
            → export ONNX + tier3_oneclass
            → cascade_r18.py + api/main.py
```

### Runtime orchestration (Fig. 1)

```
20×17 window → Tier-1 gate → [ALLOW | escalate]
             → Tier-2 ONNX → [BLOCK | FLAG | ALLOW]
             → Tier-3 Mahalanobis → [FLAG anomaly | pass]
             → ALLOW/FLAG/BLOCK + tier_trace
```

---

## Compiling without LaTeX

Use the markdown report as source and paste figures manually:

1. `docs/FINAL_SUBMISSION_REPORT.md` — full text
2. Insert figures from `results/ieee_figures/*.png` into Word/Google Docs
3. Apply IEEE two-column template from your institution

---

## Regenerating after retrain

```bash
python scripts/eval/measure_cascade_flow.py
python -c "from api.tier_metrics import compute_tier_metrics; compute_tier_metrics(force=True)"
python scripts/benchmark/benchmark_latency.py
python scripts/analysis/generate_ieee_figures_r18.py
```

---

*Hybrid-Sentinel R18 — IEEE report package*
