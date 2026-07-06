#!/usr/bin/env bash
# Sync figures into ieee_overleaf/figures/ for Overleaf upload.
# Run from repo root: bash scripts/analysis/sync_ieee_overleaf_folder.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$ROOT/ieee_overleaf/figures"
mkdir -p "$DEST"

copy() {
  local src="$1" dst="$2"
  if [[ -f "$src" ]]; then
    cp "$src" "$DEST/$dst"
    echo "  $dst"
  else
    echo "  SKIP (missing): $src"
  fi
}

echo "Syncing figures -> $DEST"
copy "$ROOT/results/ieee_figures/fig_r18_architecture_flow.pdf"           architecture_flow.pdf
copy "$ROOT/results/ieee_figures/fig_r18_orchestration_pipeline.pdf"      orchestration_pipeline.pdf
copy "$ROOT/results/proof_of_work_visuals/fig_architecture_diagram.pdf"   architecture_detailed.pdf
copy "$ROOT/results/ieee_figures/fig_r18_cascade_funnel.pdf"              cascade_funnel.pdf
copy "$ROOT/results/submission_figures/fig_submission_tier2_confusion_matrix.pdf" confusion_matrix.pdf
copy "$ROOT/results/submission_figures/fig_submission_tier2_per_class_heatmap.pdf" per_class_heatmap.pdf
copy "$ROOT/results/submission_figures/fig_submission_tier_metric_comparison.pdf" tier_metrics_bars.pdf
copy "$ROOT/results/submission_figures/fig_submission_metrics_summary_table.pdf" metrics_table.pdf
copy "$ROOT/results/submission_figures/fig_submission_tier1_confusion_matrix.pdf" tier1_confusion.pdf
copy "$ROOT/results/submission_figures/fig_submission_tier3_roc_curve.pdf" tier3_roc.pdf
copy "$ROOT/results/submission_figures/fig_submission_latency_throughput.pdf" latency_throughput.pdf
copy "$ROOT/results/submission_figures/fig_submission_e2e_security_rates.pdf" e2e_security.pdf
copy "$ROOT/results/ieee_figures/fig_r18_class_distribution.pdf"          class_distribution.pdf

# Optional: generate docker topology if script exists
if [[ -f "$ROOT/results/proof_of_work_visuals/fig_docker_topology.pdf" ]]; then
  copy "$ROOT/results/proof_of_work_visuals/fig_docker_topology.pdf" docker_topology.pdf
else
  (cd "$ROOT" && .venv/bin/python scripts/analysis/plot_docker_topology.py 2>/dev/null) || true
  if [[ -f "$ROOT/results/proof_of_work_visuals/fig_docker_topology.pdf" ]]; then
    copy "$ROOT/results/proof_of_work_visuals/fig_docker_topology.pdf" docker_topology.pdf
  fi
fi

echo "Done. Upload ieee_overleaf/ to Overleaf (zip the folder)."
