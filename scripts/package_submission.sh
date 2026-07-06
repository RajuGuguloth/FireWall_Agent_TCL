#!/usr/bin/env bash
# Package Hybrid-Sentinel R18 for internship / TCL archival submission.
# Usage: bash scripts/package_submission.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/submission"
STAMP="$(date +%Y%m%d)"
SRC_ZIP="$OUT/firewall_ml_project_R18_source_${STAMP}.zip"
MODELS_ZIP="$OUT/firewall_ml_project_R18_models_${STAMP}.zip"

mkdir -p "$OUT"

echo "==> Packaging source (code + docs + results JSON) ..."
cd "$ROOT"
zip -r "$SRC_ZIP" \
  README.md config.py requirements.txt requirements_api.txt \
  Dockerfile docker-compose.yml \
  api/ src/ scripts/ tests/ docs/ legacy/ \
  results/r18_tier_metrics.json \
  results/r18_latency_benchmark.json \
  results/r18_api_latency_check.json \
  results/Hybrid_Sentinel_Job_Portfolio_Report.md \
  results/ieee_figures/ \
  results/submission_figures/ \
  results/submission_metrics_r18.json \
  results/submission_metrics_table.tex \
  -x "*.pyc" "*__pycache__*" "*.DS_Store" \
     "*/.venv/*" "*/venv/*" \
     "legacy/tier1/*" \
  2>/dev/null || true

# Add report deliverables if present
for report_file in \
  Hybrid_Sentinel_IEEE_Overleaf.zip \
  Hybrid_Sentinel__A_Three_Tier_Confidence_GatedAI_Firewall_for_NDN_Edge_Deployment.pdf
do
  if [[ -f "$report_file" ]]; then
    zip -u "$SRC_ZIP" "$report_file" >/dev/null
  fi
done

# Add optional cascade flow log if present
if [[ -f results/cascade_flow_final.txt ]]; then
  zip -u "$SRC_ZIP" results/cascade_flow_final.txt
fi

echo "==> Packaging R18 models ..."
zip -j "$MODELS_ZIP" \
  models/tier1_gate_v6.pkl \
  models/tier2_cnn_gru_v1_r18.pth \
  models/tier2_r18_temperature.json \
  models/tier3_oneclass_v6.pkl \
  models/serialized/v6_scaler.pkl \
  models/serialized/v6_encoder.pkl \
  models/onnx/tier2_cnn_gru_r18.onnx \
  models/onnx/tier2_embedding_r18.onnx \
  2>/dev/null || {
    echo "WARNING: Some model files missing — train/export pipeline may be needed."
  }

echo ""
echo "Done."
echo "  Source : $SRC_ZIP"
echo "  Models : $MODELS_ZIP"
echo ""
echo "Also submit:"
echo "  docs/FINAL_SUBMISSION_REPORT.md"
echo "  docs/DESIGN_AND_IMPLEMENTATION.md"
du -sh "$OUT"/*.zip 2>/dev/null || true
