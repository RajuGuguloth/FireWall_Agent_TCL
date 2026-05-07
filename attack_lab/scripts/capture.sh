#!/bin/bash
# ─────────────────────────────────────────────
#  NDN Firewall Project — Traffic Capture Tool
#  Tushar Sood | Week 1
#  Usage: ./capture.sh LABEL DURATION_SECONDS
# ─────────────────────────────────────────────

LABEL=$1
DURATION=$2
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PCAP="/captures/pcap/${LABEL}_${TIMESTAMP}.pcap"
FLOWS="/captures/flows/${LABEL}_${TIMESTAMP}_flows.csv"
LOG="/captures/logs/${LABEL}_${TIMESTAMP}.log"

if [ -z "$LABEL" ] || [ -z "$DURATION" ]; then
  echo "Usage: ./capture.sh LABEL DURATION_SECONDS"
  echo "Example: ./capture.sh BRUTE_FORCE 120"
  exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee $LOG
echo "🔴 START CAPTURE" | tee -a $LOG
echo "   Label:     $LABEL" | tee -a $LOG
echo "   Duration:  ${DURATION}s" | tee -a $LOG
echo "   Output:    $PCAP" | tee -a $LOG
echo "   Time:      $(date)" | tee -a $LOG
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a $LOG

tcpdump -i any \
  host 10.10.0.10 \
  -w $PCAP \
  -G $DURATION \
  -W 1 \
  2>> $LOG

echo "" | tee -a $LOG
echo "✅ Capture complete → $PCAP" | tee -a $LOG
PCAP_SIZE=$(du -sh $PCAP 2>/dev/null | cut -f1)
echo "   File size: $PCAP_SIZE" | tee -a $LOG

echo "" | tee -a $LOG
echo "⚙️  Converting to flow CSV..." | tee -a $LOG

pip3 install cicflowmeter -q 2>/dev/null
cicflowmeter -f $PCAP -c $FLOWS 2>> $LOG

if [ -f "$FLOWS" ]; then
  FLOW_COUNT=$(wc -l < $FLOWS)
  echo "✅ Flows saved → $FLOWS" | tee -a $LOG
  echo "   Flow count: $FLOW_COUNT rows" | tee -a $LOG
else
  echo "⚠️  Flow conversion failed — check $LOG" | tee -a $LOG
fi

echo "" | tee -a $LOG
echo "🗑️  Removing raw pcap to save disk space..." | tee -a $LOG
rm -f $PCAP
echo "   Done. Flows kept, pcap deleted." | tee -a $LOG
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a $LOG
