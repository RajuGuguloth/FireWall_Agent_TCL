#!/bin/bash
DURATION=${1:-300}
OUTPUT_DIR="raw"
INTERFACE="en0"
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/baseline_${TIMESTAMP}.pcap"
echo "Starting packet capture..."
echo "Duration: $DURATION seconds"
echo "Output: $OUTPUT_FILE"
sudo tcpdump -i "$INTERFACE" -G "$DURATION" -W 1 -w "$OUTPUT_FILE"
echo "Capture complete! File: $OUTPUT_FILE"
