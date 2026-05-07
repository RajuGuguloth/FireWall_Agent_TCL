#!/bin/bash
echo "Starting HTTP flood attack capture..."
sudo tcpdump -i lo0 'port 8080' -w httpflood_attack.pcap &
TCPDUMP_PID=$!
sleep 3
echo "Running HTTP flood (500 requests)..."
for i in {1..500}; do
  curl -s http://localhost:8080/ > /dev/null &
done
wait
sleep 3
echo "Stopping capture..."
sudo kill $TCPDUMP_PID
sleep 2
echo "Done! Checking file..."
ls -lh httpflood_attack.pcap
tcpdump -r httpflood_attack.pcap | wc -l
