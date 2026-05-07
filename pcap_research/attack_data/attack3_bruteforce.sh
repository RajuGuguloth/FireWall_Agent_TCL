#!/bin/bash
echo "Starting brute force attack capture..."
sudo tcpdump -i lo0 'port 8080' -w bruteforce_attack.pcap &
TCPDUMP_PID=$!
sleep 3
echo "Running brute force (100 login attempts)..."
for i in {1..100}; do
  curl -s -X POST http://localhost:8080/login \
    -d "username=admin&password=pass$i" \
    > /dev/null
  sleep 0.05
done
sleep 3
echo "Stopping capture..."
sudo kill $TCPDUMP_PID
sleep 2
echo "Done! Checking file..."
ls -lh bruteforce_attack.pcap
tcpdump -r bruteforce_attack.pcap | wc -l
