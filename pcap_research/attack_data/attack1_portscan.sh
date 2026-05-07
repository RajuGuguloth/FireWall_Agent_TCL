#!/bin/bash
echo "Starting port scan attack capture..."
sudo tcpdump -i lo0 'port 8080' -w portscan_attack.pcap &
TCPDUMP_PID=$!
sleep 5
echo "Running nmap port scan..."
nmap -sT -p 8000-9000 localhost
sleep 5
echo "Stopping capture..."
sudo kill $TCPDUMP_PID
sleep 2
echo "Done! Checking file..."
ls -lh portscan_attack.pcap
