#!/bin/bash
echo "=================================================="
echo "Generating diverse port scan attacks..."
echo "=================================================="

cd data/raw/attacks

# Attack 1: SYN Scan (stealthy)
echo "1. Generating SYN scan..."
sudo tcpdump -i lo0 'port 8080' -w portscan_syn.pcap &
TCPDUMP_PID=$!
sleep 3
nmap -sS -p 8000-8100 localhost > /dev/null 2>&1
sleep 3
sudo kill $TCPDUMP_PID
echo "   ✅ SYN scan captured"

# Attack 2: Connect Scan (full TCP)
echo "2. Generating Connect scan..."
sudo tcpdump -i lo0 'port 8080' -w portscan_connect.pcap &
TCPDUMP_PID=$!
sleep 3
nmap -sT -p 8000-8200 localhost > /dev/null 2>&1
sleep 3
sudo kill $TCPDUMP_PID
echo "   ✅ Connect scan captured"

# Attack 3: Aggressive scan (OS detection + version)
echo "3. Generating Aggressive scan..."
sudo tcpdump -i lo0 'port 8080' -w portscan_aggressive.pcap &
TCPDUMP_PID=$!
sleep 3
nmap -A -p 8000-8100 localhost > /dev/null 2>&1
sleep 5
sudo kill $TCPDUMP_PID
echo "   ✅ Aggressive scan captured"

# Attack 4: UDP scan
echo "4. Generating UDP scan..."
sudo tcpdump -i lo0 'portrange 8000-8100' -w portscan_udp.pcap &
TCPDUMP_PID=$!
sleep 3
nmap -sU -p 8000-8050 localhost > /dev/null 2>&1
sleep 5
sudo kill $TCPDUMP_PID
echo "   ✅ UDP scan captured"

# Attack 5: Fast scan (top 100 ports)
echo "5. Generating Fast scan..."
sudo tcpdump -i lo0 -w portscan_fast.pcap &
TCPDUMP_PID=$!
sleep 3
nmap -F localhost > /dev/null 2>&1
sleep 3
sudo kill $TCPDUMP_PID
echo "   ✅ Fast scan captured"

cd ../../..
echo ""
echo "=================================================="
echo "✅ All port scans generated!"
echo "=================================================="
ls -lh data/raw/attacks/portscan_*.pcap
