#!/usr/bin/env bash
# run on the host, not in jupyter
set -e
cd "$(dirname "$0")/.."
sudo iptables -I INPUT -p tcp --dport 8080 -j ACCEPT || true
docker compose up -d --build
sleep 5
docker compose ps
IP=$(hostname -I | awk '{print $1}')
echo
echo "Dashboard:  http://${IP}:8080   (use the public IP shown in the LinuxONE dashboard)"
echo "Health:     curl -s localhost:8080/api/health"
echo "Stream:     docker exec -it safepause python scripts/simulate_stream.py --reset --rate 15"
