#!/bin/bash
# scavenge-example.sh: Example script to free VRAM on Proxmox/Docker

# 1. Stop Whisper SST Container
echo "[SCAVENGE] Stopping Whisper SST..."
pct stop 105 || docker stop whisper-sst

# 2. Unload TTS Model (if running as a separate service)
echo "[SCAVENGE] Unloading TTS Service..."
systemctl stop tts-piper.service

# 3. Wait for VRAM to settle
sleep 2

# 4. (Optional) Trigger a resize on Sluice to use the newly freed VRAM
# curl -X POST http://localhost:8001/v1/admin/resize -H "Content-Type: application/json" -d '{"new_size": 131072}'

echo "[SCAVENGE] Complete. VRAM reclaimed."
