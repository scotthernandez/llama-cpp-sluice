#!/bin/bash
# recovery-example.sh: Restore resources after a large inference completes

# 1. Start Whisper SST Container
echo "[RECOVERY] Restarting Whisper SST..."
pct start 105 || docker start whisper-sst

# 2. Restart TTS Service
echo "[RECOVERY] Restarting TTS Service..."
systemctl start tts-piper.service

echo "[RECOVERY] Complete. All services online."
