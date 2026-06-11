import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sluice.server
from sluice.server import app

client = TestClient(app)

@pytest.mark.asyncio
async def test_sampling_and_stop_logic():
    with patch("sluice.server.low_level_generate") as mock_gen, \
         patch("sluice.server.get_tokens", return_value=[1,2,3]):
        
        # Simulating a stop sequence match
        mock_gen.return_value = ("Observation: tool output", 3, 5, "stop")
        
        payload = {
            "messages": [{"role": "user", "content": "run tool"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 20,
            "min_p": 0.1,
            "stop": ["Observation:"]
        }
        
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        
        # Verify the call to low_level_generate received the request object with sampling params
        args, kwargs = mock_gen.call_args
        request_obj = args[7] # 8th positional arg is 'request'
        assert request_obj.temperature == 0.7
        assert request_obj.stop == ["Observation:"]

@pytest.mark.asyncio
async def test_deterministic_seed():
    with patch("sluice.server.low_level_generate", return_value=("fixed", 3, 1, "stop")), \
         patch("sluice.server.get_tokens", return_value=[1,2,3]):
        
        payload = {
            "messages": [{"role": "user", "content": "deterministic"}],
            "seed": 42
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        
        args, _ = sluice.server.low_level_generate.call_args
        assert args[7].seed == 42
