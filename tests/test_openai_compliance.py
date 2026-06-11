import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sluice.server
from sluice.server import app

client = TestClient(app)

def test_openai_compatibility_raw_json():
    # Rely on conftest.py for BANK/ENGINE mocks
    with patch("sluice.server.low_level_generate") as mock_gen, \
         patch("sluice.server.get_tokens", return_value=[1,2,3]):
        mock_gen.return_value = ("Hello from Sluice!", 5, 3, "stop")
        
        response = client.post("/v1/chat/completions", json={
            "model": "sluice",
            "messages": [{"role": "user", "content": "hi"}]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"

def test_litellm_parsing_logic():
    from litellm import ModelResponse
    our_response = {
        "id": "sluice-1",
        "object": "chat.completion",
        "created": 123456789,
        "model": "sluice-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Verified."}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }
    parsed = ModelResponse(**our_response)
    assert parsed.choices[0].message.content == "Verified."

def test_openai_streaming_format():
    with patch("sluice.server.low_level_stream_start", return_value=3), \
         patch("sluice.server.low_level_stream_step") as mock_step, \
         patch("sluice.server.get_tokens", return_value=[1,2,3]):
        
        # (piece, finish_reason)
        mock_step.side_effect = [("Hi", None), (None, "stop")]
        
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True
        })
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        assert "data: {" in response.text
