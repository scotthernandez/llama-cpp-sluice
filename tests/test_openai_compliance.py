import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import os
import json

# Set dummy env vars for Sluice
os.environ["SLUICE_MODEL_PATH"] = "fake.gguf"

from sluice.server import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_engine_and_bank():
    with patch("sluice.server.ENGINE") as mock_engine, \
         patch("sluice.server.BANK.acquire", return_value=1), \
         patch("sluice.server.BANK.release"), \
         patch("llama_cpp.llama_memory_seq_rm"):
        mock_engine.get_model_ptr.return_value = MagicMock()
        mock_engine.get_context_ptr.return_value = MagicMock()
        mock_engine.get_memory.return_value = MagicMock()
        yield

def test_openai_compatibility_raw_json():
    """
    Directly verify that our FastAPI server output matches the OpenAI spec.
    This is more reliable than trying to deep-patch LiteLLM in a unit test.
    """
    with patch("sluice.server.low_level_generate") as mock_gen:
        mock_gen.return_value = ("Hello from Sluice!", 5, 3)
        
        payload = {
            "model": "sluice",
            "messages": [{"role": "user", "content": "hi"}]
        }
        response = client.post("/v1/chat/completions", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify OpenAI required fields
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert data["choices"][0]["message"]["content"] == "Hello from Sluice!"
        assert data["usage"]["total_tokens"] == 8

def test_litellm_parsing_logic():
    """
    Verify that the LiteLLM library can correctly parse OUR response JSON.
    """
    from litellm import ModelResponse
    
    our_response = {
        "id": "sluice-1",
        "object": "chat.completion",
        "created": 123456789,
        "model": "sluice-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Verified."},
                "finish_reason": "stop"
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }
    
    # LiteLLM uses its own classes for responses
    parsed = ModelResponse(**our_response)
    assert parsed.choices[0].message.content == "Verified."
    assert parsed.usage.total_tokens == 15

def test_openai_streaming_format():
    """Verify that our streaming output follows the SSE data: {json} format."""
    with patch("sluice.server.low_level_stream_generator") as mock_stream:
        def chunk_gen(*args, **kwargs):
            yield 'data: {"id": "1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": null}]}\n\n'
            yield 'data: [DONE]\n\n'

        mock_stream.side_effect = chunk_gen
        
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True
        })
        
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        assert "data: {" in response.text
        assert "[DONE]" in response.text
