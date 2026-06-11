import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
import sluice.server
from sluice.server import app, CACHE

client = TestClient(app)

@pytest.mark.asyncio
async def test_radix_pinning():
    with patch("sluice.server.low_level_generate", return_value=("Cached", 200, 5, "stop")), \
         patch("sluice.server.get_tokens", return_value=[1] * 200):
        
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "long"}],
            "cache_prompt": True
        })
        assert response.status_code == 200
        assert len(CACHE.cache) == 1
        assert sluice.server.BANK.used["precision"] == 2048

def test_embeddings_endpoint():
    import llama_cpp
    with patch("sluice.server.get_tokens", return_value=[1,2,3]), \
         patch("llama_cpp.llama_decode", return_value=0):
        response = client.post("/v1/embeddings", json={
            "input": ["a", "b"]
        })
        assert response.status_code == 200
        assert len(response.json()["data"]) == 2
