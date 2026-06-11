import sys
from unittest.mock import MagicMock, patch

import pytest
import asyncio
import os
import json

import sluice.server
from sluice.server import app, CACHE
from sluice.bank import TokenBank

@pytest.fixture(autouse=True)
def reset_state():
    CACHE.cache = {}

@pytest.mark.asyncio
async def test_radix_cache_pinning():
    from fastapi.testclient import TestClient
    client = TestClient(app)
    
    with patch("sluice.server.low_level_generate") as mock_gen, \
         patch("sluice.server.get_tokens") as mock_tokens:
        
        mock_tokens.return_value = [1] * 200
        mock_gen.return_value = ("Cached", 200, 5)
        
        # 200 tokens * 1.2 = 240 ctx
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "cache_prompt": True
        })
        assert response.status_code == 200
        assert len(CACHE.cache) == 1
        assert sluice.server.BANK.used == 2048

@pytest.mark.asyncio
async def test_embeddings_endpoint():
    from fastapi.testclient import TestClient
    client = TestClient(app)
    
    with patch("sluice.server.get_tokens", return_value=[1,2,3]):
        response = client.post("/v1/embeddings", json={
            "input": ["a", "b"]
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2
        assert data["data"][0]["embedding"] == [0.1] * 128

@pytest.mark.asyncio
async def test_grammar_json_mode():
    from fastapi.testclient import TestClient
    client = TestClient(app)
    import llama_cpp
    
    with patch("sluice.server.low_level_generate") as mock_gen, \
         patch("sluice.server.get_tokens", return_value=[1,2,3]):
        mock_gen.return_value = ("{}", 10, 1)
        
        payload = {
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": {
                "type": "json_object",
                "schema": {"properties": {"a": {"type": "integer"}}}
            }
        }
        
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert llama_cpp.LlamaGrammar.from_json_schema.called
