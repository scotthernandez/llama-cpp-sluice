import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sluice.server
from sluice.server import app

def test_admin_drain():
    client = TestClient(app)
    response = client.post("/v1/admin/drain")
    # Verify why it might be 404. 
    # Check if the prefix /v1/admin/drain is correct.
    assert response.status_code == 200
    assert sluice.server.BANK.is_draining is True

@pytest.mark.asyncio
async def test_chat_basic():
    client = TestClient(app)
    with patch("sluice.server.low_level_generate") as mock_gen, \
         patch("sluice.server.get_tokens", return_value=[1,2,3]):
        mock_gen.return_value = ("Hi", 3, 1)
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}]
        })
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "Hi"

@pytest.mark.asyncio
async def test_radix_pinning():
    client = TestClient(app)
    from sluice.server import CACHE
    with patch("sluice.server.low_level_generate", return_value=("Cached", 200, 5)), \
         patch("sluice.server.get_tokens", return_value=[1] * 200):
        
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "long"}],
            "cache_prompt": True
        })
        assert response.status_code == 200
        assert len(CACHE.cache) == 1
