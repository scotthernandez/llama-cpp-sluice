import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
import sluice.server
from sluice.server import app

def test_admin_drain():
    client = TestClient(app)
    response = client.post("/v1/admin/drain")
    assert response.status_code == 200
    assert sluice.server.BANK.is_draining is True

@pytest.mark.asyncio
async def test_chat_basic():
    client = TestClient(app)
    with patch("sluice.server.low_level_generate") as mock_gen, \
         patch("sluice.server.get_tokens", return_value=[1,2,3]):
        mock_gen.return_value = ("Hi", 3, 1, "stop")
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}]
        })
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "Hi"
