import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import os

# Set dummy env vars before importing server
os.environ["SLUICE_MODEL_PATH"] = "fake.gguf"

from sluice.server import app, BANK

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_engine():
    with patch("sluice.server.ENGINE") as mock:
        mock.get_model_ptr.return_value = MagicMock()
        mock.get_context_ptr.return_value = MagicMock()
        mock.get_memory.return_value = MagicMock()
        yield mock

@pytest.fixture(autouse=True)
def reset_bank():
    # Reset bank state before each test
    BANK.used = 0
    BANK.is_draining = False
    BANK.is_expanded = False
    BANK.waiting_large = 0
    BANK.active_seqs = {}

@pytest.fixture(autouse=True)
def mock_llama_cpp_rm():
    with patch("llama_cpp.llama_memory_seq_rm") as mock:
        yield mock

def test_root_not_found():
    response = client.get("/")
    assert response.status_code == 404

def test_chat_completions_basic(mock_llama_cpp_rm):
    with patch("sluice.server.low_level_generate") as mock_gen:
        mock_gen.return_value = ("Hello world", 5, 2)
        
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10
        }
        response = client.post("/v1/chat/completions", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Hello world"
        assert data["usage"]["total_tokens"] == 7
        mock_llama_cpp_rm.assert_called_once()

def test_virtual_url_context():
    with patch("sluice.server.low_level_generate") as mock_gen:
        mock_gen.return_value = ("Context test", 1, 1)
        response = client.post("/v1/ctx/16384/chat/completions", json={
            "messages": [{"role": "user", "content": "test"}]
        })
        assert response.status_code == 200

def test_admin_drain_resume():
    response = client.post("/v1/admin/drain")
    assert response.status_code == 200
    assert response.json()["status"] == "draining"
    assert BANK.is_draining is True
    
    response = client.post("/v1/admin/resume")
    assert response.status_code == 200
    assert BANK.is_draining is False

@pytest.mark.asyncio
async def test_admin_resize():
    with patch("sluice.server.ENGINE.hot_swap_context") as mock_swap:
        # Note: server.py expects BASE_POOL to be defined.
        # We use Body(..., embed=True) for resize.
        response = client.post("/v1/admin/resize", json={"new_size": 4096})
        assert response.status_code == 200
        assert response.json()["new_size"] == 4096
        mock_swap.assert_called_with(4096)
