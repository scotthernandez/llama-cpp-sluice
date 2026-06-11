import pytest
import asyncio
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
    BANK.total = 98304 # Default
    BANK.reserved_for_large = 32768
    BANK.large_threshold = 16384

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
        response = client.post("/v1/admin/resize", json={"new_size": 4096})
        assert response.status_code == 200
        assert response.json()["new_size"] == 4096
        mock_swap.assert_called_with(4096)

@pytest.mark.asyncio
async def test_elasticity_emergency_expansion():
    with patch("sluice.server.AUTO_ELASTICITY", True), \
         patch("sluice.server.ENGINE.hot_swap_context") as mock_swap, \
         patch("sluice.server.low_level_generate") as mock_gen:
        
        mock_gen.return_value = ("Expanded", 1, 1)
        
        # Force a timeout on first acquire to trigger expansion
        BANK.total = 10
        BANK.reserved_for_large = 5
        BANK.large_threshold = 20
        
        # Use a mock to make the first acquire call fail fast
        with patch.object(BANK, 'acquire', side_effect=[asyncio.TimeoutError("first fail"), 123]):
            response = client.post("/v1/ctx/20/chat/completions", json={
                "messages": [{"role": "user", "content": "test"}]
            })
        
        assert response.status_code == 200
        assert BANK.is_expanded is True
        assert BANK.total == 20
        mock_swap.assert_called_with(20)
