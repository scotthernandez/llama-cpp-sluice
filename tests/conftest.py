import sys
from unittest.mock import MagicMock, AsyncMock
import os

# 1. Mock llama_cpp and its internals BEFORE any imports
mock_llama = MagicMock()
mock_internals = MagicMock()

# Set successful defaults
mock_llama.llama_decode.return_value = 0
mock_llama.llama_batch_init.return_value = MagicMock()
mock_llama.llama_tokenize.return_value = 5

sys.modules['llama_cpp'] = mock_llama
sys.modules['llama_cpp._internals'] = mock_internals

# 2. Setup mock model/context classes
mock_internals.LlamaModel.return_value = MagicMock()
mock_internals.LlamaContext.return_value = MagicMock()

# 3. Import Sluice modules and inject mock dependencies
import sluice.bank
import sluice.engine
import sluice.server

# 4. Initialize globals to avoid NoneType errors in tests
sluice.server.ENGINE = MagicMock()
sluice.server.ENGINE.get_train_n_ctx.return_value = 32768
sluice.server.ENGINE.get_chat_template.return_value = None
sluice.server.ENGINE.get_n_embd.return_value = 128
sluice.server.ENGINE.get_embeddings.return_value = [0.1] * 128

# Use a real TokenBank but mock the hooks
sluice.server.BANK = sluice.bank.TokenBank(98304, 32768)

# 5. Provide fixtures
import pytest

@pytest.fixture(autouse=True)
def mock_llama_cpp_fixture():
    return mock_llama

@pytest.fixture(autouse=True)
def clean_bank():
    sluice.server.BANK.used = 0
    sluice.server.BANK.active_seqs = {}
    sluice.server.BANK.pinned_seqs = {}
    sluice.server.BANK.waiting_large = 0
    sluice.server.BANK.is_draining = False
    sluice.server.BANK.is_expanded = False
    sluice.server.BANK.total = 98304
