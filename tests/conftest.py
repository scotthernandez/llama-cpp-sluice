import sys
from unittest.mock import MagicMock, patch
import os

# 1. Mock llama_cpp and its internals
mock_llama = MagicMock()
mock_llama.LLAMA_SPLIT_MODE_LAYER = 2
mock_llama.LLAMA_SPLIT_MODE_NONE = 0
mock_llama.LLAMA_SPLIT_MODE_ROW = 1
mock_llama.GGML_TYPE_F16 = 1
mock_llama.GGML_TYPE_Q4_0 = 2
mock_llama.llama_decode.return_value = 0
mock_internals = MagicMock()
sys.modules['llama_cpp'] = mock_llama
sys.modules['llama_cpp._internals'] = mock_internals

# 2. Setup mock model/context classes
mock_model_obj = MagicMock()
mock_ctx_obj = MagicMock()
mock_internals.LlamaModel.return_value = mock_model_obj
mock_internals.LlamaContext.return_value = mock_ctx_obj

import pytest

try:
    import sluice.bank
except ImportError:
    sluice = MagicMock()
    sys.modules['sluice'] = sluice
    sys.modules['sluice.bank'] = MagicMock()

try:
    import sluice.engine
except ImportError:
    if 'sluice' not in sys.modules:
        sluice = MagicMock()
        sys.modules['sluice'] = sluice
    sys.modules['sluice.engine'] = MagicMock()

try:
    import sluice.server
    # Provide pools that actually work with the barrier
    from sluice.pools import PoolConfig
    TEST_POOLS = [
        PoolConfig(name="precision", max_tokens=100000, precision_threshold=8192),
        PoolConfig(name="efficiency", max_tokens=100000, precision_threshold=0)
    ]
    sluice.server.POOLS = TEST_POOLS
    sluice.server.BASE_POOL = TEST_POOLS[0].max_tokens
    sluice.server.LARGE_THRESHOLD = 16384
    sluice.server.RESERVED_POOL = 1000

    sluice.server.BANK = sluice.bank.TokenBank(100000, 1000)

    # Initialize globals for server tests
    sluice.server.engine = MagicMock()
    sluice.server.engine.get_model_ptr.return_value = MagicMock()
    sluice.server.engine.get_context_ptr.return_value = MagicMock()
    sluice.server.engine.get_chat_template.return_value = None
    sluice.server.engine.get_train_n_ctx.return_value = 32768
    sluice.server.engine.get_n_embd.return_value = 128
    sluice.server.engine.get_embeddings.return_value = [0.1] * 128
    sluice.server.engine.get_frag_ratio.return_value = 0.0
except ImportError:
    # If server can't be imported, create minimal stubs
    if 'sluice' not in sys.modules:
        sluice = MagicMock()
        sys.modules['sluice'] = sluice
    sys.modules['sluice.server'] = MagicMock()
    sluice = sys.modules['sluice']
    if not hasattr(sluice, 'server'):
        sluice.server = MagicMock()
    if not hasattr(sluice, 'bank'):
        sluice.bank = MagicMock()
    sluice.server.BANK = MagicMock()
    sluice.server.BANK.used = 0
    sluice.server.BANK.active_seqs = {}
    sluice.server.BANK.pinned_seqs = {}
    sluice.server.BANK.available_sids = list(range(16))
    sluice.server.BANK.waiting_large = 0
    sluice.server.BANK.is_draining = False
    sluice.server.BANK.is_expanded = False
    sluice.server.BANK.capacity = 100000

@pytest.fixture(autouse=True)
def clean_bank_fixture():
    bank = sluice.server.BANK
    bank.used = 0
    bank.active_seqs = {}
    bank.pinned_seqs = {}
    bank.available_sids = list(range(16))
    bank.waiting_large = 0
    bank.is_draining = False
    bank.is_expanded = False
    bank.capacity = 100000
    yield bank
