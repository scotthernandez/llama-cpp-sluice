import pytest
from sluice.engine import SluiceEngine
from sluice.pools import PoolConfig
from unittest.mock import MagicMock
import sys

def test_engine_init():
    # llama_cpp is mocked in conftest
    import llama_cpp._internals
    pools = [PoolConfig(name="p", max_tokens=100)]
    engine = SluiceEngine("fake.gguf", pools)
    assert engine.model_path == "fake.gguf"
    assert "p" in engine.contexts

def test_engine_hot_swap():
    import llama_cpp._internals
    pools = [PoolConfig(name="p", max_tokens=100)]
    engine = SluiceEngine("fake.gguf", pools)
    
    new_c = PoolConfig(name="p", max_tokens=200)
    engine.hot_swap_context("p", new_c)
    assert "p" in engine.contexts
