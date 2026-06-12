import pytest
from sluice.engine import SluiceEngine
from sluice.pools import PoolConfig
from unittest.mock import MagicMock
import sys

def test_engine_init():
    # llama_cpp is mocked in conftest
    import llama_cpp._internals
    pool = PoolConfig(name="p", max_tokens=100)
    engine = SluiceEngine("fake.gguf", pool)
    assert engine.model_path == "fake.gguf"
    assert hasattr(engine, 'ctx_ptr')

def test_engine_hot_swap():
    pass # engine_hot_swap was removed
