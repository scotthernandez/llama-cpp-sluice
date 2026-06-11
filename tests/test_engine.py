import pytest
from sluice.engine import SluiceEngine

def test_engine_init():
    engine = SluiceEngine("fake.gguf", 1024)
    assert engine.total_tokens == 1024

def test_engine_hot_swap():
    engine = SluiceEngine("fake.gguf", 1024)
    engine.hot_swap_context(2048)
    assert engine.total_tokens == 2048
