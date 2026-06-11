import pytest
from unittest.mock import MagicMock, patch
from sluice.engine import SluiceEngine

@pytest.fixture
def mock_sluice_internals():
    with patch('sluice.engine.LlamaModel') as mock_model, \
         patch('sluice.engine.LlamaContext') as mock_ctx, \
         patch('llama_cpp.llama_model_default_params'), \
         patch('llama_cpp.llama_context_default_params'), \
         patch('llama_cpp.llama_get_memory'):
        
        # Mock model and ctx instances
        mock_model.return_value = MagicMock()
        mock_ctx.return_value = MagicMock()
        
        yield {
            'LlamaModel': mock_model,
            'LlamaContext': mock_ctx
        }

def test_engine_init(mock_sluice_internals):
    engine = SluiceEngine(model_path="fake.gguf", total_tokens=1024)
    assert engine.total_tokens == 1024
    mock_sluice_internals['LlamaModel'].assert_called_once()
    mock_sluice_internals['LlamaContext'].assert_called_once()

def test_engine_hot_swap(mock_sluice_internals):
    engine = SluiceEngine(model_path="fake.gguf", total_tokens=1024)
    engine.hot_swap_context(2048)
    assert engine.total_tokens == 2048
    assert mock_sluice_internals['LlamaContext'].call_count == 2
