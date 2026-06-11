import llama_cpp
from llama_cpp._internals import LlamaModel, LlamaContext

class SluiceEngine:
    def __init__(self, model_path: str, total_tokens: int):
        print(f"[SLUICE] Initializing Shared Weight Engine: {model_path}")
        self.model_path = model_path
        self.total_tokens = total_tokens
        
        # 1. Load Model Weights once
        mparams = llama_cpp.llama_model_default_params()
        mparams.n_gpu_layers = -1  # Max offload
        self.model = LlamaModel(path_model=model_path, params=mparams, verbose=False)
        
        # 2. Create Initial Master Context
        self.context = self._create_context(total_tokens)
        
    def _create_context(self, n_ctx: int):
        cparams = llama_cpp.llama_context_default_params()
        cparams.n_ctx = n_ctx
        cparams.flash_attn = True
        return LlamaContext(model=self.model, params=cparams, verbose=False)

    def hot_swap_context(self, new_size: int):
        """Recreates the context with a new size without dropping weights."""
        print(f"[ENGINE] Hot-swapping context: {self.total_tokens} -> {new_size}")
        # Managed LlamaContext handles llama_free automatically on destruction
        self.context = self._create_context(new_size)
        self.total_tokens = new_size

    def get_memory(self):
        """Returns the memory object for advanced cache manipulation."""
        return llama_cpp.llama_get_memory(self.context.ctx)

    def get_context_ptr(self):
        return self.context.ctx

    def get_model_ptr(self):
        return self.model.model
