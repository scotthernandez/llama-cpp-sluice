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
        
        # 2. Create the Master Context (The Pool)
        cparams = llama_cpp.llama_context_default_params()
        cparams.n_ctx = total_tokens
        cparams.flash_attn = True
        self.context = LlamaContext(model=self.model, params=cparams, verbose=False)
        
    def get_context_ptr(self):
        return self.context.ctx

    def get_model_ptr(self):
        return self.model.model
