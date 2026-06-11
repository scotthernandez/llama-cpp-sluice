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
        cparams.embeddings = True # Enable embeddings
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

    def clone_sequence(self, src_sid: int, dest_sid: int, length: int):
        """Zero-copy clone of KV cache tokens from one sequence to another."""
        llama_cpp.llama_memory_seq_cp(self.context.ctx, src_sid, dest_sid, 0, length)

    def remove_sequence(self, sid: int):
        """Removes a specific sequence from the KV cache."""
        llama_cpp.llama_memory_seq_rm(self.context.ctx, sid, -1, -1)

    def get_train_n_ctx(self) -> int:
        """Returns the native training context limit of the model."""
        return llama_cpp.llama_n_ctx_train(self.model.model)

    def get_chat_template(self) -> Optional[str]:
        """Returns the Jinja2 chat template from model metadata if available."""
        return self.model.metadata().get("tokenizer.chat_template")

    def get_n_embd(self) -> int:
        """Returns the embedding dimension size."""
        return llama_cpp.llama_n_embd(self.model.model)

    def get_embeddings(self, sid: int) -> List[float]:
        """Retrieves embeddings for a processed sequence."""
        embd_ptr = llama_cpp.llama_get_embeddings_seq(self.context.ctx, sid)
        if not embd_ptr:
            raise RuntimeError(f"Failed to retrieve embeddings for sequence {sid}")
        
        n_embd = self.get_n_embd()
        return [float(embd_ptr[i]) for i in range(n_embd)]
