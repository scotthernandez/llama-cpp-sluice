import llama_cpp
import os
import json
from typing import Optional, List, Dict
from llama_cpp._internals import LlamaModel, LlamaContext
from .pools import PoolConfig

class SluiceEngine:
    def __init__(self, model_path: str, pools: List[PoolConfig]):
        print(f"[SLUICE] Initializing Multi-Pool Engine: {model_path}")
        self.model_path = model_path
        
        # 1. Load Model Weights once
        mparams = llama_cpp.llama_model_default_params()
        mparams.n_gpu_layers = -1
        self.model = LlamaModel(path_model=model_path, params=mparams, verbose=False)
        
        # 2. Create Contexts for each Pool
        self.contexts: Dict[str, LlamaContext] = {}
        for config in pools:
            self.contexts[config.name] = self._create_context(config)
            print(f"[ENGINE] Initialized Pool '{config.name}' ({config.max_tokens} tokens, K={config.type_k}, V={config.type_v})")
        
    def _create_context(self, config: PoolConfig):
        cparams = llama_cpp.llama_context_default_params()
        cparams.n_ctx = config.max_tokens
        cparams.type_k = config.type_k
        cparams.type_v = config.type_v
        cparams.flash_attn = True
        cparams.embeddings = True
        return LlamaContext(model=self.model, params=cparams, verbose=False)

    def hot_swap_context(self, pool_name: str, new_config: PoolConfig):
        """Recreates a specific context without dropping weights."""
        print(f"[ENGINE] Hot-swapping pool '{pool_name}'...")
        self.contexts[pool_name] = self._create_context(new_config)

    def defrag(self, pool_name: str):
        """Triggers internal KV cache compaction for a specific pool."""
        llama_cpp.llama_kv_cache_defrag(self.contexts[pool_name].ctx)

    def get_frag_ratio(self, pool_name: str) -> float:
        ctx = self.contexts[pool_name]
        n_used = llama_cpp.llama_get_kv_cache_used_cells(ctx.ctx)
        n_total = llama_cpp.llama_n_ctx(ctx.ctx)
        if n_total == 0: return 0.0
        return 1.0 - (n_used / n_total)

    def get_context_ptr(self, pool_name: str):
        return self.contexts[pool_name].ctx

    def get_model_ptr(self):
        return self.model.model

    def clone_sequence(self, pool_name: str, src_sid: int, dest_sid: int, length: int):
        llama_cpp.llama_memory_seq_cp(self.contexts[pool_name].ctx, src_sid, dest_sid, 0, length)

    def remove_sequence(self, pool_name: str, sid: int):
        llama_cpp.llama_memory_seq_rm(self.contexts[pool_name].ctx, sid, -1, -1)

    def get_train_n_ctx(self) -> int:
        return llama_cpp.llama_n_ctx_train(self.model.model)

    def get_chat_template(self) -> Optional[str]:
        return self.model.metadata().get("tokenizer.chat_template")

    def get_n_embd(self) -> int:
        return llama_cpp.llama_n_embd(self.model.model)

    def get_embeddings(self, pool_name: str, sid: int) -> List[float]:
        embd_ptr = llama_cpp.llama_get_embeddings_seq(self.contexts[pool_name].ctx, sid)
        if not embd_ptr:
            raise RuntimeError(f"Failed to retrieve embeddings from pool {pool_name}")
        n_embd = self.get_n_embd()
        return [float(embd_ptr[i]) for i in range(n_embd)]
