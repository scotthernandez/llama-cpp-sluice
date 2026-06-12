import llama_cpp
import os
import json
import ctypes
from typing import Optional, List, Dict
from .pools import PoolConfig

class SluiceEngine:
    def __init__(self, model_path: str, pools: List[PoolConfig], 
                 mmproj_path: Optional[str] = None,
                 tensor_split: Optional[List[float]] = None,
                 n_batch: int = 512,
                 n_ubatch: int = 256,
                 n_gpu_layers: int = -1,
                 split_mode: int = 2,
                 flash_attn: bool = True,
                 embeddings: bool = True,
                 n_threads: Optional[int] = None,
                 n_threads_batch: Optional[int] = None,
                 use_mlock: bool = False,
                 use_mmap: bool = True,
                 rope_freq_base: Optional[float] = None,
                 rope_freq_scale: Optional[float] = None):
        
        print(f"[SLUICE] Initializing Raw Pointer Engine: {model_path}")
        self.model_path = model_path
        self.n_batch = n_batch
        self.n_ubatch = n_ubatch
        self.flash_attn = flash_attn
        self.embeddings = embeddings
        self.n_threads = n_threads or os.cpu_count() or 4
        self.n_threads_batch = n_threads_batch or os.cpu_count() or 4
        
        # 1. Initialize Backend
        llama_cpp.llama_backend_init()
        
        # 2. Load Model (Raw Pointer)
        mparams = llama_cpp.llama_model_default_params()
        mparams.n_gpu_layers = n_gpu_layers
        mparams.split_mode = split_mode
        mparams.use_mlock = use_mlock
        mparams.use_mmap = use_mmap
        
        if tensor_split:
            ts_array = (ctypes.c_float * len(tensor_split))(*tensor_split)
            mparams.tensor_split = ts_array

        if rope_freq_base is not None: mparams.rope_freq_base = ctypes.c_float(rope_freq_base)
        if rope_freq_scale is not None: mparams.rope_freq_scale = ctypes.c_float(rope_freq_scale)

        self.model_ptr = llama_cpp.llama_load_model_from_file(model_path.encode('utf-8'), mparams)
        if not self.model_ptr:
            raise RuntimeError(f"Failed to load model: {model_path}")
        
        # 3. Create Contexts (Raw Pointers)
        self.ctx_ptrs: Dict[str, ctypes.c_void_p] = {}
        for config in pools:
            self.ctx_ptrs[config.name] = self._create_raw_context(config)
            print(f"[ENGINE] Created Raw Pool '{config.name}'")
        
    def _create_raw_context(self, config: PoolConfig):
        cparams = llama_cpp.llama_context_default_params()
        cparams.n_ctx = config.max_tokens
        cparams.n_batch = self.n_batch
        cparams.n_ubatch = self.n_ubatch
        cparams.n_threads = self.n_threads
        cparams.n_threads_batch = self.n_threads_batch
        cparams.type_k = config.type_k
        cparams.type_v = config.type_v
        cparams.flash_attn = self.flash_attn
        cparams.embeddings = self.embeddings
        
        ctx = llama_cpp.llama_new_context_with_model(self.model_ptr, cparams)
        if not ctx:
            raise RuntimeError(f"Failed to create context for pool {config.name}")
        return ctx

    def hot_swap_context(self, pool_name: str, new_config: PoolConfig):
        """Safely free and recreate a context pointer."""
        if pool_name in self.ctx_ptrs:
            print(f"[ENGINE] Manually freeing context for {pool_name}...")
            llama_cpp.llama_free(self.ctx_ptrs[pool_name])
        self.ctx_ptrs[pool_name] = self._create_raw_context(new_config)

    def defrag(self, pool_name: str):
        llama_cpp.llama_kv_cache_defrag(self.ctx_ptrs[pool_name])

    def get_frag_ratio(self, pool_name: str) -> float:
        try:
            ctx = self.ctx_ptrs[pool_name]
            if hasattr(llama_cpp, "llama_get_kv_cache_used_cells"):
                n_used = llama_cpp.llama_get_kv_cache_used_cells(ctx)
                n_total = llama_cpp.llama_n_ctx(ctx)
                if n_total == 0: return 0.0
                return 1.0 - (n_used / n_total)
        except Exception: pass
        return 0.0

    def get_context_ptr(self, pool_name: str):
        return self.ctx_ptrs[pool_name]

    def get_model_ptr(self):
        return self.model_ptr

    def clone_sequence(self, pool_name: str, src_sid: int, dest_sid: int, length: int):
        llama_cpp.llama_memory_seq_cp(self.ctx_ptrs[pool_name], src_sid, dest_sid, 0, length)

    def remove_sequence(self, pool_name: str, sid: int):
        llama_cpp.llama_memory_seq_rm(self.ctx_ptrs[pool_name], sid, -1, -1)

    def get_train_n_ctx(self) -> int:
        return llama_cpp.llama_n_ctx_train(self.model_ptr)

    def get_chat_template(self) -> Optional[str]:
        # Using raw metadata access
        buf = ctypes.create_string_buffer(2048)
        res = llama_cpp.llama_model_meta_val_str(self.model_ptr, b"tokenizer.chat_template", buf, 2048)
        if res > 0: return buf.value.decode('utf-8')
        return None

    def get_n_embd(self) -> int:
        return llama_cpp.llama_n_embd(self.model_ptr)

    def get_embeddings(self, pool_name: str, sid: int) -> List[float]:
        embd_ptr = llama_cpp.llama_get_embeddings_seq(self.ctx_ptrs[pool_name], sid)
        if not embd_ptr: return []
        n_embd = self.get_n_embd()
        return [float(embd_ptr[i]) for i in range(n_embd)]

    def __del__(self):
        """Final manual cleanup of raw pointers."""
        try:
            for ptr in self.ctx_ptrs.values():
                llama_cpp.llama_free(ptr)
            if hasattr(self, 'model_ptr'):
                llama_cpp.llama_free_model(self.model_ptr)
            llama_cpp.llama_backend_free()
        except Exception: pass
