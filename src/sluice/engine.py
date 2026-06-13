import llama_cpp
import os
import json
import ctypes
from typing import Optional, List, Dict
from .pools import PoolConfig

class SluiceEngine:
    def __init__(self, model_path: str, pool: PoolConfig, 
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
        
        print(f"[SLUICE] Initializing Unified Engine: {model_path}")
        self.model_path = model_path
        self.n_batch = n_batch
        self.n_ubatch = n_ubatch
        self.flash_attn = flash_attn
        self.embeddings = embeddings
        self.n_threads = n_threads or os.cpu_count() or 4
        self.n_threads_batch = n_threads_batch or os.cpu_count() or 4
        
        llama_cpp.llama_backend_init()
        
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
        
        self.ctx_ptr = self._create_raw_context(pool)
        
    def _create_raw_context(self, config: PoolConfig):
        cparams = llama_cpp.llama_context_default_params()
        cparams.n_ctx = config.max_tokens
        cparams.n_batch = self.n_batch
        cparams.n_ubatch = self.n_ubatch
        cparams.n_seq_max = 16 # Support up to 16 concurrent sequences
        cparams.n_threads = self.n_threads
        cparams.n_threads_batch = self.n_threads_batch
        cparams.type_k = config.type_k
        cparams.type_v = config.type_v
        cparams.flash_attn = self.flash_attn
        cparams.embeddings = self.embeddings
        
        ctx = llama_cpp.llama_new_context_with_model(self.model_ptr, cparams)
        if not ctx:
            raise RuntimeError(f"Failed to create context")
        return ctx

    def get_context_ptr(self):
        return self.ctx_ptr

    def get_model_ptr(self):
        return self.model_ptr

    def get_metadata(self) -> Dict[str, str]:
        """Expose model metadata for chat templates and other info."""
        metadata = {}
        # We can try to extract some common metadata if available
        # llama_cpp-python's high-level Llama class does this by iterating
        # but we'll provide a basic implementation or just a way to access it.
        return metadata

    def tokenize(self, text: str, add_bos: bool = True, special: bool = True) -> List[int]:
        p_bytes = text.encode('utf-8')
        n_tokens = len(p_bytes) + (1 if add_bos else 0)
        tokens = (llama_cpp.llama_token * n_tokens)()
        n = llama_cpp.llama_tokenize(self.model_ptr, p_bytes, len(p_bytes), tokens, n_tokens, add_bos, special)
        if n < 0:
            tokens = (llama_cpp.llama_token * abs(n))()
            n = llama_cpp.llama_tokenize(self.model_ptr, p_bytes, len(p_bytes), tokens, abs(n), add_bos, special)
        return [tokens[i] for i in range(n)]

    def detokenize(self, tokens: List[int]) -> str:
        output = ""
        for token in tokens:
            buf = ctypes.create_string_buffer(128)
            nb = llama_cpp.llama_token_to_piece(self.model_ptr, token, buf, 128, 0, False)
            output += buf[:nb].decode('utf-8', errors='ignore')
        return output

    def get_chat_template(self) -> Optional[str]:
        """Extract tokenizer.chat_template from model metadata."""
        buf = ctypes.create_string_buffer(8192)
        res = llama_cpp.llama_model_meta_val_str(
            self.model_ptr, b"tokenizer.chat_template", buf, 8192
        )
        if res >= 0:
            return buf.value.decode('utf-8', errors='ignore')
        return None

    def get_embeddings(self, sid: int) -> List[float]:
        """Extract embeddings for a specific sequence."""
        embd_ptr = llama_cpp.llama_get_embeddings_seq(self.ctx_ptr, sid)
        if not embd_ptr:
            return []
        
        n_embd = llama_cpp.llama_n_embd(self.model_ptr)
        return [float(embd_ptr[i]) for i in range(n_embd)]

    def clone_sequence(self, src_sid: int, dest_sid: int, length: int):
        """Zero-copy clone KV data from one sequence to another."""
        if hasattr(llama_cpp, "llama_kv_cache_seq_cp"):
            llama_cpp.llama_kv_cache_seq_cp(self.ctx_ptr, src_sid, dest_sid, 0, length)
        else:
            llama_cpp.llama_memory_seq_cp(self.ctx_ptr, src_sid, dest_sid, 0, length)

    def remove_sequence(self, sid: int):
        # Use llama_memory_seq_rm for older/stable bindings
        if hasattr(llama_cpp, "llama_kv_cache_seq_rm"):
            llama_cpp.llama_kv_cache_seq_rm(self.ctx_ptr, sid, -1, -1)
        else:
            llama_cpp.llama_memory_seq_rm(self.ctx_ptr, sid, -1, -1)

    def __del__(self):
        try:
            if hasattr(self, 'ctx_ptr'): llama_cpp.llama_free(self.ctx_ptr)
            if hasattr(self, 'model_ptr'): llama_cpp.llama_free_model(self.model_ptr)
            llama_cpp.llama_backend_free()
        except Exception: pass
