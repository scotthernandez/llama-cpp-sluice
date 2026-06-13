import os
import sys
import time
import uuid
import json
import argparse
import asyncio
import threading
import signal
import ctypes
from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Body, Header, Path, Response, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import llama_cpp
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .bank import TokenBank, BankSaturated
from .engine import SluiceEngine
from .pools import PoolConfig
from .middleware.trimmer import MiddleOutTrimmer
from .radix_cache import RadixCache

# --- CLI & Config ---
def parse_args():
    parser = argparse.ArgumentParser(description="Llama-CPP Sluice: Stable Unified Server")
    parser.add_argument("-m", "--model", type=str, default=os.getenv("SLUICE_MODEL_PATH", "/models/gguf/model.gguf"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SLUICE_PORT", "8001")))
    parser.add_argument("--host", type=str, default=os.getenv("SLUICE_HOST", "0.0.0.0"))
    parser.add_argument("--api-key", type=str, default=os.getenv("SLUICE_API_KEY"))
    parser.add_argument("-c", "--ctx-size", type=int, default=int(os.getenv("SLUICE_BASE_POOL", "2048")))
    parser.add_argument("-b", "--batch-size", type=int, default=int(os.getenv("SLUICE_BATCH_SIZE", "512")))
    parser.add_argument("-ub", "--ubatch-size", type=int, default=int(os.getenv("SLUICE_UBATCH_SIZE", "256")))
    parser.add_argument("-ts", "--tensor-split", type=str, default=os.getenv("SLUICE_TENSOR_SPLIT"))
    parser.add_argument("--alias", type=str, default=os.getenv("SLUICE_MODEL_ALIAS", "sluice-model"))
    parser.add_argument("-fa", "--flash-attn", action="store_true", default=os.getenv("SLUICE_FLASH_ATTN", "true").lower() == "true")
    parser.add_argument("-sm", "--split-mode", type=int, default=int(os.getenv("SLUICE_SPLIT_MODE", "2")))
    parser.add_argument("--mlock", action="store_true", default=os.getenv("SLUICE_USE_MLOCK", "false").lower() == "true")
    parser.add_argument("--no-mmap", action="store_false", dest="mmap", default=os.getenv("SLUICE_USE_MMAP", "true").lower() == "true")
    parser.add_argument("-t", "--threads", type=int, default=int(os.getenv("SLUICE_N_THREADS", str(os.cpu_count() or 4))))
    parser.add_argument("-tb", "--threads-batch", type=int, default=int(os.getenv("SLUICE_N_THREADS_BATCH", str(os.cpu_count() or 4))))
    parser.add_argument("--ssl-key-file", type=str, default=os.getenv("SLUICE_SSL_KEY_FILE"))
    parser.add_argument("--ssl-cert-file", type=str, default=os.getenv("SLUICE_SSL_CERT_FILE"))
    parser.add_argument("--reasoning-format", type=str, default=os.getenv("SLUICE_REASONING_FORMAT", "none"))

    # Sluice specific
    parser.add_argument("--reserved-pool", type=int, help="Tokens reserved for large requests (Defaults to 1/3 of ctx-size)")
    parser.add_argument("--large-threshold", type=int, help="Threshold for 'Large' request classification (Defaults to 1/2 of ctx-size)")
    parser.add_argument("--no-adaptive-trimming", action="store_false", dest="trimming", default=os.getenv("SLUICE_ENABLE_ADAPTIVE_TRIMMING", "true").lower() == "true")
    parser.add_argument("--cache-size", type=int, default=int(os.getenv("SLUICE_CACHE_SIZE", "16")), help="Number of prefilled prefixes to keep in VRAM")

    if "pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST"):
        return parser.parse_args([])
    return parser.parse_args()

ARGS = parse_args()

# --- Config Initialization ---
RESERVED_POOL = ARGS.reserved_pool if ARGS.reserved_pool is not None else (ARGS.ctx_size // 4)
LARGE_THRESHOLD = ARGS.large_threshold if ARGS.large_threshold is not None else (ARGS.ctx_size // 2)

from concurrent.futures import ThreadPoolExecutor

# --- Globals ---
engine: Optional[SluiceEngine] = None
BANK: Optional[TokenBank] = None
TRIMMER: Optional[MiddleOutTrimmer] = None
RADIX_CACHE: Optional[RadixCache] = None
SHUTDOWN_EVENT = threading.Event()

# Atomic lock for global engine/bank state changes (hot-reloads, cleanup)
engine_lock = threading.Lock()

# Hardware execution executor (Properly serializes C-level inference to a single background thread)
# Inference is serialized to a single worker.  Every code path that calls
# llama_decode / llama_sampler_sample acquires this lock before proceeding,
# so at most one sequence generates at any moment (prefill + full decode
# loop run atomically).
#
# Why serialisation is intentional:
#
#   * Prevents starvation — the TokenBank's anti-starvation bank reserves
#     VRAM for large requests; without serialisation a flood of tiny
#     requests could starve them.
#   * Eliminates latency spikes — concurrent multi-sequence decoding on the
#     same context causes KV-cache contention and non-deterministic decode
#     ordering, which spikes latency unpredictably.
#   * Avoids C-level crashes — the underlying llama.cpp C API does not
#     support true concurrent multi-sequence decoding on the same context
#     pointer.  Calling llama_decode from multiple OS threads can trigger
#     SIGSEGV or silent KV-cache corruption.
#
# Configuration note:
#
#   ``cparams.n_seq_max = 16`` (engine.py, line 125) advertises capacity for
#   up to 16 concurrent sequences, but the executor effectively limits live
#   inference to 1 sequence.  Users should not interpret 16 as "16 parallel
#   generations."  True concurrency would require a multi-engine /
#   multi-context deployment, which is not currently implemented.
llm_executor = ThreadPoolExecutor(max_workers=1)

def signal_handler(sig, frame):
    print(f"\n[SLUICE] Signal {sig} received. Starting graceful shutdown...")
    SHUTDOWN_EVENT.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

app = FastAPI(title="Llama-CPP Sluice: Unified Stable Gateway")
security = HTTPBearer(auto_error=False)

@app.on_event("startup")
async def startup():
    global engine, BANK, TRIMMER, RADIX_CACHE
    if not os.path.exists(ARGS.model):
        print(f"[ERROR] Model not found: {ARGS.model}")
        return
    
    loop = asyncio.get_running_loop()

    def on_cache_evict(key, sid):
        print(f"[CACHE] Evicting prefix: {key[:32]}... (sid={sid})")
        # 1. Free C-level memory (serialized via executor)
        def c_cleanup():
            # Use lock to check engine global
            with engine_lock:
                local_engine = engine
            if local_engine: local_engine.remove_sequence(sid)
        loop.run_in_executor(llm_executor, c_cleanup)
        # 2. Release token budget in the bank
        asyncio.run_coroutine_threadsafe(BANK.evict(sid), loop)

    RADIX_CACHE = RadixCache(max_size=ARGS.cache_size, on_evict=on_cache_evict)

    ts_list = [float(x.strip()) for x in ARGS.tensor_split.split(",")] if ARGS.tensor_split else None
    
    # Initialize the "Bare Metal" Engine
    print(f"[SLUICE] Initializing Sluice Engine: {ARGS.model}")
    pool_cfg = PoolConfig(name="main", max_tokens=ARGS.ctx_size)
    
    with engine_lock:
        engine = SluiceEngine(
            model_path=ARGS.model,
            pool=pool_cfg,
            mmproj_path=None,
            tensor_split=ts_list,
            n_batch=ARGS.batch_size,
            n_ubatch=ARGS.ubatch_size,
            n_gpu_layers=-1,
            split_mode=ARGS.split_mode,
            flash_attn=ARGS.flash_attn,
            embeddings=True,
            n_threads=ARGS.threads,
            n_threads_batch=ARGS.threads_batch,
            use_mlock=ARGS.mlock,
            use_mmap=ARGS.mmap
        )
        
        BANK = TokenBank(ARGS.ctx_size, RESERVED_POOL, LARGE_THRESHOLD, max_sequences=128)
    
    def get_tokens(text): 
        with engine_lock:
            local_engine = engine
        if not local_engine: return []
        return local_engine.tokenize(text, add_bos=True, special=True)
    
    def format_p(msgs, tools=None):
        p = ""
        for m in msgs:
            role = m['role'] if isinstance(m, dict) else m.role
            content = m['content'] if isinstance(m, dict) else m.content
            if content: p += f"{role}: {content}\n"
        return p + "assistant: "
        
    TRIMMER = MiddleOutTrimmer(get_tokens_func=get_tokens, format_prompt_func=format_p)

@app.on_event("shutdown")
async def shutdown():
    global engine, BANK
    SHUTDOWN_EVENT.set()
    print("[SLUICE] Shutting down gateway...")
    if BANK:
        await BANK.drain()
    llm_executor.shutdown(wait=True)
    with engine_lock:
        if engine:
            del engine
        engine = None
        BANK = None
    print("[SLUICE] Gateway offline.")

# --- OpenAI Compatibility ---

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatCompletionRequest(BaseModel):
    model: str = "sluice-model"
    messages: List[ChatMessage]
    tools: Optional[List[Dict[str, Any]]] = None
    max_tokens: int = 128
    temperature: float = 0.7
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    repeat_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    seed: Optional[int] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class EmbeddingRequest(BaseModel):
    model: str = "sluice-model"
    input: Union[str, List[str]]

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]

async def verify_auth(auth: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if ARGS.api_key and (not auth or auth.credentials != ARGS.api_key):
        raise HTTPException(status_code=401, detail="Invalid API Key")

# --- Inference Core ---

def get_tokens(prompt: str, local_engine: SluiceEngine) -> List[int]:
    return local_engine.tokenize(prompt, add_bos=True, special=True)

def format_prompt(messages: List[ChatMessage], local_engine: SluiceEngine, tools: Optional[List[Dict[str, Any]]] = None) -> str:
    template = local_engine.get_chat_template()
    if template:
        try:
            from jinja2 import Template
            msgs_list = [m.model_dump(exclude_none=True) for m in messages]
            return Template(template).render(messages=msgs_list, tools=tools, add_generation_prompt=True)
        except Exception as e:
            print(f"[ERROR] Chat template rendering failed: {e}")

    # Basic fallback
    p = ""
    for m in messages:
        if m.content: p += f"{m.role}: {m.content}\n"
    return p + "assistant: "

def apply_sampling(c_ptr, m_ptr, last_tokens: List[int], request: ChatCompletionRequest, grammar: Optional[Any] = None, idx: int = -1):
    import llama_cpp._internals as internals
    sampler = internals.LlamaSampler()
    
    if last_tokens:
        sampler.add_penalties(
            penalty_last_n=min(len(last_tokens), 64),
            penalty_repeat=request.repeat_penalty,
            penalty_freq=request.frequency_penalty,
            penalty_present=request.presence_penalty
        )
        for t in last_tokens[-64:]:
            sampler.accept(t)

    if request.temperature <= 0:
        sampler.add_greedy()
    else:
        sampler.add_top_k(request.top_k)
        sampler.add_top_p(request.top_p, 1)
        sampler.add_min_p(request.min_p, 1)
        sampler.add_temp(request.temperature)
        sampler.add_dist(request.seed if request.seed is not None else 42)
        
    return llama_cpp.llama_sampler_sample(sampler.sampler, c_ptr, idx)

def check_stop_sequences(text: str, stop: Optional[Union[str, List[str]]]) -> bool:
    if not stop: return False
    stops = [stop] if isinstance(stop, str) else stop
    for s in stops:
        if text.endswith(s): return True
    return False

# --- Admin Routes ---

@app.post("/v1/admin/drain", dependencies=[Depends(verify_auth)])
async def admin_drain():
    with engine_lock:
        local_bank = BANK
    if local_bank:
        asyncio.create_task(local_bank.drain())
        return {"status": "draining"}
    return {"status": "error", "message": "Bank not initialized"}

@app.post("/v1/admin/resume", dependencies=[Depends(verify_auth)])
async def admin_resume():
    with engine_lock:
        local_bank = BANK
    if local_bank:
        await local_bank.resume()
        return {"status": "running"}
    return {"status": "error", "message": "Bank not initialized"}

@app.get("/v1/admin/stats", dependencies=[Depends(verify_auth)])
async def admin_stats():
    with engine_lock:
        local_bank = BANK
    if local_bank:
        return local_bank.get_stats()
    return {"status": "error", "message": "Bank not initialized"}

@app.get("/v1/models", dependencies=[Depends(verify_auth)])
async def list_models():
    return {"object": "list", "data": [{"id": ARGS.alias, "object": "model", "created": int(time.time()), "owned_by": "sluice"}]}

def low_level_generate(local_engine: SluiceEngine, sid: int, tokens: List[int], max_tokens: int, budget: int, request: ChatCompletionRequest, src_sid: Optional[int] = None, prefix_len: int = 0):
    m_ptr, c_ptr = local_engine.get_model_ptr(), local_engine.get_context_ptr()
    n_tokens = len(tokens)
    batch = llama_cpp.llama_batch_init(ARGS.batch_size, 0, 1)
    try:
        start_idx = 0
        if src_sid is not None and prefix_len > 0:
            print(f"[CACHE] Cloning {prefix_len} tokens from sid {src_sid} to sid {sid}")
            local_engine.clone_sequence(src_sid, sid, prefix_len)
            start_idx = prefix_len

        # Decode prompt (skipping cached prefix)
        for i in range(start_idx, n_tokens, ARGS.batch_size):
            chunk = tokens[i:i + ARGS.batch_size]
            batch.n_tokens = len(chunk)
            for j in range(len(chunk)):
                idx = i + j
                batch.token[j] = chunk[j]
                batch.pos[j] = idx
                batch.n_seq_id[j] = 1
                batch.seq_id[j][0] = sid
                batch.logits[j] = (idx == n_tokens - 1)
            
            if llama_cpp.llama_decode(c_ptr, batch) != 0: 
                raise RuntimeError("Decode prefill fail")
        
        output_text = ""
        last_tokens = list(tokens)
        n_cur, finish_reason = n_tokens, "stop"
        
        for i in range(max_tokens):
            if SHUTDOWN_EVENT.is_set():
                finish_reason = "shutdown"
                break
            if (n_cur + 1) >= budget:
                finish_reason = "length"
                break
            
            ntid = apply_sampling(c_ptr, m_ptr, last_tokens, request, idx=-1)
            if ntid == llama_cpp.llama_token_eos(m_ptr): break
            
            buf = ctypes.create_string_buffer(128)
            nb = llama_cpp.llama_token_to_piece(m_ptr, ntid, buf, 128, 0, False)
            piece = buf[:nb].decode('utf-8', errors='ignore')
            output_text += piece
            last_tokens.append(ntid)
            
            if check_stop_sequences(output_text, request.stop):
                break
                
            batch.n_tokens, batch.token[0], batch.pos[0], batch.logits[0] = 1, ntid, n_cur, True
            if llama_cpp.llama_decode(c_ptr, batch) != 0: break
            n_cur += 1
            
        return output_text, n_tokens, (n_cur - n_tokens), finish_reason
    finally: llama_cpp.llama_batch_free(batch)

def low_level_stream_start(local_engine: SluiceEngine, sid: int, tokens: List[int], request: ChatCompletionRequest, src_sid: Optional[int] = None, prefix_len: int = 0):
    m_ptr, c_ptr = local_engine.get_model_ptr(), local_engine.get_context_ptr()
    n_tokens = len(tokens)
    batch = llama_cpp.llama_batch_init(ARGS.batch_size, 0, 1)
    try:
        start_idx = 0
        if src_sid is not None and prefix_len > 0:
            print(f"[CACHE] Cloning {prefix_len} tokens from sid {src_sid} to sid {sid}")
            local_engine.clone_sequence(src_sid, sid, prefix_len)
            start_idx = prefix_len

        # Decode prompt (skipping cached prefix)
        for i in range(start_idx, n_tokens, ARGS.batch_size):
            chunk = tokens[i:i + ARGS.batch_size]
            batch.n_tokens = len(chunk)
            for j in range(len(chunk)):
                idx = i + j
                batch.token[j] = chunk[j]
                batch.pos[j] = idx
                batch.n_seq_id[j] = 1
                batch.seq_id[j][0] = sid
                batch.logits[j] = (idx == n_tokens - 1)
            
            if llama_cpp.llama_decode(c_ptr, batch) != 0: 
                raise RuntimeError("Prefill fail")
        
        # Sample first token immediately
        ntid = apply_sampling(c_ptr, m_ptr, tokens, request, idx=-1)
    finally: llama_cpp.llama_batch_free(batch)
    return n_tokens, ntid

def low_level_stream_step(local_engine: SluiceEngine, sid: int, n_cur: int, last_tokens: List[int], request: ChatCompletionRequest, budget: int, prev_ntid: int):
    if SHUTDOWN_EVENT.is_set(): return None, None, "shutdown"
    m_ptr, c_ptr = local_engine.get_model_ptr(), local_engine.get_context_ptr()
    if (n_cur + 1) >= budget: return None, None, "length"
    
    if prev_ntid == llama_cpp.llama_token_eos(m_ptr): return None, prev_ntid, "stop"
    
    batch = llama_cpp.llama_batch_init(1, 0, 1)
    try:
        batch.n_tokens = 1
        batch.token[0], batch.pos[0], batch.n_seq_id[0], batch.seq_id[0][0], batch.logits[0] = prev_ntid, n_cur, 1, sid, True
        if llama_cpp.llama_decode(c_ptr, batch) != 0: return None, prev_ntid, "error"
    finally: llama_cpp.llama_batch_free(batch)

    ntid = apply_sampling(c_ptr, m_ptr, last_tokens + [prev_ntid], request, idx=-1)
    
    buf = ctypes.create_string_buffer(128)
    nb = llama_cpp.llama_token_to_piece(m_ptr, prev_ntid, buf, 128, 0, False)
    piece = buf[:nb].decode('utf-8', errors='ignore')
    
    return piece, ntid, None

@app.post("/v1/chat/completions", dependencies=[Depends(verify_auth)])
async def chat_completions(request: ChatCompletionRequest):
    # Snapshot globals for request duration to ensure consistency
    with engine_lock:
        local_engine = engine
        local_bank = BANK
        local_trimmer = TRIMMER
        local_cache = RADIX_CACHE
    
    if not local_engine or not local_bank:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    rid = f"sluice-{uuid.uuid4().hex[:8]}"
    
    # 1. Identity prefix for caching (first system message)
    cache_key = None
    if request.messages and request.messages[0].role == "system":
        cache_key = request.messages[0].content

    src_sid = None
    prefix_tokens = []
    if cache_key and local_cache:
        cached_val = local_cache.get(cache_key)
        if cached_val:
            src_sid, prefix_tokens = cached_val
            print(f"[CACHE] Hit! prefix_len={len(prefix_tokens)}")

    prompt = format_prompt(request.messages, local_engine, request.tools)
    tokens = get_tokens(prompt, local_engine)
    
    prompt_len = len(tokens)
    needed = prompt_len + request.max_tokens
    
    active_messages = request.messages
    if ARGS.trimming and needed > local_bank.get_available_for_large():
        available = local_bank.get_available_for_large()
        if available >= 4096:
            active_messages = local_trimmer.trim(request.messages, available - request.max_tokens, tools=request.tools)
            prompt = format_prompt(active_messages, local_engine, request.tools)
            tokens = get_tokens(prompt, local_engine)
            needed = len(tokens) + request.max_tokens
        else:
            return Response(json.dumps({"error": {"message": f"VRAM Full. Need {needed}, avail {available}"}}), status_code=429, headers={"Retry-After": "5"})

    sid = None
    try:
        sid = await local_bank.acquire(needed)
        
        # If we had a cache hit, ensure the prefix tokens match the start of our current prompt
        actual_prefix_len = 0
        if src_sid is not None and tokens[:len(prefix_tokens)] == prefix_tokens:
            actual_prefix_len = len(prefix_tokens)
        else:
            src_sid = None # Miss or mismatch

        if request.stream:
            async def stream_gen():
                last_tokens = list(tokens)
                try:
                    loop = asyncio.get_event_loop()
                    
                    def start():
                        return low_level_stream_start(local_engine, sid, tokens, request, src_sid=src_sid, prefix_len=actual_prefix_len)
                            
                    n_cur, prev_ntid = await loop.run_in_executor(llm_executor, start)
                    
                    # 2. Populate cache on miss (using a cloned sequence to keep it "frozen")
                    if cache_key and src_sid is None and local_cache:
                        sys_prompt = format_prompt(request.messages[:1], local_engine)
                        sys_tokens = get_tokens(sys_prompt, local_engine)
                        if sys_tokens:
                            # We clone the prefilled state into a new "static" sequence for the cache
                            cache_sid = await local_bank.acquire(len(sys_tokens), timeout=5.0)
                            try:
                                def clone():
                                    local_engine.clone_sequence(sid, cache_sid, len(sys_tokens))
                                await loop.run_in_executor(llm_executor, clone)
                                local_cache.put(cache_key, (cache_sid, sys_tokens))
                                await local_bank.release(cache_sid, pin=True)
                                print(f"[CACHE] Prefilled prefix cached on sid {cache_sid}")
                            except Exception as ce:
                                print(f"[CACHE] Failed to clone for cache: {ce}")
                                await local_bank.release(cache_sid)

                    for _ in range(request.max_tokens):
                        def step():
                            return low_level_stream_step(local_engine, sid, n_cur, last_tokens, request, needed, prev_ntid)
                        
                        piece, ntid, finish = await loop.run_in_executor(llm_executor, step)
                        
                        if finish:
                            yield f"data: {json.dumps({'id': rid, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}], 'model': ARGS.alias})}\n\n"
                            break
                        
                        last_tokens.append(prev_ntid)
                        prev_ntid = ntid
                        yield f"data: {json.dumps({'id': rid, 'choices': [{'index': 0, 'delta': {'content': piece}, 'finish_reason': None}], 'model': ARGS.alias})}\n\n"
                        n_cur += 1
                    yield "data: [DONE]\n\n"
                except Exception as stream_e:
                    print(f"[STREAM ERROR] {stream_e}")
                    yield f"data: {json.dumps({'error': str(stream_e)})}\n\n"
                finally:
                    if sid is not None:
                        # Only cleanup and release if NOT in cache
                        is_cached = local_cache and local_cache.has_value(sid)
                        if not is_cached:
                            def cleanup():
                                local_engine.remove_sequence(sid)
                            await asyncio.get_event_loop().run_in_executor(llm_executor, cleanup)
                            await local_bank.release(sid)
                        else:
                            # Pin it in the bank
                            await local_bank.release(sid, pin=True)

            return StreamingResponse(stream_gen(), media_type="text/event-stream")

        loop = asyncio.get_event_loop()
        def run():
            return low_level_generate(local_engine, sid, tokens, request.max_tokens, needed, request, src_sid=src_sid, prefix_len=actual_prefix_len)

        text, n_p, n_g, f_r = await loop.run_in_executor(llm_executor, run)
        
        # 2. Populate cache on miss (non-streaming)
        if cache_key and src_sid is None and local_cache:
            sys_prompt = format_prompt(request.messages[:1], local_engine)
            sys_tokens = get_tokens(sys_prompt, local_engine)
            if sys_tokens:
                cache_sid = await local_bank.acquire(len(sys_tokens), timeout=5.0)
                try:
                    def clone():
                        local_engine.clone_sequence(sid, cache_sid, len(sys_tokens))
                    await loop.run_in_executor(llm_executor, clone)
                    local_cache.put(cache_key, (cache_sid, sys_tokens))
                    await local_bank.release(cache_sid, pin=True)
                    print(f"[CACHE] Prefilled prefix cached on sid {cache_sid}")
                except Exception as ce:
                    print(f"[CACHE] Failed to clone for cache: {ce}")
                    await local_bank.release(cache_sid)

        return {
            "id": rid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": ARGS.alias,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": f_r}],
            "usage": {"prompt_tokens": n_p, "completion_tokens": n_g, "total_tokens": n_p + n_g}
        }
    except Exception as e:
        print(f"[ERROR] Chat completions failed: {e}")
        if isinstance(e, BankSaturated):
            return Response(json.dumps({"error": {"message": str(e)}}), status_code=429, headers={"Retry-After": "5"})
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if sid is not None and not request.stream:
            is_cached = local_cache and local_cache.has_value(sid)
            if not is_cached:
                def cleanup():
                    local_engine.remove_sequence(sid)
                await asyncio.get_event_loop().run_in_executor(llm_executor, cleanup)
                await local_bank.release(sid)
            else:
                await local_bank.release(sid, pin=True)

@app.post("/v1/embeddings", dependencies=[Depends(verify_auth)])
async def embeddings(request: EmbeddingRequest):
    # Snapshot globals
    with engine_lock:
        local_engine = engine
        local_bank = BANK
    
    if not local_engine or not local_bank:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    inputs = [request.input] if isinstance(request.input, str) else request.input
    data = []
    total_p = 0
    
    for idx, text in enumerate(inputs):
        tokens = get_tokens(text, local_engine)
        total_p += len(tokens)
        sid = None
        try:
            sid = await local_bank.acquire(len(tokens))
            m_ptr, c_ptr = local_engine.get_model_ptr(), local_engine.get_context_ptr()
            batch = llama_cpp.llama_batch_init(len(tokens), 0, 1)
            try:
                batch.n_tokens = len(tokens)
                for i in range(len(tokens)):
                    batch.token[i], batch.pos[i], batch.n_seq_id[i], batch.seq_id[i][0], batch.logits[i] = tokens[i], i, 1, sid, False
                
                def decode():
                    if llama_cpp.llama_decode(c_ptr, batch) != 0:
                        raise RuntimeError("Embedding decode fail")
                    return local_engine.get_embeddings(sid)
                
                vec = await asyncio.get_event_loop().run_in_executor(llm_executor, decode)
                data.append({"object": "embedding", "index": idx, "embedding": vec})
            finally:
                llama_cpp.llama_batch_free(batch)
        finally:
            if sid is not None:
                def cleanup():
                    local_engine.remove_sequence(sid)
                await asyncio.get_event_loop().run_in_executor(llm_executor, cleanup)
                await local_bank.release(sid)
            
    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {"prompt_tokens": total_p, "total_tokens": total_p}
    }

@app.get("/metrics", dependencies=[Depends(verify_auth)])
async def metrics(): return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

def main():
    import uvicorn
    uvicorn_kwargs = {"app": app, "host": ARGS.host, "port": ARGS.port}
    if ARGS.ssl_cert_file and ARGS.ssl_key_file:
        if os.path.exists(ARGS.ssl_cert_file) and os.path.exists(ARGS.ssl_key_file):
            uvicorn_kwargs["ssl_keyfile"], uvicorn_kwargs["ssl_certfile"] = ARGS.ssl_key_file, ARGS.ssl_cert_file
    uvicorn.run(**uvicorn_kwargs)

if __name__ == "__main__":
    main()
