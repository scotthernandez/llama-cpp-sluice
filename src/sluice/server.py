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
import concurrent.futures
from typing import Optional, List, Dict, Any, Union, Callable, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Body, Header, Path, Response, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import llama_cpp
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .bank import TokenBank, BankSaturated
from .engine import SluiceEngine, RadixNode, RadixCache
from .pools import PoolConfig
from .middleware.trimmer import MiddleOutTrimmer

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
    parser.add_argument("--cache-size", type=int, default=int(os.getenv("SLUICE_CACHE_SIZE", "32")), help="Number of prefilled prefixes to keep in VRAM")

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

# Task 11/13: Module-level hardware mutex for C-layer safety
hardware_lock = threading.Lock()

# Task 11: Scaled ThreadPoolExecutor based on CPU cores
llm_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4, thread_name_prefix="llm-worker")

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

    def on_cache_evict(key, val):
        sid, _ = val
        print(f"[CACHE] Evicting prefix: {key[:32]}... (sid={sid})")
        async def do_evict():
            def c_cleanup():
                with engine_lock:
                    if engine: 
                        with hardware_lock:
                            engine.remove_sequence(sid)
            await loop.run_in_executor(llm_executor, c_cleanup)
            await BANK.dec_ref(sid)
        asyncio.run_coroutine_threadsafe(do_evict(), loop)

    RADIX_CACHE = RadixCache(max_size=ARGS.cache_size, on_evict=on_cache_evict)
    ts_list = [float(x.strip()) for x in ARGS.tensor_split.split(",")] if ARGS.tensor_split else None
    
    with engine_lock:
        engine = SluiceEngine(
            model_path=ARGS.model,
            pool=PoolConfig(name="main", max_tokens=ARGS.ctx_size),
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
        BANK = TokenBank(ARGS.ctx_size, RESERVED_POOL, LARGE_THRESHOLD, max_sequences=128, shutdown_event=SHUTDOWN_EVENT)
    
    def get_tokens_func(text): 
        with engine_lock:
            if not engine: return []
            return engine.tokenize(text, add_bos=True, special=True)
    
    def format_p_func(msgs, tools=None):
        with engine_lock:
            if not engine: return ""
            return format_prompt(msgs, engine, tools)
        
    TRIMMER = MiddleOutTrimmer(get_tokens_func=get_tokens_func, format_prompt_func=format_p_func)

@app.on_event("shutdown")
async def shutdown():
    global engine, BANK
    SHUTDOWN_EVENT.set()
    if BANK: await BANK.drain()
    
    # Task 14: Isolate Synchronous Executor Shutdown Track
    print("[SLUICE] Shutting down executor...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, llm_executor.shutdown, True)
    
    with engine_lock:
        if engine: del engine
        engine = BANK = None
    print("[SLUICE] Gateway offline.")

# --- Models ---

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

# --- Helpers ---

async def verify_auth(auth: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if ARGS.api_key and (not auth or auth.credentials != ARGS.api_key):
        raise HTTPException(status_code=401, detail="Invalid API Key")

def format_prompt(messages: List[ChatMessage], local_engine: SluiceEngine, tools: Optional[List[Dict[str, Any]]] = None) -> str:
    template = local_engine.get_chat_template()
    if template:
        try:
            from jinja2 import Template
            msgs_list = [m.model_dump(exclude_none=True) for m in messages]
            return Template(template).render(messages=msgs_list, tools=tools, add_generation_prompt=True)
        except Exception: pass
    p = ""
    for m in messages:
        if m.content: p += f"{m.role}: {m.content}\n"
    return p + "assistant: "

@asynccontextmanager
async def use_sid(bank: TokenBank, engine: SluiceEngine, needed: int):
    sid = await bank.acquire(needed)
    try:
        yield sid
    finally:
        if sid is not None:
            def cleanup(): 
                with hardware_lock:
                    engine.remove_sequence(sid)
            await asyncio.get_event_loop().run_in_executor(llm_executor, cleanup)
            await bank.dec_ref(sid)

@asynccontextmanager
async def use_cached_sid(bank: TokenBank, cache: Optional[RadixCache], key: Optional[str]):
    src_sid, prefix_tokens = None, []
    if key and cache:
        cached = cache.get(key)
        if cached:
            src_sid, prefix_tokens = cached
            await bank.inc_ref(src_sid)
    try:
        yield src_sid, prefix_tokens
    finally:
        if src_sid is not None:
            await bank.dec_ref(src_sid)

# --- Low-Level Inference ---

def apply_sampling(c_ptr, m_ptr, last_tokens: List[int], request: ChatCompletionRequest, idx: int = -1):
    import llama_cpp._internals as internals
    sampler = internals.LlamaSampler()
    if last_tokens:
        sampler.add_penalties(len(last_tokens), request.repeat_penalty, request.frequency_penalty, request.presence_penalty)
        for t in last_tokens[-64:]: sampler.accept(t)
    if request.temperature <= 0: sampler.add_greedy()
    else:
        sampler.add_top_k(request.top_k)
        sampler.add_top_p(request.top_p, 1)
        sampler.add_min_p(request.min_p, 1)
        sampler.add_temp(request.temperature)
        sampler.add_dist(request.seed or 42)
    with hardware_lock:
        return llama_cpp.llama_sampler_sample(sampler.sampler, c_ptr, idx)

def low_level_generate(local_engine: SluiceEngine, sid: int, tokens: List[int], max_tokens: int, budget: int, request: ChatCompletionRequest, src_sid: Optional[int] = None, prefix_len: int = 0):
    m_ptr, c_ptr = local_engine.get_model_ptr(), local_engine.get_context_ptr()
    batch = llama_cpp.llama_batch_init(ARGS.batch_size, 0, 1)
    try:
        start_idx = 0
        if src_sid is not None and prefix_len > 0:
            with hardware_lock:
                local_engine.clone_sequence(src_sid, sid, prefix_len)
            start_idx = prefix_len
        for i in range(start_idx, len(tokens), ARGS.batch_size):
            chunk = tokens[i:i + ARGS.batch_size]
            batch.n_tokens = len(chunk)
            for j in range(len(chunk)):
                idx = i + j
                batch.token[j], batch.pos[j], batch.n_seq_id[j], batch.seq_id[j][0], batch.logits[j] = chunk[j], idx, 1, sid, (idx == len(tokens) - 1)
            with hardware_lock:
                if llama_cpp.llama_decode(c_ptr, batch) != 0: raise RuntimeError("Decode fail")
        
        output, last_tokens, n_cur, reason = "", list(tokens), len(tokens), "stop"
        for i in range(max_tokens):
            if SHUTDOWN_EVENT.is_set(): reason = "shutdown"; break
            if (n_cur + 1) >= budget: reason = "length"; break
            ntid = apply_sampling(c_ptr, m_ptr, last_tokens, request)
            if ntid == llama_cpp.llama_token_eos(m_ptr): break
            buf = ctypes.create_string_buffer(128)
            nb = llama_cpp.llama_token_to_piece(m_ptr, ntid, buf, 128, 0, False)
            output += buf[:nb].decode('utf-8', errors='ignore')
            last_tokens.append(ntid)
            if request.stop and any(output.endswith(s) for s in ([request.stop] if isinstance(request.stop, str) else request.stop)): break
            batch.n_tokens, batch.token[0], batch.pos[0], batch.logits[0] = 1, ntid, n_cur, True
            with hardware_lock:
                if llama_cpp.llama_decode(c_ptr, batch) != 0: break
            n_cur += 1
        return output, len(tokens), (n_cur - len(tokens)), reason
    finally: llama_cpp.llama_batch_free(batch)

def low_level_stream_start(local_engine: SluiceEngine, sid: int, tokens: List[int], request: ChatCompletionRequest, src_sid: Optional[int] = None, prefix_len: int = 0):
    m_ptr, c_ptr = local_engine.get_model_ptr(), local_engine.get_context_ptr()
    batch = llama_cpp.llama_batch_init(ARGS.batch_size, 0, 1)
    try:
        start_idx = 0
        if src_sid is not None and prefix_len > 0:
            with hardware_lock:
                local_engine.clone_sequence(src_sid, sid, prefix_len)
            start_idx = prefix_len
        for i in range(start_idx, len(tokens), ARGS.batch_size):
            chunk = tokens[i:i + ARGS.batch_size]
            batch.n_tokens = len(chunk)
            for j in range(len(chunk)):
                idx = i + j
                batch.token[j], batch.pos[j], batch.n_seq_id[j], batch.seq_id[j][0], batch.logits[j] = chunk[j], idx, 1, sid, (idx == len(tokens) - 1)
            with hardware_lock:
                if llama_cpp.llama_decode(c_ptr, batch) != 0: raise RuntimeError("Prefill fail")
        ntid = apply_sampling(c_ptr, m_ptr, tokens, request, idx=-1)
    finally: llama_cpp.llama_batch_free(batch)
    return len(tokens), ntid

# Task 12: Upgrade Inference loops to Coarse-Grained Stream Bursting
def low_level_stream_burst(local_engine: SluiceEngine, sid: int, n_cur: int, last_tokens: List[int], request: ChatCompletionRequest, budget: int, prev_ntid: int):
    burst_pieces = []
    current_ntid = prev_ntid
    m_ptr, c_ptr = local_engine.get_model_ptr(), local_engine.get_context_ptr()
    batch = llama_cpp.llama_batch_init(1, 0, 1)
    try:
        for _ in range(16): # Burst size 16
            if SHUTDOWN_EVENT.is_set(): return burst_pieces, current_ntid, "shutdown"
            if (n_cur + 1) >= budget: return burst_pieces, current_ntid, "length"
            if current_ntid == llama_cpp.llama_token_eos(m_ptr): return burst_pieces, current_ntid, "stop"
            
            batch.n_tokens = 1
            batch.token[0], batch.pos[0], batch.n_seq_id[0], batch.seq_id[0][0], batch.logits[0] = current_ntid, n_cur, 1, sid, True
            with hardware_lock:
                if llama_cpp.llama_decode(c_ptr, batch) != 0: return burst_pieces, current_ntid, "error"
            
            buf = ctypes.create_string_buffer(128)
            nb = llama_cpp.llama_token_to_piece(m_ptr, current_ntid, buf, 128, 0, False)
            burst_pieces.append(buf[:nb].decode('utf-8', errors='ignore'))
            
            last_tokens.append(current_ntid)
            n_cur += 1
            current_ntid = apply_sampling(c_ptr, m_ptr, last_tokens, request, idx=-1)
            
            if request.stop and any("".join(burst_pieces).endswith(s) for s in ([request.stop] if isinstance(request.stop, str) else request.stop)):
                return burst_pieces, current_ntid, "stop"
        return burst_pieces, current_ntid, None
    finally: llama_cpp.llama_batch_free(batch)

# --- Routes ---

@app.post("/v1/chat/completions", dependencies=[Depends(verify_auth)])
async def chat_completions(request: ChatCompletionRequest):
    with engine_lock: l_eng, l_bank, l_cache, l_trim = engine, BANK, RADIX_CACHE, TRIMMER
    if not l_eng or not l_bank: raise HTTPException(status_code=503, detail="Engine not initialized")

    rid = f"sluice-{uuid.uuid4().hex[:8]}"
    cache_key = request.messages[0].content if (request.messages and request.messages[0].role == "system") else None
    
    async with use_cached_sid(l_bank, l_cache, cache_key) as (src_sid, prefix_tokens):
        prompt = format_prompt(request.messages, l_eng, request.tools)
        tokens = l_eng.tokenize(prompt, add_bos=True, special=True)
        needed = len(tokens) + request.max_tokens
        
        if ARGS.trimming and needed > l_bank.get_available_for_large():
            active = l_trim.trim(request.messages, l_bank.get_available_for_large() - request.max_tokens, tools=request.tools)
            tokens = l_eng.tokenize(format_prompt(active, l_eng, request.tools), add_bos=True, special=True)
            needed = len(tokens) + request.max_tokens

        async with use_sid(l_bank, l_eng, needed) as sid:
            actual_prefix_len = len(prefix_tokens) if (src_sid is not None and tokens[:len(prefix_tokens)] == prefix_tokens) else 0
            if actual_prefix_len == 0 and src_sid is not None:
                await l_bank.dec_ref(src_sid)
                src_sid = None

            if request.stream:
                async def stream_gen():
                    try:
                        loop = asyncio.get_event_loop()
                        n_cur, prev_ntid = await loop.run_in_executor(llm_executor, low_level_stream_start, l_eng, sid, tokens, request, src_sid, actual_prefix_len)
                        
                        if cache_key and src_sid is None and l_cache:
                            sys_tks = l_eng.tokenize(format_prompt(request.messages[:1], l_eng), add_bos=True, special=True)
                            if sys_tks:
                                async with use_sid(l_bank, l_eng, len(sys_tks)) as c_sid:
                                    def clone():
                                        with hardware_lock: l_eng.clone_sequence(sid, c_sid, len(sys_tks))
                                    await loop.run_in_executor(llm_executor, clone)
                                    l_cache.put(cache_key, (c_sid, sys_tks))
                                    await l_bank.inc_ref(c_sid)

                        last_tokens = list(tokens)
                        while True:
                            res = await loop.run_in_executor(llm_executor, low_level_stream_burst, l_eng, sid, n_cur, last_tokens, request, needed, prev_ntid)
                            pieces, next_ntid, finish = res
                            for p in pieces:
                                yield f"data: {json.dumps({'id': rid, 'choices': [{'index': 0, 'delta': {'content': p}, 'finish_reason': None}], 'model': ARGS.alias})}\n\n"
                                n_cur += 1
                            prev_ntid = next_ntid
                            if finish:
                                yield f"data: {json.dumps({'id': rid, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}], 'model': ARGS.alias})}\n\n"
                                break
                        yield "data: [DONE]\n\n"
                    except Exception as e: yield f"data: {json.dumps({'error': str(e)})}\n\n"
                return StreamingResponse(stream_gen(), media_type="text/event-stream")

            res = await asyncio.get_event_loop().run_in_executor(llm_executor, low_level_generate, l_eng, sid, tokens, request.max_tokens, needed, request, src_sid, actual_prefix_len)
            text, n_p, n_g, f_r = res
            if cache_key and src_sid is None and l_cache:
                sys_tks = l_eng.tokenize(format_prompt(request.messages[:1], l_eng), add_bos=True, special=True)
                if sys_tks:
                    async with use_sid(l_bank, l_eng, len(sys_tks)) as c_sid:
                        def clone():
                            with hardware_lock: l_eng.clone_sequence(sid, c_sid, len(sys_tks))
                        await asyncio.get_event_loop().run_in_executor(llm_executor, clone)
                        l_cache.put(cache_key, (c_sid, sys_tks))
                        await l_bank.inc_ref(c_sid)
            return {"id": rid, "object": "chat.completion", "created": int(time.time()), "model": ARGS.alias, "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": f_r}], "usage": {"prompt_tokens": n_p, "completion_tokens": n_g, "total_tokens": n_p + n_g}}

@app.post("/v1/embeddings", dependencies=[Depends(verify_auth)])
async def embeddings(request: EmbeddingRequest):
    with engine_lock: l_eng, l_bank = engine, BANK
    if not l_eng or not l_bank: raise HTTPException(status_code=503, detail="Engine not initialized")
    inputs = [request.input] if isinstance(request.input, str) else request.input
    data = []
    for idx, text in enumerate(inputs):
        tokens = l_eng.tokenize(text, add_bos=True, special=True)
        async with use_sid(l_bank, l_eng, len(tokens)) as sid:
            def decode():
                m_ptr, c_ptr = l_eng.get_model_ptr(), l_eng.get_context_ptr()
                batch = llama_cpp.llama_batch_init(len(tokens), 0, 1)
                try:
                    batch.n_tokens = len(tokens)
                    for i in range(len(tokens)): batch.token[i], batch.pos[i], batch.n_seq_id[i], batch.seq_id[i][0], batch.logits[i] = tokens[i], i, 1, sid, False
                    with hardware_lock:
                        if llama_cpp.llama_decode(c_ptr, batch) != 0: raise RuntimeError("Embedding decode fail")
                        return l_eng.get_embeddings(sid)
                finally: llama_cpp.llama_batch_free(batch)
            vec = await asyncio.get_event_loop().run_in_executor(llm_executor, decode)
            data.append({"object": "embedding", "index": idx, "embedding": vec})
    return {"object": "list", "data": data, "model": "sluice-model", "usage": {"prompt_tokens": sum(len(l_eng.tokenize(i)) for i in inputs), "total_tokens": sum(len(l_eng.tokenize(i)) for i in inputs)}}

@app.post("/v1/admin/drain", dependencies=[Depends(verify_auth)])
async def admin_drain():
    with engine_lock: l_bank = BANK
    if l_bank:
        asyncio.create_task(l_bank.drain())
        return {"status": "draining"}
    raise HTTPException(status_code=503, detail="Bank not initialized")

@app.post("/v1/admin/resume", dependencies=[Depends(verify_auth)])
async def admin_resume():
    with engine_lock: l_bank = BANK
    if l_bank:
        await l_bank.resume()
        return {"status": "running"}
    raise HTTPException(status_code=503, detail="Bank not initialized")

@app.get("/v1/admin/stats", dependencies=[Depends(verify_auth)])
async def admin_stats():
    with engine_lock: l_bank = BANK
    if l_bank: return l_bank.get_stats()
    raise HTTPException(status_code=503, detail="Bank not initialized")

@app.get("/v1/models", dependencies=[Depends(verify_auth)])
async def list_models(): return {"object": "list", "data": [{"id": ARGS.alias, "object": "model", "created": int(time.time()), "owned_by": "sluice"}]}

@app.get("/metrics", dependencies=[Depends(verify_auth)])
async def metrics(): return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

def main():
    import uvicorn
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)

if __name__ == "__main__":
    main()
