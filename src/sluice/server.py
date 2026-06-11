import ctypes
import asyncio
import os
import time
import uuid
import json
import re
from typing import Optional, List, Dict, Any, Generator, Union
from fastapi import FastAPI, HTTPException, Body, Header, Path, Response, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import llama_cpp
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .bank import TokenBank
from .engine import SluiceEngine

# Configuration from Environment
MODEL_PATH = os.getenv("SLUICE_MODEL_PATH", "/models/gguf/model.gguf")
BASE_POOL = int(os.getenv("SLUICE_BASE_POOL", "98304"))
RESERVED_POOL = int(os.getenv("SLUICE_RESERVED_POOL", "32768"))
LARGE_THRESHOLD = int(os.getenv("SLUICE_LARGE_THRESHOLD", "16384"))
SCAVENGE_HOOK = os.getenv("SLUICE_SCAVENGE_HOOK")
RECOVERY_HOOK = os.getenv("SLUICE_RECOVERY_HOOK")
API_KEY = os.getenv("SLUICE_API_KEY")
AUTO_ELASTICITY = os.getenv("SLUICE_AUTO_ELASTICITY", "false").lower() == "true"
ELASTICITY_INTERVAL = float(os.getenv("SLUICE_ELASTICITY_INTERVAL", "5.0"))
PREFIX_CACHE_LIMIT = int(os.getenv("SLUICE_PREFIX_CACHE_LIMIT", "10"))
PORT = int(os.getenv("SLUICE_PORT", "8001"))

app = FastAPI(title="Llama-CPP Sluice: Dynamic Asymmetric Inference Server")
security = HTTPBearer(auto_error=False)

# --- Authentication ---

async def verify_auth(auth: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not API_KEY: return
    if auth is None or auth.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# --- Prometheus Metrics ---

METRIC_INF_LATENCY = Histogram("sluice_latency_seconds", "Latency", ["model", "type"])
METRIC_TOKENS_TOTAL = Counter("sluice_tokens_total", "Tokens", ["model"])
METRIC_POOL_USED = Gauge("sluice_vram_used", "Used tokens")
METRIC_POOL_TOTAL = Gauge("sluice_vram_total", "Total tokens")
METRIC_CACHE_HITS = Counter("sluice_prefix_cache_hits", "Cache hits")

# --- Radix Cache State ---

class PrefixCache:
    def __init__(self, limit: int):
        self.limit = limit
        self.cache: Dict[str, Dict[str, Any]] = {} # prefix_hash -> {sid, tokens, last_used}

    def get(self, prompt_tokens: List[int]) -> Optional[Dict[str, Any]]:
        # Find longest matching prefix
        # For simplicity, we just hash the first 512 tokens as a 'system prompt' candidate
        if len(prompt_tokens) < 128: return None
        prefix = tuple(prompt_tokens[:512])
        h = hash(prefix)
        if h in self.cache:
            METRIC_CACHE_HITS.inc()
            self.cache[h]["last_used"] = time.time()
            return self.cache[h]
        return None

    def put(self, tokens: List[int], sid: int):
        if len(tokens) < 128: return
        prefix = tuple(tokens[:512])
        h = hash(prefix)
        if len(self.cache) >= self.limit:
            # Evict LRU
            oldest = min(self.cache.keys(), key=lambda k: self.cache[k]["last_used"])
            asyncio.create_task(BANK.evict(self.cache[oldest]["sid"]))
            ENGINE.remove_sequence(self.cache[oldest]["sid"])
            del self.cache[oldest]
        
        self.cache[h] = {"sid": sid, "len": len(prefix), "last_used": time.time()}

CACHE = PrefixCache(PREFIX_CACHE_LIMIT)

# Global State
ENGINE: Optional[SluiceEngine] = None
BANK: TokenBank = TokenBank(BASE_POOL, RESERVED_POOL, LARGE_THRESHOLD, SCAVENGE_HOOK, RECOVERY_HOOK)

# --- Elasticity Coordinator ---

async def elasticity_loop():
    while True:
        await asyncio.sleep(ELASTICITY_INTERVAL)
        stats = BANK.get_stats()
        METRIC_POOL_USED.set(stats["used"])
        METRIC_POOL_TOTAL.set(stats["total"])
        if AUTO_ELASTICITY and BANK.is_expanded and BANK.used == 0 and BANK.waiting_large == 0:
            if RECOVERY_HOOK: await BANK._run_hook(RECOVERY_HOOK, "Recovery")
            ENGINE.hot_swap_context(BASE_POOL)
            await BANK.update_total(BASE_POOL, expanded=False)

# --- OpenAI Models ---

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatCompletionRequest(BaseModel):
    model: str = "sluice-model"
    messages: List[ChatMessage]
    tools: Optional[List[Dict[str, Any]]] = None
    max_tokens: Optional[int] = 128
    stream: bool = False
    required_ctx: Optional[int] = None
    cache_prompt: bool = True

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
    required_ctx: Optional[int] = None

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]

# --- Startup ---

@app.on_event("startup")
async def startup():
    global ENGINE
    ENGINE = SluiceEngine(MODEL_PATH, BASE_POOL)
    METRIC_POOL_TOTAL.set(BASE_POOL)
    asyncio.create_task(elasticity_loop())

# --- Admin Routes ---

@app.get("/metrics", dependencies=[Depends(verify_auth)])
async def metrics(): return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/v1/admin/resize", dependencies=[Depends(verify_auth)])
async def admin_resize(new_size: int = Body(..., embed=True)):
    await BANK.drain()
    ENGINE.hot_swap_context(new_size)
    await BANK.update_total(new_size, expanded=(new_size > BASE_POOL))
    await BANK.resume()
    return {"status": "resized", "new_size": new_size}

# --- Inference Core ---

def get_tokens(prompt: str) -> List[int]:
    m_ptr = ENGINE.get_model_ptr()
    p_bytes = prompt.encode('utf-8')
    tokens = (llama_cpp.llama_token * (len(p_bytes) + 1))()
    n = llama_cpp.llama_tokenize(m_ptr, p_bytes, len(p_bytes), tokens, len(tokens), True, True)
    return [tokens[i] for i in range(n)]

def format_prompt(messages: List[ChatMessage], tools: Optional[List[Dict[str, Any]]] = None) -> str:
    t_str = ENGINE.get_chat_template()
    if t_str:
        try:
            from jinja2 import Template
            return Template(t_str).render(messages=[m.dict(exclude_none=True) for m in messages], tools=tools, add_generation_prompt=True)
        except Exception: pass
    p = ""
    for m in messages: p += f"{m.role}: {m.content}\n"
    return p + "assistant: "

def low_level_generate(sid: int, tokens: List[int], max_tokens: int, cache_hit: Optional[Dict[str, Any]] = None):
    model_ptr = ENGINE.get_model_ptr()
    ctx_ptr = ENGINE.get_context_ptr()
    
    n_tokens = len(tokens)
    start_pos = 0
    if cache_hit:
        start_pos = cache_hit["len"]
        ENGINE.clone_sequence(cache_hit["sid"], sid, start_pos)
        print(f"[CACHE] Cloned {start_pos} tokens from seq {cache_hit['sid']} to {sid}")

    batch = llama_cpp.llama_batch_init(max(n_tokens, 512), 0, 1)
    try:
        batch.n_tokens = n_tokens - start_pos
        for i in range(batch.n_tokens):
            batch.token[i], batch.pos[i], batch.n_seq_id[i], batch.seq_id[i][0], batch.logits[i] = tokens[start_pos+i], start_pos+i, 1, sid, (i == batch.n_tokens - 1)
        
        if batch.n_tokens > 0:
            if llama_cpp.llama_decode(ctx_ptr, batch) != 0: raise RuntimeError("Decode fail")

        output, n_cur = [], n_tokens
        for _ in range(max_tokens):
            logits = llama_cpp.llama_get_logits_ith(ctx_ptr, batch.n_tokens - 1)
            candidates = (llama_cpp.llama_token_data * llama_cpp.llama_n_vocab(model_ptr))()
            for i in range(len(candidates)): candidates[i] = llama_cpp.llama_token_data(id=i, logit=logits[i], p=0.0)
            candidates_p = llama_cpp.llama_token_data_array(data=candidates, size=len(candidates), sorted=False)
            ntid = llama_cpp.llama_sample_token_greedy(ctx_ptr, ctypes.byref(candidates_p))
            if ntid == llama_cpp.llama_token_eos(model_ptr): break
            buf = ctypes.create_string_buffer(32)
            nb = llama_cpp.llama_token_to_piece(model_ptr, ntid, buf, 32, 0, False)
            output.append(buf[:nb].decode('utf-8', errors='ignore'))
            batch.n_tokens, batch.token[0], batch.pos[0], batch.logits[0] = 1, ntid, n_cur, True
            if llama_cpp.llama_decode(ctx_ptr, batch) != 0: break
            n_cur += 1
        return "".join(output), n_tokens, (n_cur - n_tokens)
    finally: llama_cpp.llama_batch_free(batch)

# --- Routes ---

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, dependencies=[Depends(verify_auth)])
async def chat_completions(request: ChatCompletionRequest):
    prompt = format_prompt(request.messages, request.tools)
    tokens = get_tokens(prompt)
    
    # 1. Check Radix Cache
    hit = CACHE.get(tokens)
    
    # 2. Bank Acquisition
    ctx_needed = request.required_ctx or max(2048, int(len(tokens) * 1.2))
    try: sid = await BANK.acquire(ctx_needed)
    except Exception as e: raise HTTPException(status_code=503, detail=str(e))

    try:
        loop = asyncio.get_event_loop()
        text, n_p, n_g = await loop.run_in_executor(None, low_level_generate, sid, tokens, request.max_tokens or 128, hit)
        
        # 3. Cache the result if requested (and it was a long system prompt)
        if request.cache_prompt and not hit and len(tokens) >= 128:
            CACHE.put(tokens, sid)
            # Re-release with 'pin=True' so BANK doesn't subtract the tokens from 'used'
            await BANK.release(sid, pin=True)
            # We don't remove_sequence yet
        else:
            ENGINE.remove_sequence(sid)
            await BANK.release(sid)

        return ChatCompletionResponse(id=f"sluice-{sid}", created=int(time.time()), model=request.model, choices=[{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], usage={"prompt_tokens": n_p, "completion_tokens": n_g, "total_tokens": n_p + n_g})
    except Exception as e:
        await BANK.release(sid)
        raise e

@app.post("/v1/embeddings", response_model=EmbeddingResponse, dependencies=[Depends(verify_auth)])
async def embeddings(request: EmbeddingRequest):
    inputs = [request.input] if isinstance(request.input, str) else request.input
    final_ctx = max(2048, int(sum(len(i) for i in inputs) * 1.5))
    sid = await BANK.acquire(final_ctx)
    try:
        data, loop = [], asyncio.get_event_loop()
        for idx, text in enumerate(inputs):
            tokens = get_tokens(text)
            def proc(s, tks):
                m_ptr, c_ptr = ENGINE.get_model_ptr(), ENGINE.get_context_ptr()
                batch = llama_cpp.llama_batch_init(len(tks), 0, 1)
                try:
                    batch.n_tokens = len(tks)
                    for i in range(len(tks)): batch.token[i], batch.pos[i], batch.n_seq_id[i], batch.seq_id[i][0], batch.logits[i] = tks[i], i, 1, s, False
                    llama_cpp.llama_decode(c_ptr, batch)
                    return ENGINE.get_embeddings(s), len(tks)
                finally:
                    llama_cpp.llama_batch_free(batch)
                    ENGINE.remove_sequence(s)
            vec, n = await loop.run_in_executor(None, proc, sid, tokens)
            data.append({"object": "embedding", "index": idx, "embedding": vec})
        return EmbeddingResponse(model=request.model, data=data, usage={"total_tokens": 0})
    finally: await BANK.release(sid)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
