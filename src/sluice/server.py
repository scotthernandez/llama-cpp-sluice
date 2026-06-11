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
BASE_POOL = int(os.getenv("SLUICE_BASE_POOL", "0")) # 0 = Auto-discover
RESERVED_POOL = int(os.getenv("SLUICE_RESERVED_POOL", "32768"))
LARGE_THRESHOLD = int(os.getenv("SLUICE_LARGE_THRESHOLD", "16384"))
SCAVENGE_HOOK = os.getenv("SLUICE_SCAVENGE_HOOK")
RECOVERY_HOOK = os.getenv("SLUICE_RECOVERY_HOOK")
API_KEY = os.getenv("SLUICE_API_KEY")
AUTO_ELASTICITY = os.getenv("SLUICE_AUTO_ELASTICITY", "false").lower() == "true"
AUTO_DEFRAG = os.getenv("SLUICE_AUTO_DEFRAG", "true").lower() == "true"
DEFRAG_THRESHOLD = float(os.getenv("SLUICE_DEFRAG_THRESHOLD", "0.15"))
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
METRIC_FRAG_RATIO = Gauge("sluice_vram_frag_ratio", "Current KV cache fragmentation ratio")

# --- Radix Cache State ---

class PrefixCache:
    def __init__(self, limit: int):
        self.limit = limit
        self.cache: Dict[int, Dict[str, Any]] = {}

    def get(self, prompt_tokens: List[int]) -> Optional[Dict[str, Any]]:
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
            oldest = min(self.cache.keys(), key=lambda k: self.cache[k]["last_used"])
            asyncio.create_task(BANK.evict(self.cache[oldest]))
            ENGINE.remove_sequence(self.cache[oldest])
            del self.cache[oldest]
        self.cache[h] = {"sid": sid, "len": len(prefix), "last_used": time.time()}

CACHE = PrefixCache(PREFIX_CACHE_LIMIT)

# Global State
ENGINE: Optional[SluiceEngine] = None
BANK: Optional[TokenBank] = None

# --- Elasticity & Defrag Coordinator ---

async def maintenance_loop():
    """Background task to manage pool shrinking, recovery, and auto-defrag."""
    while True:
        await asyncio.sleep(ELASTICITY_INTERVAL)
        stats = BANK.get_stats()
        METRIC_POOL_USED.set(stats["used"])
        METRIC_POOL_TOTAL.set(stats["total"])
        
        frag_ratio = ENGINE.get_frag_ratio()
        METRIC_FRAG_RATIO.set(frag_ratio)

        # Auto-Defrag
        if AUTO_DEFRAG and frag_ratio > DEFRAG_THRESHOLD and stats["used"] > 0:
            print(f"[MAINTENANCE] Fragmentation ({frag_ratio:.2f}) exceeds threshold. Defragmenting...")
            ENGINE.defrag()

        # Elasticity Recovery
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
    temperature: float = 0.0
    stream: bool = False
    required_ctx: Optional[int] = None
    cache_prompt: bool = True
    response_format: Optional[Dict[str, Any]] = None

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
    global ENGINE, BANK, BASE_POOL
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        return

    # 1. Initialize Engine (Weights + Temp Context)
    ENGINE = SluiceEngine(MODEL_PATH, BASE_POOL or 2048)
    
    # 2. Auto-Discovery
    train_n_ctx = ENGINE.get_train_n_ctx()
    if BASE_POOL == 0:
        print(f"[ENGINE] Auto-discovering pool size: {train_n_ctx}")
        BASE_POOL = train_n_ctx
        ENGINE.hot_swap_context(BASE_POOL)
    elif BASE_POOL > train_n_ctx:
        print(f"[WARNING] BASE_POOL ({BASE_POOL}) exceeds model limit ({train_n_ctx}).")

    # 3. Initialize Bank
    BANK = TokenBank(BASE_POOL, RESERVED_POOL, LARGE_THRESHOLD, SCAVENGE_HOOK, RECOVERY_HOOK)
    
    METRIC_POOL_TOTAL.set(BASE_POOL)
    asyncio.create_task(maintenance_loop())

# --- Admin Routes ---

@app.get("/metrics", dependencies=[Depends(verify_auth)])
async def metrics(): return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/v1/admin/drain", dependencies=[Depends(verify_auth)])
async def admin_drain():
    asyncio.create_task(BANK.drain())
    return {"status": "draining"}

@app.post("/v1/admin/resume", dependencies=[Depends(verify_auth)])
async def admin_resume():
    await BANK.resume()
    return {"status": "running"}

@app.post("/v1/admin/defrag", dependencies=[Depends(verify_auth)])
async def admin_defrag():
    ENGINE.defrag()
    return {"status": "defrag_scheduled"}

@app.post("/v1/admin/resize", dependencies=[Depends(verify_auth)])
async def admin_resize(new_size: int = Body(..., embed=True)):
    await BANK.drain()
    ENGINE.hot_swap_context(new_size)
    await BANK.update_total(new_size, expanded=(new_size > BASE_POOL))
    await BANK.resume()
    return {"status": "resized", "new_size": new_size}

@app.get("/v1/admin/self-test", dependencies=[Depends(verify_auth)])
async def admin_self_test():
    import re
    prompts_path = os.path.join(os.path.dirname(__file__), "self_test_prompts.json")
    if not os.path.exists(prompts_path): raise HTTPException(status_code=404)
    with open(prompts_path, "r") as f: tests = json.load(f)
    results = []
    try: sid = await BANK.acquire(2048, timeout=30)
    except TimeoutError: return {"status": "error"}
    try:
        loop = asyncio.get_event_loop()
        for t in tests:
            t0 = time.time()
            text, _, _ = await loop.run_in_executor(None, low_level_generate, sid, get_tokens(t["prompt"]), 32)
            passed = bool(re.search(t["expected_regex"], text, re.IGNORECASE))
            results.append({"name": t["name"], "passed": passed, "latency_ms": (time.time()-t0)*1000})
    finally:
        ENGINE.remove_sequence(sid)
        await BANK.release(sid)
    return {"status": "complete", "results": results}

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

def low_level_generate(sid: int, tokens: List[int], max_tokens: int, hit: Optional[Dict[str, Any]] = None, grammar: Optional[Any] = None):
    m_ptr, c_ptr = ENGINE.get_model_ptr(), ENGINE.get_context_ptr()
    n_tokens, start_pos = len(tokens), 0
    if hit:
        start_pos = hit["len"]
        ENGINE.clone_sequence(hit["sid"], sid, start_pos)
    batch = llama_cpp.llama_batch_init(max(n_tokens, 512), 0, 1)
    try:
        batch.n_tokens = n_tokens - start_pos
        for i in range(batch.n_tokens): batch.token[i], batch.pos[i], batch.n_seq_id[i], batch.seq_id[i][0], batch.logits[i] = tokens[start_pos+i], start_pos+i, 1, sid, (i == batch.n_tokens - 1)
        if batch.n_tokens > 0 and llama_cpp.llama_decode(c_ptr, batch) != 0: raise RuntimeError("Decode fail")
        output, n_cur = [], n_tokens
        for _ in range(max_tokens):
            logits = llama_cpp.llama_get_logits_ith(c_ptr, batch.n_tokens - 1)
            candidates = (llama_cpp.llama_token_data * llama_cpp.llama_n_vocab(m_ptr))()
            for i in range(len(candidates)): candidates[i] = llama_cpp.llama_token_data(id=i, logit=logits[i], p=0.0)
            candidates_p = llama_cpp.llama_token_data_array(data=candidates, size=len(candidates), sorted=False)
            
            if grammar: llama_cpp.llama_sample_grammar(c_ptr, ctypes.byref(candidates_p), grammar)
            ntid = llama_cpp.llama_sample_token_greedy(c_ptr, ctypes.byref(candidates_p))
            if grammar: llama_cpp.llama_grammar_accept_token(c_ptr, grammar, ntid)
            
            if ntid == llama_cpp.llama_token_eos(m_ptr): break
            
            # Task 3: 128-byte buffer
            buf = ctypes.create_string_buffer(128)
            nb = llama_cpp.llama_token_to_piece(m_ptr, ntid, buf, 128, 0, False)
            output.append(buf[:nb].decode('utf-8', errors='ignore'))
            
            batch.n_tokens, batch.token[0], batch.pos[0], batch.logits[0] = 1, ntid, n_cur, True
            if llama_cpp.llama_decode(c_ptr, batch) != 0: break
            n_cur += 1
        return "".join(output), n_tokens, (n_cur - n_tokens)
    finally: llama_cpp.llama_batch_free(batch)

def low_level_stream_generator(sid: int, tokens: List[int], max_tokens: int, rid: str, model: str, hit: Optional[Dict[str, Any]] = None, grammar: Optional[Any] = None):
    m_ptr, c_ptr = ENGINE.get_model_ptr(), ENGINE.get_context_ptr()
    n_tokens, start_pos = len(tokens), 0
    if hit:
        start_pos = hit["len"]
        ENGINE.clone_sequence(hit["sid"], sid, start_pos)
    batch = llama_cpp.llama_batch_init(max(n_tokens, 512), 0, 1)
    try:
        batch.n_tokens = n_tokens - start_pos
        for i in range(batch.n_tokens): batch.token[i], batch.pos[i], batch.n_seq_id[i], batch.seq_id[i][0], batch.logits[i] = tokens[start_pos+i], start_pos+i, 1, sid, (i == batch.n_tokens - 1)
        if batch.n_tokens > 0 and llama_cpp.llama_decode(c_ptr, batch) != 0: return
        n_cur = n_tokens
        for _ in range(max_tokens):
            logits = llama_cpp.llama_get_logits_ith(c_ptr, batch.n_tokens - 1)
            candidates = (llama_cpp.llama_token_data * llama_cpp.llama_n_vocab(m_ptr))()
            for i in range(len(candidates)): candidates[i] = llama_cpp.llama_token_data(id=i, logit=logits[i], p=0.0)
            candidates_p = llama_cpp.llama_token_data_array(data=candidates, size=len(candidates), sorted=False)
            
            if grammar: llama_cpp.llama_sample_grammar(c_ptr, ctypes.byref(candidates_p), grammar)
            ntid = llama_cpp.llama_sample_token_greedy(c_ptr, ctypes.byref(candidates_p))
            if grammar: llama_cpp.llama_grammar_accept_token(c_ptr, grammar, ntid)
            
            if ntid == llama_cpp.llama_token_eos(m_ptr): break
            
            # Task 3: 128-byte buffer
            buf = ctypes.create_string_buffer(128)
            nb = llama_cpp.llama_token_to_piece(m_ptr, ntid, buf, 128, 0, False)
            
            yield f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'content': buf[:nb].decode('utf-8', errors='ignore')}, 'finish_reason': None}]})}\n\n"
            METRIC_TOKENS_TOTAL.labels(model=model).inc()
            batch.n_tokens, batch.token[0], batch.pos[0], batch.logits[0] = 1, ntid, n_cur, True
            if llama_cpp.llama_decode(c_ptr, batch) != 0: break
            n_cur += 1
        yield f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"
    finally: llama_cpp.llama_batch_free(batch)

# --- Routes ---

# Task 4: Global execution mutex for hardware thread safety
execution_mutex = asyncio.Lock()

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, dependencies=[Depends(verify_auth)])
@app.post("/v1/ctx/{ctx_size}/chat/completions", response_model=ChatCompletionResponse, dependencies=[Depends(verify_auth)])
async def chat_completions(request: ChatCompletionRequest, ctx_size: Optional[int] = None, x_sluice_required_ctx: Optional[int] = Header(None)):
    final_ctx = ctx_size or x_sluice_required_ctx or request.required_ctx or 2048
    rid = f"sluice-{uuid.uuid4().hex[:8]}"
    prompt = format_prompt(request.messages, request.tools)
    tokens = get_tokens(prompt)
    
    grammar = None
    if request.response_format and request.response_format.get("type") == "json_object":
        schema = request.response_format.get("schema")
        if schema: grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(schema))
        else: grammar = llama_cpp.LlamaGrammar.from_string('root ::= object\n...')
        
    hit = CACHE.get(tokens)
    try: sid = await BANK.acquire(final_ctx)
    except TimeoutError as e:
        if AUTO_ELASTICITY and not BANK.is_expanded:
            async with execution_mutex: # Task 4: Protect hot-swap
                ENGINE.hot_swap_context(final_ctx)
                await BANK.update_total(final_ctx, expanded=True)
            sid = await BANK.acquire(final_ctx, timeout=10.0)
        else: raise HTTPException(status_code=503, detail=str(e))
    
    if request.stream:
        async def stream_wrapper():
            try:
                # Task 4/5: Lock the hardware turn and ensure safe release
                async with execution_mutex:
                    for chunk in low_level_stream_generator(sid, tokens, request.max_tokens or 128, rid, request.model, hit, grammar):
                        yield chunk
                        await asyncio.sleep(0)
            except Exception as e:
                print(f"[Critical Engine Failure] Streaming panic: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally: await BANK.release(sid)
        return StreamingResponse(stream_wrapper(), media_type="text/event-stream")
    
    try:
        t0 = time.time()
        # Task 4: Robust lock and exception handling
        async with execution_mutex:
            try:
                loop = asyncio.get_event_loop()
                text, n_p, n_g = await loop.run_in_executor(None, low_level_generate, sid, tokens, request.max_tokens or 128, hit, grammar)
            except Exception as core_err:
                print(f"[Critical Engine Failure] Hardware thread panicked: {core_err}")
                raise RuntimeError(f"Internal hardware execution error: {core_err}")
                
        METRIC_INF_LATENCY.labels(model=request.model, type="non-stream").observe(time.time() - t0)
        METRIC_TOKENS_TOTAL.labels(model=request.model).inc(n_g)
        
        if request.cache_prompt and not hit and len(tokens) >= 128:
            CACHE.put(tokens, sid)
            await BANK.release(sid, pin=True)
        else:
            ENGINE.remove_sequence(sid)
            await BANK.release(sid)
            
        return ChatCompletionResponse(id=rid, created=int(time.time()), model=request.model, choices=[{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], usage={"prompt_tokens": n_p, "completion_tokens": n_g, "total_tokens": n_p + n_g})
    except Exception as e:
        await BANK.release(sid)
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/embeddings", response_model=EmbeddingResponse, dependencies=[Depends(verify_auth)])
async def embeddings(request: EmbeddingRequest):
    inputs = [request.input] if isinstance(request.input, str) else request.input
    final_ctx = request.required_ctx or max(2048, int(sum(len(i) for i in inputs) * 1.5))
    sid = await BANK.acquire(final_ctx)
    try:
        data, total_p, loop = [], 0, asyncio.get_event_loop()
        # Task 4: Global lock for embeddings batch
        async with execution_mutex:
            for idx, text in enumerate(inputs):
                def proc(s, tks):
                    c_ptr = ENGINE.get_context_ptr()
                    batch = llama_cpp.llama_batch_init(len(tks), 0, 1)
                    try:
                        for i in range(len(tks)): batch.token[i], batch.pos[i], batch.n_seq_id[i], batch.seq_id[i][0], batch.logits[i] = tks[i], i, 1, s, False
                        batch.n_tokens = len(tks)
                        if llama_cpp.llama_decode(c_ptr, batch) != 0: raise RuntimeError("Decode fail")
                        return ENGINE.get_embeddings(s), len(tks)
                    finally:
                        llama_cpp.llama_batch_free(batch)
                        ENGINE.remove_sequence(s)
                vec, n = await loop.run_in_executor(None, proc, sid, get_tokens(text))
                data.append({"object": "embedding", "index": idx, "embedding": vec})
                total_p += n
        return EmbeddingResponse(model=request.model, data=data, usage={"prompt_tokens": total_p, "total_tokens": total_p})
    finally: await BANK.release(sid)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
