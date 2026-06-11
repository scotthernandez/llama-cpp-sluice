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

from .bank import TokenBank, BankSaturated
from .engine import SluiceEngine
from .pools import PoolConfig, DEFAULT_POOLS
from .middleware.trimmer import MiddleOutTrimmer

# --- Configuration (Hardware & Performance) ---
MODEL_PATH = os.getenv("SLUICE_MODEL_PATH", "/models/gguf/model.gguf")
MMPROJ_PATH = os.getenv("SLUICE_MMPROJ_PATH")
TENSOR_SPLIT = os.getenv("SLUICE_TENSOR_SPLIT")
BATCH_SIZE = int(os.getenv("SLUICE_BATCH_SIZE", "512"))
UBATCH_SIZE = int(os.getenv("SLUICE_UBATCH_SIZE", "256"))
GPU_LAYERS = int(os.getenv("SLUICE_GPU_LAYERS", "-1"))
FLASH_ATTN = os.getenv("SLUICE_FLASH_ATTN", "true").lower() == "true"
EMBEDDINGS = os.getenv("SLUICE_EMBEDDINGS", "true").lower() == "true"
SPLIT_MODE = int(os.getenv("SLUICE_SPLIT_MODE", str(llama_cpp.LLAMA_SPLIT_MODE_LAYER)))

N_THREADS = int(os.getenv("SLUICE_N_THREADS", str(os.cpu_count() or 4)))
N_THREADS_BATCH = int(os.getenv("SLUICE_N_THREADS_BATCH", str(os.cpu_count() or 4)))
USE_MLOCK = os.getenv("SLUICE_USE_MLOCK", "false").lower() == "true"
USE_MMAP = os.getenv("SLUICE_USE_MMAP", "true").lower() == "true"

# --- Configuration (Bank & Priority) ---
RESERVED_POOL = int(os.getenv("SLUICE_RESERVED_POOL", "32768"))
LARGE_THRESHOLD = int(os.getenv("SLUICE_LARGE_THRESHOLD", "16384"))
SCAVENGE_HOOK = os.getenv("SLUICE_SCAVENGE_HOOK")
RECOVERY_HOOK = os.getenv("SLUICE_RECOVERY_HOOK")
SCAVENGE_DELAY = float(os.getenv("SLUICE_SCAVENGE_DELAY", "15.0"))

# --- Configuration (Elasticity & Cache) ---
AUTO_ELASTICITY = os.getenv("SLUICE_AUTO_ELASTICITY", "false").lower() == "true"
ELASTICITY_TIMEOUT = float(os.getenv("SLUICE_ELASTICITY_TIMEOUT", "10.0"))
AUTO_DEFRAG = os.getenv("SLUICE_AUTO_DEFRAG", "true").lower() == "true"
DEFRAG_THRESHOLD = float(os.getenv("SLUICE_DEFRAG_THRESHOLD", "0.15"))
ELASTICITY_INTERVAL = float(os.getenv("SLUICE_ELASTICITY_INTERVAL", "5.0"))

CACHE_MIN_TOKENS = int(os.getenv("SLUICE_CACHE_MIN_TOKENS", "128"))
CACHE_PREFIX_LEN = int(os.getenv("SLUICE_CACHE_PREFIX_LEN", "512"))
PREFIX_CACHE_LIMIT = int(os.getenv("SLUICE_PREFIX_CACHE_LIMIT", "10"))

# --- Configuration (Negotiation & Defaults) ---
ENABLE_ADAPTIVE_TRIMMING = os.getenv("SLUICE_ENABLE_ADAPTIVE_TRIMMING", "true").lower() == "true"
TRIM_FLOOR = int(os.getenv("SLUICE_TRIM_FLOOR", "4096"))
DEFAULT_CTX = int(os.getenv("SLUICE_DEFAULT_CTX", "2048"))
DEFAULT_MAX_TOKENS = int(os.getenv("SLUICE_DEFAULT_MAX_TOKENS", "128"))
RETRY_AFTER = os.getenv("SLUICE_RETRY_AFTER", "5")
SELF_TEST_TIMEOUT = float(os.getenv("SLUICE_SELF_TEST_TIMEOUT", "30.0"))
REASONING_FORMAT = os.getenv("SLUICE_REASONING_FORMAT", "none")

API_KEY = os.getenv("SLUICE_API_KEY")
PORT = int(os.getenv("SLUICE_PORT", "8001"))
HOST = os.getenv("SLUICE_HOST", "0.0.0.0")

# Load Pools
POOLS_JSON = os.getenv("SLUICE_POOLS")
if POOLS_JSON: POOLS = [PoolConfig(**p) for p in json.loads(POOLS_JSON)]
else: POOLS = DEFAULT_POOLS
BASE_POOL = POOLS[0].max_tokens 

app = FastAPI(title="Llama-CPP Sluice: Fully Configurable Asymmetric Server")
security = HTTPBearer(auto_error=False)

# --- Authentication ---

async def verify_auth(auth: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not API_KEY: return
    if auth is None or auth.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# --- Prometheus Metrics ---

METRIC_INF_LATENCY = Histogram("sluice_latency", "Latency", ["pool", "type"])
METRIC_TOKENS_TOTAL = Counter("sluice_tokens_total", "Tokens", ["model"])
METRIC_POOL_USED = Gauge("sluice_pool_used", "Used tokens", ["pool"])
METRIC_POOL_TOTAL = Gauge("sluice_pool_total", "Total tokens", ["pool"])
METRIC_CACHE_HITS = Counter("sluice_prefix_cache_hits", "Cache hits")
METRIC_FRAG_RATIO = Gauge("sluice_vram_frag_ratio", "Frag ratio", ["pool"])

# --- Radix Cache ---

class PrefixCache:
    def __init__(self, limit: int):
        self.limit = limit
        self.cache: Dict[int, Dict[str, Any]] = {}
    def get(self, tokens: List[int]) -> Optional[Dict[str, Any]]:
        if len(tokens) < CACHE_MIN_TOKENS: return None
        h = hash(tuple(tokens[:CACHE_PREFIX_LEN]))
        return self.cache.get(h)
    def put(self, tokens: List[int], sid: int, pool: str):
        if len(tokens) < CACHE_MIN_TOKENS: return
        h = hash(tuple(tokens[:CACHE_PREFIX_LEN]))
        if len(self.cache) >= self.limit:
            oldest = min(self.cache.keys(), key=lambda k: self.cache[k]["last_used"])
            asyncio.create_task(BANK.evict(self.cache[oldest]["sid"]))
            ENGINE.remove_sequence(self.cache[oldest]["pool"], self.cache[oldest]["sid"])
            del self.cache[oldest]
        self.cache[h] = {"sid": sid, "pool": pool, "len": min(len(tokens), CACHE_PREFIX_LEN), "last_used": time.time()}

CACHE = PrefixCache(PREFIX_CACHE_LIMIT)
ENGINE: Optional[SluiceEngine] = None
BANK: Optional[TokenBank] = None
TRIMMER: Optional[MiddleOutTrimmer] = None

# --- Elasticity & Defrag Coordinator ---

async def maintenance_loop():
    while True:
        await asyncio.sleep(ELASTICITY_INTERVAL)
        stats = BANK.get_stats()
        for name in BANK.pool_names:
            METRIC_POOL_USED.labels(pool=name).set(stats["used"][name])
            METRIC_POOL_TOTAL.labels(pool=name).set(stats["total"][name])
            frag = ENGINE.get_frag_ratio(name)
            METRIC_FRAG_RATIO.labels(pool=name).set(frag)
            if AUTO_DEFRAG and frag > DEFRAG_THRESHOLD and stats["used"][name] > 0:
                ENGINE.defrag(name)

        if AUTO_ELASTICITY and BANK.is_expanded and sum(BANK.used.values()) == 0 and BANK.waiting_large == 0:
            if RECOVERY_HOOK: await BANK._run_hook(RECOVERY_HOOK, "Recovery")
            ENGINE.hot_swap_context(POOLS[0].name, POOLS[0])
            await BANK.update_capacity(POOLS[0].name, POOLS[0].max_tokens, expanded=False)

# --- OpenAI Models ---

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatCompletionRequest(BaseModel):
    model: str = "sluice-model"
    messages: List[ChatMessage]
    tools: Optional[List[Dict[str, Any]]] = None
    max_tokens: Optional[int] = Field(default_factory=lambda: DEFAULT_MAX_TOKENS)
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.05
    repeat_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
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
    global ENGINE, BANK, TRIMMER
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        return

    ts_list = None
    if TENSOR_SPLIT:
        try: ts_list = [float(x.strip()) for x in TENSOR_SPLIT.split(",")]
        except Exception: pass

    ENGINE = SluiceEngine(
        model_path=MODEL_PATH, 
        pools=POOLS,
        mmproj_path=MMPROJ_PATH,
        tensor_split=ts_list,
        n_batch=BATCH_SIZE,
        n_ubatch=UBATCH_SIZE,
        n_gpu_layers=GPU_LAYERS,
        split_mode=SPLIT_MODE,
        flash_attn=FLASH_ATTN,
        embeddings=EMBEDDINGS,
        n_threads=N_THREADS,
        n_threads_batch=N_THREADS_BATCH,
        use_mlock=USE_MLOCK,
        use_mmap=USE_MMAP
    )

    capacities = {p.name: p.max_tokens for p in POOLS}
    BANK = TokenBank(list(capacities.keys()), capacities, RESERVED_POOL, LARGE_THRESHOLD, SCAVENGE_HOOK, RECOVERY_HOOK, SCAVENGE_DELAY)
    TRIMMER = MiddleOutTrimmer(get_tokens_func=get_tokens, format_prompt_func=format_prompt)
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
async def admin_defrag(pool_name: str):
    ENGINE.defrag(pool_name)
    return {"status": "defrag_scheduled"}

@app.post("/v1/admin/resize", dependencies=[Depends(verify_auth)])
async def admin_resize(pool_name: str, new_size: int = Body(..., embed=True)):
    config = next((p for p in POOLS if p.name == pool_name), POOLS[0])
    new_config = config.copy(update={"max_tokens": new_size})
    await BANK.drain()
    ENGINE.hot_swap_context(pool_name, new_config)
    await BANK.update_capacity(pool_name, new_size, expanded=(new_size > config.max_tokens))
    await BANK.resume()
    return {"status": "resized", "pool": pool_name, "new_size": new_size}

@app.get("/v1/admin/self-test", dependencies=[Depends(verify_auth)])
async def admin_self_test():
    prompts_path = os.path.join(os.path.dirname(__file__), "self_test_prompts.json")
    if not os.path.exists(prompts_path): raise HTTPException(status_code=404)
    with open(prompts_path, "r") as f: tests = json.load(f)
    results = []
    pool_name = POOLS[0].name
    try: sid = await BANK.acquire(pool_name, DEFAULT_CTX)
    except Exception: return {"status": "error", "message": "Bank busy"}
    try:
        loop = asyncio.get_event_loop()
        for t in tests:
            t0 = time.time()
            text, _, _, _ = await loop.run_in_executor(None, low_level_generate, pool_name, sid, get_tokens(t["prompt"]), 32, None, None, 0, ChatCompletionRequest(messages=[]))
            passed = bool(re.search(t["expected_regex"], text, re.IGNORECASE))
            results.append({"name": t["name"], "passed": passed, "latency_ms": (time.time()-t0)*1000})
    finally:
        ENGINE.remove_sequence(pool_name, sid)
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
    for m in messages:
        if m.content: p += f"{m.role}: {m.content}\n"
    return p + "assistant: "

def apply_sampling(c_ptr, m_ptr, last_tokens: List[int], request: ChatCompletionRequest, grammar: Optional[Any] = None):
    # Construct candidates array
    logits = llama_cpp.llama_get_logits(c_ptr)
    n_vocab = llama_cpp.llama_n_vocab(m_ptr)
    candidates = (llama_cpp.llama_token_data * n_vocab)()
    for i in range(n_vocab):
        candidates[i] = llama_cpp.llama_token_data(id=i, logit=logits[i], p=0.0)
    candidates_p = llama_cpp.llama_token_data_array(data=candidates, size=n_vocab, sorted=False)
    
    # 1. Penalties
    if last_tokens:
        n_prev = min(len(last_tokens), 64)
        prev_array = (llama_cpp.llama_token * n_prev)(*last_tokens[-n_prev:])
        llama_cpp.llama_sample_repetition_penalty(
            c_ptr, ctypes.byref(candidates_p), prev_array, n_prev, ctypes.c_float(request.repeat_penalty)
        )
        llama_cpp.llama_sample_frequency_and_presence_penalties(
            c_ptr, ctypes.byref(candidates_p), prev_array, n_prev, 
            ctypes.c_float(request.frequency_penalty), ctypes.c_float(request.presence_penalty)
        )

    # 2. Grammar
    if grammar:
        llama_cpp.llama_sample_grammar(c_ptr, ctypes.byref(candidates_p), grammar)

    # 3. Sampling filters
    if request.temperature <= 0:
        ntid = llama_cpp.llama_sample_token_greedy(c_ptr, ctypes.byref(candidates_p))
    else:
        llama_cpp.llama_sample_top_k(c_ptr, ctypes.byref(candidates_p), ctypes.c_int(request.top_k), ctypes.c_size_t(1))
        llama_cpp.llama_sample_top_p(c_ptr, ctypes.byref(candidates_p), ctypes.c_float(request.top_p), ctypes.c_size_t(1))
        llama_cpp.llama_sample_min_p(c_ptr, ctypes.byref(candidates_p), ctypes.c_float(request.min_p), ctypes.c_size_t(1))
        llama_cpp.llama_sample_temp(c_ptr, ctypes.byref(candidates_p), ctypes.c_float(request.temperature))
        ntid = llama_cpp.llama_sample_token(c_ptr, ctypes.byref(candidates_p))
    
    if grammar:
        llama_cpp.llama_grammar_accept_token(c_ptr, grammar, ntid)
        
    return ntid

def check_stop_sequences(text: str, stop: Optional[Union[str, List[str]]]) -> bool:
    if not stop: return False
    stops = [stop] if isinstance(stop, str) else stop
    for s in stops:
        if text.endswith(s): return True
    return False

def low_level_generate(pool: str, sid: int, tokens: List[int], max_tokens: int, hit: Optional[Dict[str, Any]], grammar: Optional[Any], budget: int, request: ChatCompletionRequest):
    m_ptr, c_ptr = ENGINE.get_model_ptr(), ENGINE.get_context_ptr(pool)
    n_tokens, start_pos = len(tokens), 0
    if hit:
        start_pos = hit["len"]
        ENGINE.clone_sequence(pool, hit["sid"], sid, start_pos)
    
    batch = llama_cpp.llama_batch_init(max(n_tokens, BATCH_SIZE), 0, 1)
    try:
        batch.n_tokens = n_tokens - start_pos
        for i in range(batch.n_tokens):
            batch.token[i], batch.pos[i], batch.n_seq_id[i], batch.seq_id[i][0], batch.logits[i] = tokens[start_pos+i], start_pos+i, 1, sid, (i == batch.n_tokens - 1)
        if batch.n_tokens > 0 and llama_cpp.llama_decode(c_ptr, batch) != 0: raise RuntimeError("Decode fail")
        
        output_text = ""
        last_tokens = list(tokens)
        n_cur, finish_reason = n_tokens, "stop"
        
        for i in range(max_tokens):
            if budget > 0 and (n_cur + 1) >= budget:
                finish_reason = "length"
                break
            
            ntid = apply_sampling(c_ptr, m_ptr, last_tokens, request, grammar)
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

def low_level_stream_start(pool: str, sid: int, tokens: List[int], hit: Optional[Dict[str, Any]]):
    m_ptr, c_ptr = ENGINE.get_model_ptr(), ENGINE.get_context_ptr(pool)
    n_tokens, start_pos = len(tokens), 0
    if hit:
        start_pos = hit["len"]
        ENGINE.clone_sequence(pool, hit["sid"], sid, start_pos)
    batch = llama_cpp.llama_batch_init(max(n_tokens, BATCH_SIZE), 0, 1)
    try:
        batch.n_tokens = n_tokens - start_pos
        for i in range(batch.n_tokens):
            batch.token[i], batch.pos[i], batch.n_seq_id[i], batch.seq_id[i][0], batch.logits[i] = tokens[start_pos+i], start_pos+i, 1, sid, (i == batch.n_tokens - 1)
        if batch.n_tokens > 0 and llama_cpp.llama_decode(c_ptr, batch) != 0: raise RuntimeError("Prefill fail")
    finally: llama_cpp.llama_batch_free(batch)
    return n_tokens

def low_level_stream_step(pool: str, sid: int, n_cur: int, last_tokens: List[int], request: ChatCompletionRequest, grammar: Optional[Any], budget: int):
    m_ptr, c_ptr = ENGINE.get_model_ptr(), ENGINE.get_context_ptr(pool)
    if budget > 0 and (n_cur + 1) >= budget: return None, 0, "length"
    
    ntid = apply_sampling(c_ptr, m_ptr, last_tokens, request, grammar)
    if ntid == llama_cpp.llama_token_eos(m_ptr): return None, ntid, "stop"
    
    buf = ctypes.create_string_buffer(128)
    nb = llama_cpp.llama_token_to_piece(m_ptr, ntid, buf, 128, 0, False)
    piece = buf[:nb].decode('utf-8', errors='ignore')
    
    batch = llama_cpp.llama_batch_init(1, 0, 1)
    try:
        batch.n_tokens = 1
        batch.token[0], batch.pos[0], batch.n_seq_id[0], batch.seq_id[0][0], batch.logits[0] = ntid, n_cur, 1, sid, True
        if llama_cpp.llama_decode(c_ptr, batch) != 0: return None, ntid, "error"
    finally: llama_cpp.llama_batch_free(batch)
    
    return piece, ntid, None

# --- Routes ---

execution_mutex = asyncio.Lock()

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, dependencies=[Depends(verify_auth)])
@app.post("/v1/ctx/{ctx_size}/chat/completions", response_model=ChatCompletionResponse, dependencies=[Depends(verify_auth)])
async def chat_completions(request: ChatCompletionRequest, ctx_size: Optional[int] = None, x_sluice_required_ctx: Optional[int] = Header(None)):
    rid = f"sluice-{uuid.uuid4().hex[:8]}"
    prompt = format_prompt(request.messages, request.tools)
    tokens = get_tokens(prompt)
    
    true_required = len(tokens) + (request.max_tokens or DEFAULT_MAX_TOKENS)
    final_ctx = ctx_size or x_sluice_required_ctx or request.required_ctx or DEFAULT_CTX
    allocation_size = max(true_required, final_ctx)
    
    pool = POOLS[-1]
    for p in POOLS:
        if p.precision_threshold > 0 and allocation_size <= p.precision_threshold:
            pool = p
            break
            
    is_large = allocation_size >= LARGE_THRESHOLD
    available = BANK.get_available_for_large(pool.name) if is_large else BANK.get_available_for_small(pool.name)
    
    active_messages = request.messages
    if ENABLE_ADAPTIVE_TRIMMING and allocation_size > available:
        if available >= TRIM_FLOOR:
            active_messages = TRIMMER.trim(request.messages, available - (request.max_tokens or DEFAULT_MAX_TOKENS), request.tools)
            prompt = format_prompt(active_messages, request.tools)
            tokens = get_tokens(prompt)
            allocation_size = len(tokens) + (request.max_tokens or DEFAULT_MAX_TOKENS)
        else:
            return Response(content=json.dumps({"error": {"message": "VRAM Full."}}), status_code=429, headers={"Retry-After": RETRY_AFTER})

    grammar = None
    if request.response_format and request.response_format.get("type") == "json_object":
        schema = request.response_format.get("schema")
        if schema: grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(schema))
    hit = CACHE.get(tokens)
    
    try: sid = await BANK.acquire(pool.name, allocation_size)
    except BankSaturated:
        if AUTO_ELASTICITY and not BANK.is_expanded:
            async with execution_mutex:
                ENGINE.hot_swap_context(pool.name, pool.copy(update={"max_tokens": allocation_size}))
                await BANK.update_capacity(pool.name, allocation_size, expanded=True)
            sid = await BANK.acquire(pool.name, allocation_size, timeout=ELASTICITY_TIMEOUT)
        else: return Response(content=json.dumps({"error": {"message": "Queue Full"}}), status_code=429, headers={"Retry-After": RETRY_AFTER})
    except Exception as e: raise HTTPException(503, str(e))

    if request.stream:
        async def stream_wrapper():
            last_tokens = list(tokens)
            output_text = ""
            try:
                async with execution_mutex:
                    loop = asyncio.get_event_loop()
                    n_cur = await loop.run_in_executor(None, low_level_stream_start, pool.name, sid, tokens, hit)
                
                for _ in range(request.max_tokens or DEFAULT_MAX_TOKENS):
                    async with execution_mutex:
                        # Task 1/Fix: Return ntid and append to last_tokens
                        piece, ntid, finish = await loop.run_in_executor(None, low_level_stream_step, pool.name, sid, n_cur, last_tokens, request, grammar, allocation_size)
                    
                    if finish:
                        yield f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]})}\n\n"
                        break
                    
                    output_text += piece
                    last_tokens.append(ntid)
                    
                    yield f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'content': piece}, 'finish_reason': None}]})}\n\n"
                    
                    if check_stop_sequences(output_text, request.stop):
                        yield f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                        break
                    n_cur += 1
                    await asyncio.sleep(0)
                yield "data: [DONE]\n\n"
            finally:
                ENGINE.remove_sequence(pool.name, sid)
                await BANK.release(sid)
        return StreamingResponse(stream_wrapper(), media_type="text/event-stream")

    try:
        t0 = time.time()
        async with execution_mutex:
            text, n_p, n_g, f_r = await asyncio.get_event_loop().run_in_executor(None, low_level_generate, pool.name, sid, tokens, request.max_tokens or DEFAULT_MAX_TOKENS, hit, grammar, allocation_size, request)
        METRIC_INF_LATENCY.labels(pool=pool.name, type="non-stream").observe(time.time() - t0)
        if request.cache_prompt and not hit and len(tokens) >= CACHE_MIN_TOKENS:
            CACHE.put(tokens, sid, pool.name)
            await BANK.release(sid, pin=True)
        else:
            ENGINE.remove_sequence(pool.name, sid)
            await BANK.release(sid)
        return ChatCompletionResponse(id=rid, created=int(time.time()), model=request.model, choices=[{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": f_r}], usage={"prompt_tokens": n_p, "completion_tokens": n_g, "total_tokens": n_p + n_g})
    except Exception as e:
        ENGINE.remove_sequence(pool.name, sid)
        await BANK.release(sid)
        raise HTTPException(500, str(e))

@app.post("/v1/embeddings", response_model=EmbeddingResponse, dependencies=[Depends(verify_auth)])
async def embeddings(request: EmbeddingRequest):
    inputs = [request.input] if isinstance(request.input, str) else request.input
    final_ctx = request.required_ctx or max(DEFAULT_CTX, int(sum(len(i) for i in inputs) * 1.5))
    pool_name = POOLS[0].name
    sid = await BANK.acquire(pool_name, final_ctx)
    try:
        data, total_p = [], 0
        async with execution_mutex:
            for idx, text in enumerate(inputs):
                tokens = get_tokens(text)
                def proc(s, tks):
                    batch = llama_cpp.llama_batch_init(len(tks), 0, 1)
                    try:
                        for i in range(len(tks)): batch.token[i], batch.pos[i], batch.n_seq_id[i], batch.seq_id[i][0], batch.logits[i] = tks[i], i, 1, s, False
                        batch.n_tokens = len(tks)
                        if llama_cpp.llama_decode(ENGINE.get_context_ptr(pool_name), batch) != 0: raise RuntimeError("Decode fail")
                        return ENGINE.get_embeddings(pool_name, s), len(tks)
                    finally:
                        llama_cpp.llama_batch_free(batch)
                        ENGINE.remove_sequence(pool_name, s)
                vec, n = await asyncio.get_event_loop().run_in_executor(None, proc, sid, tokens)
                data.append({"object": "embedding", "index": idx, "embedding": vec})
                total_p += n
        return EmbeddingResponse(model=request.model, data=data, usage={"prompt_tokens": total_p, "total_tokens": total_p})
    finally: await BANK.release(sid)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
