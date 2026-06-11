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
from .pools import PoolConfig, DEFAULT_POOLS

# Configuration from Environment
MODEL_PATH = os.getenv("SLUICE_MODEL_PATH", "/models/gguf/model.gguf")
RESERVED_POOL = int(os.getenv("SLUICE_RESERVED_POOL", "32768"))
LARGE_THRESHOLD = int(os.getenv("SLUICE_LARGE_THRESHOLD", "16384"))
PRECISION_THRESHOLD = int(os.getenv("SLUICE_PRECISION_THRESHOLD", "8192"))
SCAVENGE_HOOK = os.getenv("SLUICE_SCAVENGE_HOOK")
RECOVERY_HOOK = os.getenv("SLUICE_RECOVERY_HOOK")
API_KEY = os.getenv("SLUICE_API_KEY")
AUTO_ELASTICITY = os.getenv("SLUICE_AUTO_ELASTICITY", "false").lower() == "true"
AUTO_DEFRAG = os.getenv("SLUICE_AUTO_DEFRAG", "true").lower() == "true"
DEFRAG_THRESHOLD = float(os.getenv("SLUICE_DEFRAG_THRESHOLD", "0.15"))
ELASTICITY_INTERVAL = float(os.getenv("SLUICE_ELASTICITY_INTERVAL", "5.0"))
PREFIX_CACHE_LIMIT = int(os.getenv("SLUICE_PREFIX_CACHE_LIMIT", "10"))
PORT = int(os.getenv("SLUICE_PORT", "8001"))

# Load Pools from Env or Default
POOLS_JSON = os.getenv("SLUICE_POOLS")
if POOLS_JSON: POOLS = [PoolConfig(**p) for p in json.loads(POOLS_JSON)]
else: POOLS = DEFAULT_POOLS

app = FastAPI(title="Llama-CPP Sluice: Multi-Precision Asymmetric Server")
security = HTTPBearer(auto_error=False)

# --- Metrics ---
METRIC_INF_LATENCY = Histogram("sluice_latency", "Latency", ["pool", "type"])
METRIC_POOL_USED = Gauge("sluice_pool_used", "Used tokens", ["pool"])
METRIC_POOL_TOTAL = Gauge("sluice_pool_total", "Total tokens", ["pool"])
METRIC_FRAG_RATIO = Gauge("sluice_frag_ratio", "Frag ratio", ["pool"])

# --- Radix Cache ---
class PrefixCache:
    def __init__(self, limit: int):
        self.limit = limit
        self.cache: Dict[str, Dict[str, Any]] = {}
    def get(self, tokens: List[int]) -> Optional[Dict[str, Any]]:
        if len(tokens) < 128: return None
        h = hash(tuple(tokens[:512]))
        return self.cache.get(h)
    def put(self, tokens: List[int], sid: int, pool: str):
        if len(tokens) < 128: return
        h = hash(tuple(tokens[:512]))
        if len(self.cache) >= self.limit:
            oldest = min(self.cache.keys(), key=lambda k: self.cache[k]["last_used"])
            asyncio.create_task(BANK.evict(self.cache[oldest]["sid"]))
            ENGINE.remove_sequence(self.cache[oldest]["pool"], self.cache[oldest]["sid"])
            del self.cache[oldest]
        self.cache[h] = {"sid": sid, "pool": pool, "len": 512, "last_used": time.time()}

CACHE = PrefixCache(PREFIX_CACHE_LIMIT)
ENGINE: Optional[SluiceEngine] = None
BANK: Optional[TokenBank] = None

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

@app.on_event("startup")
async def startup():
    global ENGINE, BANK
    ENGINE = SluiceEngine(MODEL_PATH, POOLS)
    capacities = {p.name: p.max_tokens for p in POOLS}
    BANK = TokenBank(list(capacities.keys()), capacities, RESERVED_POOL, LARGE_THRESHOLD, SCAVENGE_HOOK, RECOVERY_HOOK)
    asyncio.create_task(maintenance_loop())

async def verify_auth(auth: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if API_KEY and (not auth or auth.credentials != API_KEY): raise HTTPException(401)

class ChatCompletionRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_tokens: int = 128
    required_ctx: Optional[int] = None
    stream: bool = False

@app.post("/v1/chat/completions", dependencies=[Depends(verify_auth)])
async def chat(request: ChatCompletionRequest):
    # Routing Logic: No hardcoding
    # Choose pool based on precision_threshold
    needed = request.required_ctx or 2048
    pool = POOLS[-1] # Default to last (usually efficiency)
    for p in POOLS:
        if p.precision_threshold > 0 and needed <= p.precision_threshold:
            pool = p
            break
            
    sid = await BANK.acquire(pool.name, needed)
    try:
        # Simple loop for now (Continuous batching handled in v0.3.0 logic)
        # We'll stick to the verified sid tracking for v0.4.0 stability
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in request.messages]) + "\nassistant: "
        m_ptr, c_ptr = ENGINE.get_model_ptr(), ENGINE.get_context_ptr(pool.name)
        p_bytes = prompt.encode('utf-8')
        tks = (llama_cpp.llama_token * (len(p_bytes)+1))()
        n = llama_cpp.llama_tokenize(m_ptr, p_bytes, len(p_bytes), tks, len(tks), True, True)
        batch = llama_cpp.llama_batch_init(max(n, 512), 0, 1)
        batch.n_tokens = n
        for i in range(n): batch.token[i], batch.pos[i], batch.n_seq_id[i], batch.seq_id[i][0], batch.logits[i] = tks[i], i, 1, sid, (i == n-1)
        llama_cpp.llama_decode(c_ptr, batch)
        
        output = []
        for _ in range(request.max_tokens):
            logits = llama_cpp.llama_get_logits_ith(c_ptr, batch.n_tokens - 1)
            candidates = (llama_cpp.llama_token_data * llama_cpp.llama_n_vocab(m_ptr))()
            for i in range(len(candidates)): candidates[i] = llama_cpp.llama_token_data(id=i, logit=logits[i], p=0.0)
            candidates_p = llama_cpp.llama_token_data_array(data=candidates, size=len(candidates), sorted=False)
            ntid = llama_cpp.llama_sample_token_greedy(c_ptr, ctypes.byref(candidates_p))
            if ntid == llama_cpp.llama_token_eos(m_ptr): break
            buf = ctypes.create_string_buffer(128)
            nb = llama_cpp.llama_token_to_piece(m_ptr, ntid, buf, 128, 0, False)
            output.append(buf[:nb].decode('utf-8', errors='ignore'))
            batch.n_tokens, batch.token[0], batch.pos[0], batch.logits[0] = 1, ntid, n + len(output), True
            llama_cpp.llama_decode(c_ptr, batch)
        
        return {"choices": [{"message": {"role": "assistant", "content": "".join(output)}}]}
    finally:
        ENGINE.remove_sequence(pool.name, sid)
        await BANK.release(sid)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
