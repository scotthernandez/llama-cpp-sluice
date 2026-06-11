import ctypes
import asyncio
import os
import time
import uuid
import json
from typing import Optional, List, Dict, Any, Generator
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
PORT = int(os.getenv("SLUICE_PORT", "8001"))

app = FastAPI(title="Llama-CPP Sluice: Dynamic Asymmetric Inference Server")
security = HTTPBearer(auto_error=False)

# --- Authentication ---

async def verify_auth(auth: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not API_KEY:
        return
    if auth is None or auth.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

# --- Prometheus Metrics ---

METRIC_INF_LATENCY = Histogram("sluice_inference_latency_seconds", "Inference time in seconds", ["model", "type"])
METRIC_TOKENS_TOTAL = Counter("sluice_tokens_generated_total", "Total tokens generated", ["model"])
METRIC_POOL_USED = Gauge("sluice_vram_used_tokens", "Currently used tokens in the KV pool")
METRIC_POOL_TOTAL = Gauge("sluice_vram_total_tokens", "Total tokens available in the KV pool")
METRIC_WAITING_LARGE = Gauge("sluice_requests_waiting_large", "Large requests currently waiting for VRAM")
METRIC_EXPANDED = Gauge("sluice_is_expanded", "1 if the pool is in expanded/scavenged state")

# Global State
ENGINE: Optional[SluiceEngine] = None
BANK: TokenBank = TokenBank(BASE_POOL, RESERVED_POOL, LARGE_THRESHOLD, SCAVENGE_HOOK, RECOVERY_HOOK)

# --- Elasticity Coordinator ---

async def elasticity_loop():
    """Background task to manage pool shrinking and recovery."""
    while True:
        await asyncio.sleep(ELASTICITY_INTERVAL)
        
        # Update metrics every interval
        stats = BANK.get_stats()
        METRIC_POOL_USED.set(stats["used"])
        METRIC_POOL_TOTAL.set(stats["total"])
        METRIC_WAITING_LARGE.set(stats["waiting_large"])
        METRIC_EXPANDED.set(1 if stats["is_expanded"] else 0)

        if not AUTO_ELASTICITY or not BANK.is_expanded:
            continue
            
        async with BANK.condition:
            if BANK.used == 0 and BANK.waiting_large == 0:
                print("[ELASTICITY] System idle. Shrinking context back to base...")
                if RECOVERY_HOOK:
                    await BANK._run_hook(RECOVERY_HOOK, "Recovery")
                
                ENGINE.hot_swap_context(BASE_POOL)
                await BANK.update_total(BASE_POOL, expanded=False)

# --- OpenAI Compatibility Models ---

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatCompletionRequest(BaseModel):
    model: str = "sluice-model"
    messages: List[ChatMessage]
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    max_tokens: Optional[int] = 128
    temperature: float = 0.0
    stream: bool = False
    required_ctx: Optional[int] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

# --- Startup ---

@app.on_event("startup")
async def startup():
    global ENGINE
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
    ENGINE = SluiceEngine(MODEL_PATH, BASE_POOL)
    METRIC_POOL_TOTAL.set(BASE_POOL)
    if AUTO_ELASTICITY:
        asyncio.create_task(elasticity_loop())
    else:
        # Still run metrics updates if elasticity is off
        async def metrics_updater():
            while True:
                await asyncio.sleep(5)
                stats = BANK.get_stats()
                METRIC_POOL_USED.set(stats["used"])
                METRIC_POOL_TOTAL.set(stats["total"])
                METRIC_WAITING_LARGE.set(stats["waiting_large"])
        asyncio.create_task(metrics_updater())

# --- Admin Routes ---

@app.get("/metrics", dependencies=[Depends(verify_auth)])
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/v1/admin/drain", dependencies=[Depends(verify_auth)])
async def admin_drain():
    """Stops accepting new requests and waits for active ones to finish."""
    asyncio.create_task(BANK.drain())
    return {"status": "draining"}

@app.post("/v1/admin/resume", dependencies=[Depends(verify_auth)])
async def admin_resume():
    """Resumes accepting requests."""
    await BANK.resume()
    return {"status": "running"}

@app.post("/v1/admin/defrag", dependencies=[Depends(verify_auth)])
async def admin_defrag():
    """Forces internal KV cache compaction."""
    # Check if llama_kv_cache_defrag exists in this version of llama-cpp-python
    if hasattr(llama_cpp, 'llama_kv_cache_defrag'):
        ENGINE.defrag()
        return {"status": "defrag_scheduled"}
    else:
        raise HTTPException(status_code=501, detail="Internal defrag not supported in this llama-cpp-python version.")

@app.post("/v1/admin/resize", dependencies=[Depends(verify_auth)])
async def admin_resize(new_size: int = Body(..., embed=True)):
    """Gracefully drains, hot-swaps context size, and resumes."""
    await BANK.drain()
    ENGINE.hot_swap_context(new_size)
    await BANK.update_total(new_size, expanded=(new_size > BASE_POOL))
    await BANK.resume()
    return {"status": "resized", "new_size": new_size}

# --- Inference Core ---

def format_prompt(messages: List[ChatMessage], tools: Optional[List[Dict[str, Any]]] = None) -> str:
    """Simple chat-to-prompt conversion with tool support."""
    prompt = ""
    if tools:
        prompt += "Available Tools:\n"
        for tool in tools:
            prompt += f"- {json.dumps(tool)}\n"
        prompt += "\n"
    
    for m in messages:
        if m.content:
            prompt += f"{m.role}: {m.content}\n"
        if m.tool_calls:
            for tc in m.tool_calls:
                prompt += f"call: {json.dumps(tc)}\n"
                
    return prompt + "assistant: "

def low_level_generate(sid: int, prompt: str, max_tokens: int):
    """Executes non-streaming inference."""
    model_ptr = ENGINE.get_model_ptr()
    ctx_ptr = ENGINE.get_context_ptr()
    
    prompt_bytes = prompt.encode('utf-8')
    tokens_list = (llama_cpp.llama_token * (len(prompt_bytes) + 1))()
    n_tokens = llama_cpp.llama_tokenize(
        model_ptr, prompt_bytes, len(prompt_bytes), tokens_list, len(tokens_list), True, True
    )
    
    batch = llama_cpp.llama_batch_init(max(n_tokens, 512), 0, 1)
    try:
        batch.n_tokens = n_tokens
        for i in range(n_tokens):
            batch.token[i] = tokens_list[i]
            batch.pos[i] = i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = sid
            batch.logits[i] = (i == n_tokens - 1)

        if llama_cpp.llama_decode(ctx_ptr, batch) != 0:
            raise RuntimeError("Initial decode failed")

        output_text = []
        n_cur = n_tokens
        for _ in range(max_tokens):
            logits = llama_cpp.llama_get_logits_ith(ctx_ptr, batch.n_tokens - 1)
            n_vocab = llama_cpp.llama_n_vocab(model_ptr)
            candidates = (llama_cpp.llama_token_data * n_vocab)()
            for i in range(n_vocab):
                candidates[i] = llama_cpp.llama_token_data(id=i, logit=logits[i], p=0.0)
            candidates_p = llama_cpp.llama_token_data_array(data=candidates, size=n_vocab, sorted=False)
            new_token_id = llama_cpp.llama_sample_token_greedy(ctx_ptr, ctypes.byref(candidates_p))
            
            if new_token_id == llama_cpp.llama_token_eos(model_ptr):
                break
            
            buf = ctypes.create_string_buffer(32)
            n_bytes = llama_cpp.llama_token_to_piece(model_ptr, new_token_id, buf, len(buf), 0, False)
            output_text.append(buf[:n_bytes].decode('utf-8', errors='ignore'))
            
            batch.n_tokens = 1
            batch.token[0] = new_token_id
            batch.pos[0] = n_cur
            batch.n_seq_id[0] = 1
            batch.seq_id[0][0] = sid
            batch.logits[0] = True
            if llama_cpp.llama_decode(ctx_ptr, batch) != 0:
                break
            n_cur += 1
        return "".join(output_text), n_tokens, (n_cur - n_tokens)
    finally:
        llama_cpp.llama_batch_free(batch)

def low_level_stream_generator(sid: int, prompt: str, max_tokens: int, request_id: str, model_name: str):
    """Executes streaming inference yielding OpenAI-compatible SSE chunks."""
    model_ptr = ENGINE.get_model_ptr()
    ctx_ptr = ENGINE.get_context_ptr()
    
    prompt_bytes = prompt.encode('utf-8')
    tokens_list = (llama_cpp.llama_token * (len(prompt_bytes) + 1))()
    n_tokens = llama_cpp.llama_tokenize(
        model_ptr, prompt_bytes, len(prompt_bytes), tokens_list, len(tokens_list), True, True
    )
    
    batch = llama_cpp.llama_batch_init(max(n_tokens, 512), 0, 1)
    try:
        batch.n_tokens = n_tokens
        for i in range(n_tokens):
            batch.token[i] = tokens_list[i]
            batch.pos[i] = i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = sid
            batch.logits[i] = (i == n_tokens - 1)

        if llama_cpp.llama_decode(ctx_ptr, batch) != 0:
            yield f"data: {json.dumps({'error': 'Initial decode failed'})}\n\n"
            return

        n_cur = n_tokens
        for _ in range(max_tokens):
            logits = llama_cpp.llama_get_logits_ith(ctx_ptr, batch.n_tokens - 1)
            n_vocab = llama_cpp.llama_n_vocab(model_ptr)
            candidates = (llama_cpp.llama_token_data * n_vocab)()
            for i in range(n_vocab):
                candidates[i] = llama_cpp.llama_token_data(id=i, logit=logits[i], p=0.0)
            candidates_p = llama_cpp.llama_token_data_array(data=candidates, size=n_vocab, sorted=False)
            new_token_id = llama_cpp.llama_sample_token_greedy(ctx_ptr, ctypes.byref(candidates_p))
            
            if new_token_id == llama_cpp.llama_token_eos(model_ptr):
                break
            
            buf = ctypes.create_string_buffer(32)
            n_bytes = llama_cpp.llama_token_to_piece(model_ptr, new_token_id, buf, len(buf), 0, False)
            token_text = buf[:n_bytes].decode('utf-8', errors='ignore')
            
            # Yield OpenAI chunk
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": token_text}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            METRIC_TOKENS_TOTAL.labels(model=model_name).inc()

            batch.n_tokens = 1
            batch.token[0] = new_token_id
            batch.pos[0] = n_cur
            batch.n_seq_id[0] = 1
            batch.seq_id[0][0] = sid
            batch.logits[0] = True
            if llama_cpp.llama_decode(ctx_ptr, batch) != 0:
                break
            n_cur += 1
            
        # Final chunk
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    finally:
        llama_cpp.llama_batch_free(batch)
        llama_cpp.llama_memory_seq_rm(ENGINE.get_memory(), sid, -1, -1)
        # Note: BANK.release(sid) MUST be called by the wrapper after the generator is exhausted

# --- Routes ---

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, dependencies=[Depends(verify_auth)])
@app.post("/v1/ctx/{ctx_size}/chat/completions", response_model=ChatCompletionResponse, dependencies=[Depends(verify_auth)])
async def chat_completions(
    request: ChatCompletionRequest,
    ctx_size: Optional[int] = None,
    x_sluice_required_ctx: Optional[int] = Header(None)
):
    final_ctx = ctx_size or x_sluice_required_ctx or request.required_ctx or 2048
    request_id = f"sluice-{uuid.uuid4().hex[:8]}"

    try:
        sid = await BANK.acquire(final_ctx)
    except TimeoutError as e:
        if AUTO_ELASTICITY and not BANK.is_expanded:
            print(f"[ELASTICITY] Attempting emergency expansion to {final_ctx}...")
            ENGINE.hot_swap_context(final_ctx)
            await BANK.update_total(final_ctx, expanded=True)
            sid = await BANK.acquire(final_ctx, timeout=10.0)
        else:
            raise HTTPException(status_code=503, detail=str(e))

    prompt = format_prompt(request.messages, request.tools)

    if request.stream:
        async def stream_wrapper():
            try:
                # Run the generator in a thread pool since it's blocking
                # We use a custom generator that handles its ownsid cleanup
                for chunk in low_level_stream_generator(sid, prompt, request.max_tokens or 128, request_id, request.model):
                    yield chunk
                    await asyncio.sleep(0) # Yield control
            finally:
                await BANK.release(sid)
        
        return StreamingResponse(stream_wrapper(), media_type="text/event-stream")

    try:
        t0 = time.time()
        loop = asyncio.get_event_loop()
        text, n_prompt, n_gen = await loop.run_in_executor(None, low_level_generate, sid, prompt, request.max_tokens or 128)
        elapsed = time.time() - t0
        
        METRIC_INF_LATENCY.labels(model=request.model, type="non-stream").observe(elapsed)
        METRIC_TOKENS_TOTAL.labels(model=request.model).inc(n_gen)

        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }],
            usage={"prompt_tokens": n_prompt, "completion_tokens": n_gen, "total_tokens": n_prompt + n_gen}
        )
    finally:
        llama_cpp.llama_memory_seq_rm(ENGINE.get_memory(), sid, -1, -1)
        await BANK.release(sid)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
