import ctypes
import asyncio
import os
from typing import Optional
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
import llama_cpp

from .bank import TokenBank
from .engine import SluiceEngine

# Configuration from Environment
MODEL_PATH = os.getenv("SLUICE_MODEL_PATH", "/models/gguf/model.gguf")
TOTAL_POOL = int(os.getenv("SLUICE_TOTAL_POOL", "98304"))
RESERVED_POOL = int(os.getenv("SLUICE_RESERVED_POOL", "32768"))
LARGE_THRESHOLD = int(os.getenv("SLUICE_LARGE_THRESHOLD", "16384"))
PORT = int(os.getenv("SLUICE_PORT", "8001"))

app = FastAPI(title="Llama-CPP Sluice: Dynamic Asymmetric Inference Server")

import ctypes
import asyncio
import os
import time
import uuid
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Body, Header, Path
from pydantic import BaseModel, Field
import llama_cpp

from .bank import TokenBank
from .engine import SluiceEngine

# Configuration from Environment
MODEL_PATH = os.getenv("SLUICE_MODEL_PATH", "/models/gguf/model.gguf")
TOTAL_POOL = int(os.getenv("SLUICE_TOTAL_POOL", "98304"))
RESERVED_POOL = int(os.getenv("SLUICE_RESERVED_POOL", "32768"))
LARGE_THRESHOLD = int(os.getenv("SLUICE_LARGE_THRESHOLD", "16384"))
SCAVENGE_HOOK = os.getenv("SLUICE_SCAVENGE_HOOK")
PORT = int(os.getenv("SLUICE_PORT", "8001"))

app = FastAPI(title="Llama-CPP Sluice: Dynamic Asymmetric Inference Server")

# Global State
ENGINE: Optional[SluiceEngine] = None
BANK: TokenBank = TokenBank(TOTAL_POOL, RESERVED_POOL, LARGE_THRESHOLD, SCAVENGE_HOOK)

# ... (ChatMessage, ChatCompletionRequest, ChatCompletionResponse definitions) ...

# --- Admin Routes ---

@app.post("/v1/admin/drain")
async def admin_drain():
    """Stops accepting new requests and waits for active ones to finish."""
    asyncio.create_task(BANK.drain())
    return {"status": "draining"}

@app.post("/v1/admin/resume")
async def admin_resume():
    """Resumes accepting requests."""
    await BANK.resume()
    return {"status": "running"}

@app.post("/v1/admin/defrag")
async def admin_defrag():
    """Forces internal KV cache compaction."""
    ENGINE.defrag()
    return {"status": "defrag_scheduled"}

@app.post("/v1/admin/resize")
async def admin_resize(new_size: int = Body(..., embed=True)):
    """Gracefully drains, hot-swaps context size, and resumes."""
    await BANK.drain()
    ENGINE.hot_swap_context(new_size)
    await BANK.update_total(new_size)
    await BANK.resume()
    return {"status": "resized", "new_size": new_size}

# --- OpenAI Compatibility Models ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "sluice-model"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 128
    temperature: float = 0.0
    # Sluice Extension
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
def startup():
    global ENGINE
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
    ENGINE = SluiceEngine(MODEL_PATH, TOTAL_POOL)

# --- Inference Core ---

def format_prompt(messages: List[ChatMessage]) -> str:
    """Simple chat-to-prompt conversion (Agnostic)."""
    # In production, this should ideally use the model's chat template
    return "\n".join([f"{m.role}: {m.content}" for m in messages]) + "\nassistant: "

def low_level_generate(sid: int, prompt: str, max_tokens: int):
    """Executes inference using the shared engine for a specific sequence ID."""
    model_ptr = ENGINE.get_model_ptr()
    ctx_ptr = ENGINE.get_context_ptr()
    
    prompt_bytes = prompt.encode('utf-8')
    tokens_list = (llama_cpp.llama_token * (len(prompt_bytes) + 1))()
    n_tokens = llama_cpp.llama_tokenize(
        model_ptr, prompt_bytes, len(prompt_bytes), tokens_list, len(tokens_list), True, True
    )
    
    # Initialize Batch
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

# --- Routes ---

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
@app.post("/v1/ctx/{ctx_size}/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    ctx_size: Optional[int] = Path(None),
    x_sluice_required_ctx: Optional[int] = Header(None)
):
    # Determine context size priority:
    # 1. URL Path (/v1/ctx/32768/...)
    # 2. Header (X-Sluice-Required-Ctx)
    # 3. JSON body (required_ctx)
    # 4. Default (2048)
    final_ctx = ctx_size or x_sluice_required_ctx or request.required_ctx or 2048

    try:
        sid = await BANK.acquire(final_ctx)
    except TimeoutError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        prompt = format_prompt(request.messages)
        loop = asyncio.get_event_loop()
        text, n_prompt, n_gen = await loop.run_in_executor(None, low_level_generate, sid, prompt, request.max_tokens)
        
        return ChatCompletionResponse(
            id=f"sluice-{sid}-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": n_prompt,
                "completion_tokens": n_gen,
                "total_tokens": n_prompt + n_gen
            }
        )
    finally:
        llama_cpp.llama_kv_cache_seq_rm(ENGINE.get_context_ptr(), sid, -1, -1)
        await BANK.release(sid)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
