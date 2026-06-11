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

# Global State
ENGINE: Optional[SluiceEngine] = None
BANK: TokenBank = TokenBank(TOTAL_POOL, RESERVED_POOL, LARGE_THRESHOLD)

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=128, ge=1)
    required_ctx: int = Field(default=2048, ge=512)
    temperature: float = 0.0

@app.on_event("startup")
def startup():
    global ENGINE
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        # In a real startup, we might want to crash here
    ENGINE = SluiceEngine(MODEL_PATH, TOTAL_POOL)

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
            batch.logits[0] = True
            if llama_cpp.llama_decode(ctx_ptr, batch) != 0:
                break
            n_cur += 1
            
        return "".join(output_text)
    finally:
        llama_cpp.llama_batch_free(batch)

@app.post("/v1/execute")
async def execute(request: InferenceRequest):
    # 1. Wait for VRAM allocation in the bank
    try:
        sid = await BANK.acquire(request.required_ctx)
    except TimeoutError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        # 2. Run inference (threaded to not block async loop)
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, low_level_generate, sid, request.prompt, request.max_tokens)
        return {"id": sid, "text": text}
    finally:
        # 3. Release tokens and clear sequence from KV cache
        llama_cpp.llama_kv_cache_seq_rm(ENGINE.get_context_ptr(), sid, -1, -1)
        await BANK.release(sid)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
