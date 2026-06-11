import llama_cpp
from pydantic import BaseModel
from typing import List, Dict

class PoolConfig(BaseModel):
    name: str
    max_tokens: int
    type_k: int = llama_cpp.GGML_TYPE_F16
    type_v: int = llama_cpp.GGML_TYPE_F16
    precision_threshold: int = 0 # 0 means use for any size

# Default configuration: Precision (F16) and Efficiency (Q4_0)
DEFAULT_POOLS = [
    PoolConfig(
        name="precision",
        max_tokens=16384,
        type_k=llama_cpp.GGML_TYPE_F16,
        type_v=llama_cpp.GGML_TYPE_F16,
        precision_threshold=8192
    ),
    PoolConfig(
        name="efficiency",
        max_tokens=131072,
        type_k=llama_cpp.GGML_TYPE_Q4_0,
        type_v=llama_cpp.GGML_TYPE_Q4_0,
        precision_threshold=0
    )
]
