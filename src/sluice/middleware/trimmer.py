import json
from typing import List, Dict, Any, Optional, Callable

class MiddleOutTrimmer:
    """
    Algorithmic middle-out conversation trimmer for Adaptive Context Negotiation.
    Ensures that the System Prompt and Recent Context are preserved while 
    pruning older historical turns to fit within a VRAM token budget.
    """
    def __init__(self, get_tokens_func: Callable[[str], List[int]], format_prompt_func: Callable[[List[Any], Optional[List[Any]]], str]):
        self.get_tokens = get_tokens_func
        self.format_prompt = format_prompt_func

    def trim(self, messages: List[Any], target_tokens: int, tools: Optional[List[Any]] = None) -> List[Any]:
        """
        Trims messages from the middle until the total token count is <= target_tokens.
        
        1. Keep index 0 (System).
        2. Keep last 3 turns (Recent Context).
        3. Evict oldest messages in the middle (index 1 onwards).
        """
        if len(messages) <= 4:
            return messages
            
        def get_size(msgs):
            prompt = self.format_prompt(msgs, tools)
            return len(self.get_tokens(prompt))

        current_size = get_size(messages)
        if current_size <= target_tokens:
            return messages

        print(f"[TRIMMER] Target: {target_tokens}, Current: {current_size}. Starting middle-out prune...")
        
        system_msg = messages[0]
        tail_msgs = messages[-3:]
        middle_msgs = list(messages[1:-3])

        # Iteratively remove the oldest middle message until we fit
        while len(middle_msgs) > 0:
            middle_msgs.pop(0)
            if get_size([system_msg] + middle_msgs + tail_msgs) <= target_tokens:
                break
        
        final_msgs = [system_msg] + middle_msgs + tail_msgs
        print(f"[TRIMMER] Pruned {len(messages) - len(final_msgs)} messages. Final size: {get_size(final_msgs)}")
        return final_msgs
