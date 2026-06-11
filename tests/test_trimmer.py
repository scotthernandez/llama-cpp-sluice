import pytest
from sluice.middleware.trimmer import MiddleOutTrimmer

def test_middle_out_trim_logic():
    # Mock callbacks
    def mock_get_tokens(text):
        # 1 word = 1 token for simplicity
        return text.split()
        
    def mock_format_prompt(messages, tools=None):
        return " ".join([m["content"] for m in messages])

    trimmer = MiddleOutTrimmer(mock_get_tokens, mock_format_prompt)
    
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "OLD_1"},
        {"role": "user", "content": "OLD_2"},
        {"role": "user", "content": "RECENT_1"},
        {"role": "user", "content": "RECENT_2"},
        {"role": "user", "content": "RECENT_3"},
    ]
    
    # Target 4 tokens. 
    # Must keep SYSTEM (1) and RECENT_1,2,3 (3). 
    # Total = 4. OLD_1 and OLD_2 should be evicted.
    trimmed = trimmer.trim(messages, target_tokens=4)
    
    assert len(trimmed) == 4
    assert trimmed[0]["content"] == "SYSTEM"
    assert trimmed[1]["content"] == "RECENT_1"
    assert trimmed[-1]["content"] == "RECENT_3"

def test_trim_not_needed():
    def mock_get_tokens(text): return [1] * len(text)
    def mock_format_prompt(messages, tools=None): return "A" * len(messages)
    
    trimmer = MiddleOutTrimmer(mock_get_tokens, mock_format_prompt)
    messages = [{"content": "hi"}] * 10
    
    # Total 10 tokens, budget 20. No trim.
    trimmed = trimmer.trim(messages, target_tokens=20)
    assert len(trimmed) == 10
