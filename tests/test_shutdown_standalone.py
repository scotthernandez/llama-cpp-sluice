"""Standalone verification of threading.Event shutdown pattern."""
import sys
import threading
import time

def test_threading_event_basics():
    s = threading.Event()
    assert not s.is_set()
    s.set()
    assert s.is_set()

def test_generation_loop_shutdown_check():
    SHUT = threading.Event()
    def gen_loop(max_tokens):
        for i in range(max_tokens):
            if SHUT.is_set(): return "shutdown"
        return "length"
    
    assert gen_loop(100) == "length"
    SHUT.set()
    assert gen_loop(100) == "shutdown"

if __name__ == "__main__":
    test_threading_event_basics()
    test_generation_loop_shutdown_check()
    print("Standalone tests passed")
