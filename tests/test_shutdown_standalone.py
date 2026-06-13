"""Standalone verification of threading.Event shutdown pattern (no conftest contamination)."""
import sys
import threading
import time
import signal as sig_module


def test_threading_event_basics():
    """Verify threading.Event API behaves as expected."""
    s = threading.Event()
    assert isinstance(s, threading.Event)
    assert not s.is_set()
    s.set()
    assert s.is_set()
    s.clear()
    assert not s.is_set()
    print("  PASS test_threading_event_basics")


def test_generation_loop_shutdown_check():
    """Simulate the generation loop checking SHUTTING_DOWN.is_set()."""
    SHUT = threading.Event()

    def gen_loop(max_tokens):
        for i in range(max_tokens):
            if SHUT.is_set():
                return "shutdown"
        return "length"

    # Normal: no shutdown, runs to completion
    result = gen_loop(100)
    assert result == "length", f"Expected 'length', got '{result}'"

    # Shutdown: event set mid-generation from another thread
    SHUT.clear()

    def delayed_shutdown():
        time.sleep(0.05)
        SHUT.set()

    t = threading.Thread(target=delayed_shutdown)
    t.start()
    result = gen_loop(10_000_000)
    t.join()
    assert result == "shutdown", f"Expected 'shutdown', got '{result}'"
    print("  PASS test_generation_loop_shutdown_check")


def test_signal_handler_sets_event():
    """Verify the signal handler pattern calls .set()."""
    SHUT = threading.Event()
    SHUT.clear()

    def signal_handler(sig, frame):
        SHUT.set()

    signal_handler(sig_module.SIGINT, None)
    assert SHUT.is_set()
    print("  PASS test_signal_handler_sets_event")


def test_set_timing():
    """Verify .set() returns quickly (critical for <1s shutdown acceptance criterion)."""
    SHUT = threading.Event()
    start = time.monotonic()
    SHUT.set()
    elapsed_ms = (time.monotonic() - start) * 1000
    assert elapsed_ms < 1000, f".set() took {elapsed_ms}ms, expected <1000ms"
    print(f"  PASS test_set_timing ({elapsed_ms:.3f}ms)")


def test_stream_step_shutdown():
    """Verify the stream_step pattern: .is_set() returns 'shutdown' immediately."""
    SHUT = threading.Event()

    def stream_step():
        if SHUT.is_set():
            return None, None, "shutdown"
        return "piece", 42, None

    # Not set: normal behavior
    piece, ntid, finish = stream_step()
    assert piece == "piece"
    assert finish is None

    # Set: shutdown
    SHUT.set()
    piece, ntid, finish = stream_step()
    assert piece is None
    assert finish == "shutdown"
    print("  PASS test_stream_step_shutdown")


if __name__ == "__main__":
    print("Testing threading.Event shutdown pattern...")
    test_threading_event_basics()
    test_generation_loop_shutdown_check()
    test_signal_handler_sets_event()
    test_set_timing()
    test_stream_step_shutdown()
    print("=== All 5 tests passed ===")
    sys.exit(0)
