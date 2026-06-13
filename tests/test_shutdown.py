"""Tests for threading.Event-based shutdown mechanism."""
import threading
import signal
import pytest
from unittest.mock import patch, MagicMock
import sluice.server
from sluice.server import SHUTTING_DOWN, signal_handler, low_level_generate, low_level_stream_step


class TestShutdownEvent:
    """Verify SHUTTING_DOWN is a proper threading.Event with correct API usage."""

    def test_is_threading_event(self):
        """SHUTTING_DOWN is a threading.Event instance."""
        assert isinstance(SHUTTING_DOWN, threading.Event)

    def test_not_set_by_default(self):
        """Event is clear on module load (no shutdown in progress)."""
        assert SHUTTING_DOWN.is_set() is False

    def test_set_makes_is_set_true(self):
        """After .set(), .is_set() returns True."""
        SHUTTING_DOWN.set()
        try:
            assert SHUTTING_DOWN.is_set() is True
        finally:
            SHUTTING_DOWN.clear()

    def test_clear_makes_is_set_false(self):
        """After .clear(), .is_set() returns False."""
        SHUTTING_DOWN.set()
        SHUTTING_DOWN.clear()
        assert SHUTTING_DOWN.is_set() is False

    def test_signal_handler_sets_event(self):
        """signal_handler calls SHUTTING_DOWN.set() on any signal."""
        # Ensure clean state
        SHUTTING_DOWN.clear()
        assert not SHUTTING_DOWN.is_set()

        signal_handler(signal.SIGINT, None)
        assert SHUTTING_DOWN.is_set()

        # Clean up
        SHUTTING_DOWN.clear()

    def test_sigterm_also_sets_event(self):
        """SIGTERM also triggers shutdown via signal_handler."""
        SHUTTING_DOWN.clear()
        signal_handler(signal.SIGTERM, None)
        assert SHUTTING_DOWN.is_set()
        SHUTTING_DOWN.clear()


class TestGenerationLoopShutdown:
    """Verify the generation loop checks SHUTTING_DOWN.is_set() and exits gracefully."""

    def setup_method(self):
        # Ensure clean state before each test
        SHUTTING_DOWN.clear()

    def teardown_method(self):
        SHUTTING_DOWN.clear()

    def test_low_level_generate_exits_on_shutdown(self):
        """When SHUTTING_DOWN is set, low_level_generate returns 'shutdown' reason."""
        import sluice.server
        import ctypes
        import llama_cpp

        # EOS check should not fire (keep generating)
        orig_eos = llama_cpp.llama_token_eos
        llama_cpp.llama_token_eos = MagicMock(return_value=-1)

        # Token-to-piece returns dummy text
        orig_ttp = llama_cpp.llama_token_to_piece
        buf = ctypes.create_string_buffer(b"ok")
        llama_cpp.llama_token_to_piece = MagicMock(return_value=4)

        try:
            # Configure: enough budget, many max_tokens, so the loop should
            # run until SHUTTING_DOWN.is_set() is True
            max_tokens = 100
            budget = 999_999

            # Set the shutdown event mid-way — simulate SIGINT
            # We'll patch is_set to return True after the prefill phase
            from unittest.mock import PropertyMock
            engine = sluice.server.engine

            # Mock the sampling function to return a dummy token
            def fake_sampling(*args, **kwargs):
                return 42

            with patch("sluice.server.apply_sampling", side_effect=fake_sampling):
                # Simulate: run generation with shutdown event ALREADY set
                # after prefill. The prefill uses batch_size chunks.
                # We need the loop to actually enter the generation loop
                # and then check is_set(). Let's use a call counter.
                call_count = [0]

                class FakeEvent:
                    """Simulate SHUTTING_DOWN becoming set mid-generation."""
                    def is_set(self):
                        call_count[0] += 1
                        return call_count[0] > 1  # not set on first check

                # Check that the code reads the global SHUTTING_DOWN
                # We can't easily replace the global binding inside the
                # function without reloading, so let's test differently:
                # Patch the global reference in the module and reload.
                import importlib
                original = sluice.server.SHUTTING_DOWN
                sluice.server.SHUTTING_DOWN = FakeEvent()
                try:
                    # Now low_level_generate will read our FakeEvent
                    # Need engine pointers mocked
                    text, n_p, n_g, reason = sluice.server.low_level_generate(
                        sid=1, tokens=[1, 2, 3], max_tokens=10,
                        budget=budget,
                        request=MagicMock(stop=None, repeat_penalty=1.1,
                                         frequency_penalty=0.0,
                                         presence_penalty=0.0,
                                         temperature=0.7, top_k=40, top_p=0.95,
                                         min_p=0.05, seed=None)
                    )
                    # The FakeEvent returns True on the second is_set() call,
                    # so after at least one token is generated, it should exit
                    assert reason == "shutdown", \
                        f"Expected 'shutdown' but got '{reason}'"
                finally:
                    sluice.server.SHUTTING_DOWN = original
        finally:
            llama_cpp.llama_token_eos = orig_eos
            llama_cpp.llama_token_to_piece = orig_ttp

    def test_low_level_stream_step_checks_shutdown(self):
        """low_level_stream_step returns 'shutdown' when event is set."""
        SHUTTING_DOWN.set()
        try:
            piece, ntid, finish = low_level_stream_step(
                sid=1, n_cur=0, last_tokens=[1, 2, 3],
                request=MagicMock(), budget=999, prev_ntid=42
            )
            assert piece is None
            assert finish == "shutdown"
        finally:
            SHUTTING_DOWN.clear()

    def test_low_level_stream_step_normal_when_not_set(self):
        """low_level_stream_step proceeds normally when event is not set."""
        import llama_cpp
        SHUTTING_DOWN.clear()

        # Mock EOS to avoid early stop
        orig_eos = llama_cpp.llama_token_eos
        llama_cpp.llama_token_eos = MagicMock(return_value=-1)

        # Mock token_to_piece
        import ctypes
        orig_ttp = llama_cpp.llama_token_to_piece
        buf = ctypes.create_string_buffer(b"hi")
        llama_cpp.llama_token_to_piece = MagicMock(return_value=2)

        try:
            piece, ntid, finish = low_level_stream_step(
                sid=0, n_cur=0, last_tokens=[1, 2, 3],
                request=MagicMock(), budget=999, prev_ntid=42
            )
            # Should NOT be shutdown
            assert finish is None or finish != "shutdown", \
                f"Unexpected finish reason: {finish}"
        finally:
            llama_cpp.llama_token_eos = orig_eos
            llama_cpp.llama_token_to_piece = orig_ttp
