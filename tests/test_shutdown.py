"""Tests for threading.Event-based shutdown mechanism."""
import threading
import signal
import pytest
from unittest.mock import patch, MagicMock
import sluice.server
from sluice.server import SHUTDOWN_EVENT, signal_handler, low_level_generate, low_level_stream_step


class TestShutdownEvent:
    """Verify SHUTDOWN_EVENT is a proper threading.Event with correct API usage."""

    def test_is_threading_event(self):
        """SHUTDOWN_EVENT is a threading.Event instance."""
        assert isinstance(SHUTDOWN_EVENT, threading.Event)

    def test_not_set_by_default(self):
        """Event is clear on module load (no shutdown in progress)."""
        assert SHUTDOWN_EVENT.is_set() is False

    def test_set_makes_is_set_true(self):
        """After .set(), .is_set() returns True."""
        SHUTDOWN_EVENT.set()
        try:
            assert SHUTDOWN_EVENT.is_set() is True
        finally:
            SHUTDOWN_EVENT.clear()

    def test_clear_makes_is_set_false(self):
        """After .clear(), .is_set() returns False."""
        SHUTDOWN_EVENT.set()
        SHUTDOWN_EVENT.clear()
        assert SHUTDOWN_EVENT.is_set() is False

    def test_signal_handler_sets_event(self):
        """signal_handler calls SHUTDOWN_EVENT.set() on any signal."""
        # Ensure clean state
        SHUTDOWN_EVENT.clear()
        assert not SHUTDOWN_EVENT.is_set()

        signal_handler(signal.SIGINT, None)
        assert SHUTDOWN_EVENT.is_set()

        # Clean up
        SHUTDOWN_EVENT.clear()

    def test_sigterm_also_sets_event(self):
        """SIGTERM also triggers shutdown via signal_handler."""
        SHUTDOWN_EVENT.clear()
        signal_handler(signal.SIGTERM, None)
        assert SHUTDOWN_EVENT.is_set()
        SHUTDOWN_EVENT.clear()


class TestGenerationLoopShutdown:
    """Verify the generation loop checks SHUTDOWN_EVENT.is_set() and exits gracefully."""

    def setup_method(self):
        # Ensure clean state before each test
        SHUTDOWN_EVENT.clear()

    def teardown_method(self):
        SHUTDOWN_EVENT.clear()

    def test_low_level_generate_exits_on_shutdown(self):
        """When SHUTDOWN_EVENT is set, low_level_generate returns 'shutdown' reason."""
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
            # run until SHUTDOWN_EVENT.is_set() is True
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
                call_count = [0]

                class FakeEvent:
                    """Simulate SHUTDOWN_EVENT becoming set mid-generation."""
                    def is_set(self):
                        call_count[0] += 1
                        return call_count[0] > 1  # not set on first check

                original = sluice.server.SHUTDOWN_EVENT
                sluice.server.SHUTDOWN_EVENT = FakeEvent()
                try:
                    # Now low_level_generate will read our FakeEvent
                    text, n_p, n_g, reason = sluice.server.low_level_generate(
                        sid=1, tokens=[1, 2, 3], max_tokens=10,
                        budget=budget,
                        request=MagicMock(stop=None, repeat_penalty=1.1,
                                         frequency_penalty=0.0,
                                         presence_penalty=0.0,
                                         temperature=0.7, top_k=40, top_p=0.95,
                                         min_p=0.05, seed=None)
                    )
                    assert reason == "shutdown"
                finally:
                    sluice.server.SHUTDOWN_EVENT = original
        finally:
            llama_cpp.llama_token_eos = orig_eos
            llama_cpp.llama_token_to_piece = orig_ttp

    def test_low_level_stream_step_checks_shutdown(self):
        """low_level_stream_step returns 'shutdown' when event is set."""
        SHUTDOWN_EVENT.set()
        try:
            piece, ntid, finish = low_level_stream_step(
                sid=1, n_cur=0, last_tokens=[1, 2, 3],
                request=MagicMock(), budget=999, prev_ntid=42
            )
            assert piece is None
            assert finish == "shutdown"
        finally:
            SHUTDOWN_EVENT.clear()
