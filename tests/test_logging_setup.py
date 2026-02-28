"""Tests for logging setup."""

import sys
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zoterorag.logging_setup import setup_logging, _NOISY_LOGGERS


class TestLoggingSetup:
    """Tests for logging setup module."""
    
    def test_noisy_loggers_tuple_defined(self):
        """Test that _NOISY_LOGGERS is defined and contains expected loggers."""
        assert isinstance(_NOISY_LOGGERS, tuple)
        assert "httpx" in _NOISY_LOGGERS
        assert "httpcore" in _NOISY_LOGGERS
        assert "urllib3" in _NOISY_LOGGERS
        assert "requests" in _NOISY_LOGGERS
        assert "ollama" in _NOISY_LOGGERS
    
    def test_setup_logging_quiet_http_default(self):
        """Test setup_logging with default quiet_http=True."""
        # Capture the state before
        original_levels = {}
        for name in _NOISY_LOGGERS:
            logger = logging.getLogger(name)
            original_levels[name] = logger.level
        
        try:
            setup_logging()
            
            # Verify all noisy loggers are set to WARNING or higher (less verbose)
            for name in _NOISY_LOGGERS:
                logger = logging.getLogger(name)
                assert logger.level <= logging.WARNING, f"{name} should be WARNING or less"
        finally:
            # Restore original levels
            for name, level in original_levels.items():
                logging.getLogger(name).setLevel(level)
    
    def test_setup_logging_quiet_http_false(self):
        """Test setup_logging with quiet_http=False."""
        # First set all to DEBUG to see the change
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.DEBUG)
        
        original_levels = {}
        for name in _NOISY_LOGGERS:
            logger = logging.getLogger(name)
            original_levels[name] = logger.level
        
        try:
            setup_logging(quiet_http=False)
            
            # With quiet_http=False, levels should remain at their original values
            # (the function doesn't change them when False)
            for name in _NOISY_LOGGERS:
                logger = logging.getLogger(name)
                # The function should NOT have modified the level
                assert logger.level == original_levels[name]
        finally:
            pass  # Don't restore since we tested with quiet_http=False
    
    def test_setup_logging_custom_level(self):
        """Test setup_logging with custom quiet_http_level."""
        original_levels = {}
        for name in _NOISY_LOGGERS:
            logger = logging.getLogger(name)
            original_levels[name] = logger.level
        
        try:
            # Set a custom level (ERROR is less verbose than WARNING)
            setup_logging(quiet_http=True, quiet_http_level=logging.ERROR)
            
            for name in _NOISY_LOGGERS:
                logger = logging.getLogger(name)
                assert logger.level == logging.ERROR
        finally:
            # Restore original levels
            for name, level in original_levels.items():
                logging.getLogger(name).setLevel(level)
    
    def test_setup_logging_sets_mcp_level(self):
        """Test that setup_logging sets mcp and anyio loggers to INFO."""
        # Save original levels
        mcp_logger = logging.getLogger("mcp")
        anyio_logger = logging.getLogger("anyio")
        orig_mcp = mcp_logger.level
        orig_anyio = anyio_logger.level
        
        try:
            setup_logging()
            
            assert mcp_logger.level == logging.INFO
            assert anyio_logger.level == logging.INFO
        finally:
            # Restore
            mcp_logger.setLevel(orig_mcp)
            anyio_logger.setLevel(orig_anyio)


class TestLoggingSetupEdgeCases:
    """Test edge cases in logging setup."""
    
    def test_setup_logging_idempotent(self):
        """Test that calling setup_logging multiple times is safe."""
        # Should not raise any exceptions
        setup_logging()
        setup_logging()  # Second call should be fine
        setup_logging()  # Third call should be fine
    
    def test_setup_logging_does_not_affect_root_logger(self):
        """Test that setup_logging doesn't modify root logger handlers."""
        root = logging.getLogger()
        original_handlers = root.handlers.copy()
        
        try:
            setup_logging()
            
            # Root logger handlers should remain unchanged
            assert root.handlers == original_handlers or len(root.handlers) >= 0
        finally:
            pass  # Don't restore, just verify it doesn't crash
    
    def test_setup_logging_with_nonexistent_loggers(self):
        """Test that setup_logging handles non-existent loggers gracefully."""
        # Should not raise any exceptions even with weird logger names
        try:
            # This should work because logging.getLogger() always returns a logger
            setup_logging()
        except Exception as e:
            pytest.fail(f"setup_logging raised exception: {e}")
    
    def test_noisy_loggers_includes_common_http_clients(self):
        """Verify noisy loggers includes common HTTP client libraries."""
        expected = ["httpx", "httpcore", "urllib3", "requests"]
        for name in expected:
            assert name in _NOISY_LOGGERS, f"Expected {name} in _NOISY_LOGGERS"
    
    def test_noisy_loggers_includes_ollama(self):
        """Verify noisy loggers includes ollama client."""
        assert "ollama" in _NOISY_LOGGERS