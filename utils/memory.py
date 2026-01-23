"""
GPU Memory Management Utilities
===============================

Provides context managers and utilities for proper GPU memory cleanup
to prevent memory leaks in the AUTO-ROTO pipeline.

Usage:
    # Simple cleanup
    from utils.memory import clear_gpu_memory
    clear_gpu_memory()

    # Context manager for automatic cleanup
    from utils.memory import gpu_memory_context
    with gpu_memory_context():
        # GPU operations here
        pass  # Memory cleaned up automatically

    # Model lifecycle management
    from utils.memory import ModelContextManager
    with ModelContextManager() as manager:
        model = manager.load_model(SomeModel, "path/to/weights")
        # Use model
        pass  # Model released and memory cleaned automatically
"""

import gc
import logging
from contextlib import contextmanager
from typing import Optional, Any, Callable, TypeVar

logger = logging.getLogger("AutoRoto.Memory")

T = TypeVar('T')


def clear_gpu_memory(verbose: bool = False) -> None:
    """
    Clear GPU memory and run garbage collection.

    Call this between pipeline stages to prevent memory accumulation.
    Safe to call even if CUDA is not available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if verbose:
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                logger.debug(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Error clearing GPU memory: {e}")

    gc.collect()

    if verbose:
        logger.debug("Memory cleanup completed")


def get_gpu_memory_stats() -> dict:
    """
    Get current GPU memory statistics.

    Returns:
        Dictionary with memory stats, or empty dict if CUDA unavailable.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'reserved_gb': torch.cuda.memory_reserved() / 1e9,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
                'device': torch.cuda.get_device_name(),
            }
    except ImportError:
        pass
    return {}


@contextmanager
def gpu_memory_context(cleanup_on_enter: bool = False, verbose: bool = False):
    """
    Context manager that clears GPU memory on exit.

    Args:
        cleanup_on_enter: If True, also clear memory before entering
        verbose: If True, log memory statistics

    Usage:
        with gpu_memory_context():
            # GPU operations
            model = load_model()
            result = model(input)
        # Memory automatically cleaned here
    """
    if cleanup_on_enter:
        clear_gpu_memory(verbose=verbose)

    try:
        yield
    finally:
        clear_gpu_memory(verbose=verbose)


class GPUMemoryManager:
    """
    Manager for tracking and cleaning up GPU memory.

    Can be used as a context manager or instantiated for manual control.
    Tracks peak memory usage for profiling.
    """

    def __init__(self, verbose: bool = False, auto_cleanup: bool = True):
        self.verbose = verbose
        self.auto_cleanup = auto_cleanup
        self._peak_memory = 0.0
        self._initial_memory = 0.0

    def __enter__(self) -> "GPUMemoryManager":
        self._initial_memory = self._get_allocated_memory()
        if self.verbose:
            logger.debug(f"Entering GPU context (initial: {self._initial_memory:.2f}GB)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.auto_cleanup:
            self.cleanup()

        if self.verbose:
            logger.debug(f"Exiting GPU context (peak: {self._peak_memory:.2f}GB)")

        return False  # Don't suppress exceptions

    def _get_allocated_memory(self) -> float:
        """Get current allocated GPU memory in GB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1e9
        except ImportError:
            pass
        return 0.0

    def update_peak(self) -> None:
        """Update peak memory tracking."""
        current = self._get_allocated_memory()
        self._peak_memory = max(self._peak_memory, current)

    def cleanup(self) -> None:
        """Manually trigger memory cleanup."""
        clear_gpu_memory(verbose=self.verbose)

    @property
    def peak_memory_gb(self) -> float:
        """Get peak memory usage in GB."""
        return self._peak_memory

    @property
    def memory_increase_gb(self) -> float:
        """Get memory increase since context entry."""
        return self._get_allocated_memory() - self._initial_memory


class ModelContextManager:
    """
    Context manager for managing model lifecycle.

    Automatically releases models and cleans GPU memory when exiting.
    Supports registering multiple models for cleanup.

    Usage:
        with ModelContextManager() as manager:
            depth_model = manager.register(DepthEstimator(device='cuda'))
            vitmatte = manager.register(ViTMatteRefiner())
            # Use models...
        # All models released and memory cleaned
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._models: list = []
        self._memory_manager = GPUMemoryManager(verbose=verbose)

    def __enter__(self) -> "ModelContextManager":
        self._memory_manager.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Release all registered models
        for model in reversed(self._models):
            self._release_model(model)

        self._models.clear()
        self._memory_manager.__exit__(exc_type, exc_val, exc_tb)
        return False

    def register(self, model: T) -> T:
        """
        Register a model for automatic cleanup.

        Args:
            model: Model instance to manage

        Returns:
            The same model instance (for chaining)
        """
        self._models.append(model)
        return model

    def _release_model(self, model: Any) -> None:
        """Release a model and its resources."""
        try:
            # Try common release methods
            if hasattr(model, 'release') and callable(model.release):
                model.release()
                if self.verbose:
                    logger.debug(f"Released model via release(): {type(model).__name__}")
            elif hasattr(model, 'close') and callable(model.close):
                model.close()
                if self.verbose:
                    logger.debug(f"Released model via close(): {type(model).__name__}")

            # Move model to CPU to free GPU memory
            if hasattr(model, 'to') and callable(model.to):
                try:
                    model.to('cpu')
                except Exception:
                    pass

            # Delete model attribute if present
            if hasattr(model, 'model'):
                del model.model

        except Exception as e:
            logger.warning(f"Error releasing model {type(model).__name__}: {e}")


class CleanupMixin:
    """
    Mixin class that adds automatic cleanup behavior to model classes.

    Add this as a base class to get automatic __del__ and release() methods.

    Usage:
        class MyModel(CleanupMixin):
            def __init__(self):
                self.model = load_something()

            def _cleanup(self):
                # Custom cleanup logic
                if hasattr(self, 'model'):
                    del self.model
    """

    _cleanup_called: bool = False

    def __del__(self):
        """Destructor that ensures cleanup is called."""
        if not getattr(self, '_cleanup_called', False):
            self.release()

    def release(self) -> None:
        """
        Release resources held by this instance.

        Override _cleanup() for custom cleanup logic.
        """
        if self._cleanup_called:
            return

        self._cleanup_called = True

        # Call custom cleanup if defined
        if hasattr(self, '_cleanup') and callable(self._cleanup):
            try:
                self._cleanup()
            except Exception as e:
                logger.warning(f"Error in cleanup for {type(self).__name__}: {e}")

        # Clear GPU memory
        clear_gpu_memory()

    def _cleanup(self) -> None:
        """
        Override this method to add custom cleanup logic.

        This is called by release() before GPU memory is cleared.
        """
        pass
