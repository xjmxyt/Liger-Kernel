"""
CuTile backend adapters for Liger-Kernel.

Two ways to enable the CuTile JSD kernels:

1. Environment variable (zero code change):
       CUTILE_BACKEND=1 python train.py
   Checked in liger_kernel/transformers/__init__.py after all submodules are
   loaded — patches LigerJSDFunction / LigerFusedLinearJSDFunction with
   tilegym's public classes. No circular import risk.

2. Context manager (fine-grained control):
       from liger_kernel.ops.backends._cutile import tilegym_enabled
       with tilegym_enabled():
           loss = model(...)
           loss.backward()
"""

import contextlib
import importlib
import threading

# tilegym_enabled() is typically entered once per forward pass inside a
# benchmark hot-loop (rep=100+). Without caching, every entry would re-execute
# imports and availability checks. Python already caches modules in sys.modules,
# but the repeated attribute lookups still add up. This list is populated on
# the first successful _get_tilegym_refs() call so that all subsequent entries
# are a single global read.
#
# Note: ALL_PATCHES is computed at ops/__init__.py import time. If tilegym is
# not installed when that module is first imported it will remain empty for the
# lifetime of the process — retrying _get_tilegym_refs() after a lazy install
# will NOT pick up the newly installed package.
#
# Each element is a (module_obj, attr_name, replacement_class) triple resolved
# from the PatchSpecs returned by each ops/<op>.py patches() function. The
# central code here is completely unaware of individual ops.
_tilegym_refs = None
_tilegym_refs_lock = threading.Lock()


def _get_tilegym_refs():
    """Resolve and cache (module, attr, replacement) triples for every CuTile op.

    Reads ALL_PATCHES from _cutile/ops/__init__.py (which aggregates PATCHES
    from every individual op module) and resolves the module path strings into
    live module objects. The result is cached so imports run exactly once.

    Thread-safe: uses a double-checked lock so concurrent first-callers do not
    both initialise the cache.

    Raises ImportError if no CuTile ops are available.
    """
    global _tilegym_refs
    if _tilegym_refs is not None:
        return _tilegym_refs

    with _tilegym_refs_lock:
        # Second check inside the lock — another thread may have initialised
        # while we were waiting.
        if _tilegym_refs is not None:
            return _tilegym_refs

        from liger_kernel.ops.backends._cutile.ops import ALL_PATCHES, _IMPORT_ERRORS

        if not ALL_PATCHES:
            # Do NOT set _tilegym_refs here: ALL_PATCHES is frozen at import
            # time (see module docstring), but leaving _tilegym_refs as None
            # keeps the error path consistent and avoids caching a broken state.
            raise ImportError(
                "tilegym cutile backend is not available. Install it from the ocean repo."
            ) from (_IMPORT_ERRORS[0] if _IMPORT_ERRORS else None)

        # Resolve module path strings to live module objects once.
        _tilegym_refs = [
            (importlib.import_module(spec.module_path), spec.attr_name, spec.replacement)
            for spec in ALL_PATCHES
        ]

    return _tilegym_refs


def _patch_tilegym_classes():
    """Replace Liger*Function classes with CuTile implementations.

    Iterates over every patch declared in the ops layer and applies it.
    Adding a new op requires no changes here — only in ops/__init__.py.

    Raises ImportError if tilegym is not installed.
    """
    for layer, attr_name, replacement in _get_tilegym_refs():
        setattr(layer, attr_name, replacement)


@contextlib.contextmanager
def tilegym_enabled():
    """Temporarily replace Liger*Function classes with CuTile implementations.

    Patches every attr declared in the ops layer, yields, then restores
    originals. Safe to call repeatedly — module and class references are
    resolved once and cached by _get_tilegym_refs().

    Adding a new op requires no changes here — only in ops/__init__.py.
    """
    refs = _get_tilegym_refs()
    originals = [(layer, attr_name, getattr(layer, attr_name)) for layer, attr_name, _ in refs]

    for layer, attr_name, replacement in refs:
        setattr(layer, attr_name, replacement)

    try:
        yield
    finally:
        for layer, attr_name, orig in originals:
            setattr(layer, attr_name, orig)
