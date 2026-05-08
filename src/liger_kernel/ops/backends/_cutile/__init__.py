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

# Module-level cache: populated on the first _get_tilegym_refs() call so that
# repeated tilegym_enabled() entries (e.g. inside a benchmark hot-loop) pay
# zero import overhead after the first invocation.
_tilegym_refs = None


def _get_tilegym_refs():
    """Resolve and cache tilegym class and module references (runs once).

    Returns a 4-tuple: (jsd_layer, fljsd_layer, LigerJSDFunction, LigerFusedLinearJSDFunction)
    Raises ImportError if tilegym is not installed.
    """
    global _tilegym_refs
    if _tilegym_refs is not None:
        return _tilegym_refs

    import liger_kernel.transformers.fused_linear_jsd as _fljsd_layer
    import liger_kernel.transformers.jsd as _jsd_layer
    from liger_kernel.ops.backends._cutile.ops.fused_linear_jsd import (
        LigerFusedLinearJSDFunction,
        _TILEGYM_AVAILABLE as _FLJSD_AVAILABLE,
        _TILEGYM_IMPORT_ERROR as _FLJSD_ERROR,
    )
    from liger_kernel.ops.backends._cutile.ops.jsd import (
        LigerJSDFunction,
        _TILEGYM_AVAILABLE as _JSD_AVAILABLE,
        _TILEGYM_IMPORT_ERROR as _JSD_ERROR,
    )

    if not (_JSD_AVAILABLE and _FLJSD_AVAILABLE):
        # Do NOT set _tilegym_refs here. If tilegym is installed later in the
        # same process (e.g. lazy pip install in a notebook), the next call
        # will retry the import instead of returning a stale failure.
        raise ImportError(
            "tilegym cutile backend is not available. Install it from the ocean repo."
        ) from (_JSD_ERROR or _FLJSD_ERROR)

    _tilegym_refs = (_jsd_layer, _fljsd_layer, LigerJSDFunction, LigerFusedLinearJSDFunction)
    return _tilegym_refs


def _patch_tilegym_classes():
    """Replace LigerJSDFunction and LigerFusedLinearJSDFunction with CuTile classes.

    Patches the class bindings in liger_kernel.transformers.jsd and
    liger_kernel.transformers.fused_linear_jsd so that LigerJSD and
    LigerFusedLinearJSD automatically use tilegym's public CuTile classes.

    Raises ImportError if tilegym is not installed.
    """
    jsd_layer, fljsd_layer, LigerJSDFunction, LigerFusedLinearJSDFunction = _get_tilegym_refs()
    jsd_layer.LigerJSDFunction = LigerJSDFunction
    fljsd_layer.LigerFusedLinearJSDFunction = LigerFusedLinearJSDFunction


@contextlib.contextmanager
def tilegym_enabled():
    """Temporarily replace JSD Function classes with CuTile implementations.

    Patches LigerJSDFunction and LigerFusedLinearJSDFunction in the
    transformers layer. All modules are fully loaded by the time a with-block
    is entered, so there is no circular import risk. Uses tilegym's public
    JSDFunction / FusedLinearJSDFunction classes. Restores originals on exit.

    Module and class references are resolved once and cached, so entering
    this context manager repeatedly (e.g. inside a benchmark loop) incurs
    no import overhead after the first call.
    """
    jsd_layer, fljsd_layer, LigerJSDFunction, LigerFusedLinearJSDFunction = _get_tilegym_refs()

    orig_jsd_fn = jsd_layer.LigerJSDFunction
    orig_fljsd_fn = fljsd_layer.LigerFusedLinearJSDFunction

    jsd_layer.LigerJSDFunction = LigerJSDFunction
    fljsd_layer.LigerFusedLinearJSDFunction = LigerFusedLinearJSDFunction

    try:
        yield
    finally:
        jsd_layer.LigerJSDFunction = orig_jsd_fn
        fljsd_layer.LigerFusedLinearJSDFunction = orig_fljsd_fn
