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


def _patch_tilegym_classes():
    """Replace LigerJSDFunction and LigerFusedLinearJSDFunction with CuTile classes.

    Patches the class bindings in liger_kernel.transformers.jsd and
    liger_kernel.transformers.fused_linear_jsd so that LigerJSD and
    LigerFusedLinearJSD automatically use tilegym's public CuTile classes.

    Raises ImportError if tilegym is not installed.
    """
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
        raise ImportError(
            "tilegym cutile backend is not available. Install it from the ocean repo."
        ) from (_JSD_ERROR or _FLJSD_ERROR)

    _jsd_layer.LigerJSDFunction = LigerJSDFunction
    _fljsd_layer.LigerFusedLinearJSDFunction = LigerFusedLinearJSDFunction


@contextlib.contextmanager
def tilegym_enabled():
    """Temporarily replace JSD Function classes with CuTile implementations.

    Patches LigerJSDFunction and LigerFusedLinearJSDFunction in the
    transformers layer. All modules are fully loaded by the time a with-block
    is entered, so there is no circular import risk. Uses tilegym's public
    JSDFunction / FusedLinearJSDFunction classes. Restores originals on exit.
    """
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
        raise ImportError(
            "tilegym cutile backend is not available. Install it from the ocean repo."
        ) from (_JSD_ERROR or _FLJSD_ERROR)

    orig = {
        "jsd_fn": _jsd_layer.LigerJSDFunction,
        "fljsd_fn": _fljsd_layer.LigerFusedLinearJSDFunction,
    }

    _jsd_layer.LigerJSDFunction = LigerJSDFunction
    _fljsd_layer.LigerFusedLinearJSDFunction = LigerFusedLinearJSDFunction

    try:
        yield
    finally:
        _jsd_layer.LigerJSDFunction = orig["jsd_fn"]
        _fljsd_layer.LigerFusedLinearJSDFunction = orig["fljsd_fn"]
