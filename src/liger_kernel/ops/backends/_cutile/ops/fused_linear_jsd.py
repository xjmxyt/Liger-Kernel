try:
    from tilegym.suites.liger.cutile.fused_linear_jsd import (
        FusedLinearJSDFunction as _LigerFusedLinearJSDFunction,
    )
    _IMPORT_ERROR = None
except ImportError as exc:
    _LigerFusedLinearJSDFunction = None
    _IMPORT_ERROR = exc


def patches():
    """Return [(module_path, attr_name, replacement_class)] for the fused-linear-JSD op.

    Raises ImportError if tilegym is not installed.
    """
    if _IMPORT_ERROR is not None:
        raise ImportError("tilegym fused-linear-JSD backend unavailable") from _IMPORT_ERROR
    return [
        (
            "liger_kernel.transformers.fused_linear_jsd",
            "LigerFusedLinearJSDFunction",
            _LigerFusedLinearJSDFunction,
        )
    ]
