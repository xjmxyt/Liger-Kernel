from . import PatchSpec

try:
    from tilegym.suites.liger.cutile.fused_linear_jsd import (
        FusedLinearJSDFunction as _LigerFusedLinearJSDFunction,
    )
    _IMPORT_ERROR = None
except ImportError as exc:
    _LigerFusedLinearJSDFunction = None
    _IMPORT_ERROR = exc


def patches() -> list[PatchSpec]:
    """Return PatchSpecs for the fused-linear-JSD op, or raise ImportError if unavailable."""
    if _IMPORT_ERROR is not None:
        raise ImportError("tilegym fused-linear-JSD backend unavailable") from _IMPORT_ERROR
    return [
        PatchSpec(
            module_path="liger_kernel.transformers.fused_linear_jsd",
            attr_name="LigerFusedLinearJSDFunction",
            replacement=_LigerFusedLinearJSDFunction,
        )
    ]
