from ._spec import PatchSpec

try:
    from tilegym.suites.liger.cutile.cross_entropy import (
        CrossEntropyCuTileFunction as _LigerCrossEntropyFunction,
    )

    _IMPORT_ERROR = None
except ImportError as exc:
    _LigerCrossEntropyFunction = None
    _IMPORT_ERROR = exc


def patches() -> list[PatchSpec]:
    """Return PatchSpecs for the cross_entropy op, or raise ImportError if unavailable."""
    if _IMPORT_ERROR is not None:
        raise ImportError("tilegym cross_entropy backend unavailable") from _IMPORT_ERROR
    return [
        PatchSpec(
            module_path="liger_kernel.transformers.cross_entropy",
            attr_name="LigerCrossEntropyFunction",
            replacement=_LigerCrossEntropyFunction,
        )
    ]
