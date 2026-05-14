from ._spec import PatchSpec

try:
    from tilegym.suites.liger.cutile.geglu import GEGLUCuTileFunction as _LigerGELUMulFunction

    _IMPORT_ERROR = None
except ImportError as exc:
    _LigerGELUMulFunction = None
    _IMPORT_ERROR = exc


def patches() -> list[PatchSpec]:
    """Return PatchSpecs for the geglu op, or raise ImportError if unavailable."""
    if _IMPORT_ERROR is not None:
        raise ImportError("tilegym geglu backend unavailable") from _IMPORT_ERROR
    return [
        PatchSpec(
            module_path="liger_kernel.transformers.geglu",
            attr_name="LigerGELUMulFunction",
            replacement=_LigerGELUMulFunction,
        )
    ]
