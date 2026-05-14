from ._spec import PatchSpec

try:
    from tilegym.suites.liger.cutile.layer_norm import LayerNormCuTileFunction as _LigerLayerNormFunction

    _IMPORT_ERROR = None
except ImportError as exc:
    _LigerLayerNormFunction = None
    _IMPORT_ERROR = exc


def patches() -> list[PatchSpec]:
    """Return PatchSpecs for the layer_norm op, or raise ImportError if unavailable."""
    if _IMPORT_ERROR is not None:
        raise ImportError("tilegym layer_norm backend unavailable") from _IMPORT_ERROR
    return [
        PatchSpec(
            module_path="liger_kernel.transformers.layer_norm",
            attr_name="LigerLayerNormFunction",
            replacement=_LigerLayerNormFunction,
        )
    ]
