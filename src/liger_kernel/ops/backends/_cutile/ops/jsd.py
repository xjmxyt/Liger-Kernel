from ._spec import PatchSpec

try:
    from tilegym.suites.liger.cutile.jsd import JSDFunction as _LigerJSDFunction
    _IMPORT_ERROR = None
except ImportError as exc:
    _LigerJSDFunction = None
    _IMPORT_ERROR = exc


def patches() -> list[PatchSpec]:
    """Return PatchSpecs for the JSD op, or raise ImportError if unavailable."""
    if _IMPORT_ERROR is not None:
        raise ImportError("tilegym JSD backend unavailable") from _IMPORT_ERROR
    return [
        PatchSpec(
            module_path="liger_kernel.transformers.jsd",
            attr_name="LigerJSDFunction",
            replacement=_LigerJSDFunction,
        )
    ]
