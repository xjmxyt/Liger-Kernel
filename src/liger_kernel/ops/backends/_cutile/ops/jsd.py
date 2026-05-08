try:
    from tilegym.suites.liger.cutile.jsd import JSDFunction as _LigerJSDFunction
    _IMPORT_ERROR = None
except ImportError as exc:
    _LigerJSDFunction = None
    _IMPORT_ERROR = exc


def patches():
    """Return [(module_path, attr_name, replacement_class)] for the JSD op.

    Raises ImportError if tilegym is not installed.
    """
    if _IMPORT_ERROR is not None:
        raise ImportError("tilegym JSD backend unavailable") from _IMPORT_ERROR
    return [("liger_kernel.transformers.jsd", "LigerJSDFunction", _LigerJSDFunction)]
