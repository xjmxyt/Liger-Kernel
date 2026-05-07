try:
    from tilegym.suites.liger.cutile.jsd import JSDFunction as LigerJSDFunction

    _TILEGYM_IMPORT_ERROR = None
    _TILEGYM_AVAILABLE = True
except ImportError as exc:
    LigerJSDFunction = None
    _TILEGYM_IMPORT_ERROR = exc
    _TILEGYM_AVAILABLE = False


def _require_tilegym() -> None:
    if not _TILEGYM_AVAILABLE:
        raise ImportError(
            "tilegym cutile backend is not available. Install it from the ocean repo."
        ) from _TILEGYM_IMPORT_ERROR
