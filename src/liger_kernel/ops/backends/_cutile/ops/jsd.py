try:
    from tilegym.suites.liger.cutile.jsd import JSDFunction as LigerJSDFunction

    _TILEGYM_IMPORT_ERROR = None
    _TILEGYM_AVAILABLE = True
except ImportError as exc:
    LigerJSDFunction = None
    _TILEGYM_IMPORT_ERROR = exc
    _TILEGYM_AVAILABLE = False
