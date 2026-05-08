try:
    from tilegym.suites.liger.cutile.fused_linear_jsd import FusedLinearJSDFunction as LigerFusedLinearJSDFunction

    _TILEGYM_IMPORT_ERROR = None
    _TILEGYM_AVAILABLE = True
except ImportError as exc:
    LigerFusedLinearJSDFunction = None
    _TILEGYM_IMPORT_ERROR = exc
    _TILEGYM_AVAILABLE = False
