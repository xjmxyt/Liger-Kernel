try:
    from tilegym.suites.liger.cutile.jsd import JSDFunction as LigerJSDFunction

    # Each entry: (transformers_module_path, attr_name, replacement_class).
    # The central tilegym_enabled() reads PATCHES from every op module so it
    # never needs to know about individual ops.
    PATCHES = [("liger_kernel.transformers.jsd", "LigerJSDFunction", LigerJSDFunction)]
    IMPORT_ERROR = None
except ImportError as exc:
    LigerJSDFunction = None
    PATCHES = []
    IMPORT_ERROR = exc
