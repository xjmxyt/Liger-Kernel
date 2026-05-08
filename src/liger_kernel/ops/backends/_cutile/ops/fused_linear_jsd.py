try:
    from tilegym.suites.liger.cutile.fused_linear_jsd import (
        FusedLinearJSDFunction as LigerFusedLinearJSDFunction,
    )

    # Each entry: (transformers_module_path, attr_name, replacement_class).
    # The central tilegym_enabled() reads PATCHES from every op module so it
    # never needs to know about individual ops.
    PATCHES = [
        (
            "liger_kernel.transformers.fused_linear_jsd",
            "LigerFusedLinearJSDFunction",
            LigerFusedLinearJSDFunction,
        )
    ]
    IMPORT_ERROR = None
except ImportError as exc:
    LigerFusedLinearJSDFunction = None
    PATCHES = []
    IMPORT_ERROR = exc
