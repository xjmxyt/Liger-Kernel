"""CuTile operator adapters — aggregates patch specs from all op modules.

To add a new CuTile op:
  1. Create ops/<op_name>.py that imports the tilegym class and defines:
       PATCHES = [("liger_kernel.transformers.<op>", "LigerXxxFunction", LigerXxxFunction)]
       IMPORT_ERROR = None   # or the ImportError if unavailable
  2. Add one line here:
       from .<op_name> import PATCHES as _<OP>_PATCHES, IMPORT_ERROR as _<OP>_ERROR

No changes to _cutile/__init__.py are ever needed.
"""

from .fused_linear_jsd import IMPORT_ERROR as _FLJSD_ERROR
from .fused_linear_jsd import LigerFusedLinearJSDFunction
from .fused_linear_jsd import PATCHES as _FLJSD_PATCHES
from .jsd import IMPORT_ERROR as _JSD_ERROR
from .jsd import LigerJSDFunction
from .jsd import PATCHES as _JSD_PATCHES

# --- add new op patches here, one line each ---
ALL_PATCHES = _JSD_PATCHES + _FLJSD_PATCHES

TILEGYM_AVAILABLE = bool(ALL_PATCHES)
_IMPORT_ERRORS = [e for e in [_JSD_ERROR, _FLJSD_ERROR] if e is not None]

__all__ = [
    "LigerJSDFunction",
    "LigerFusedLinearJSDFunction",
    "TILEGYM_AVAILABLE",
    "ALL_PATCHES",
]
