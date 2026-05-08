"""CuTile operator adapters — aggregates patch specs from all op modules.

To add a new CuTile op:
  1. Create ops/<op_name>.py that defines a patches() function:
       def patches():
           if <unavailable>:
               raise ImportError(...) from _error
           return [("liger_kernel.transformers.<op>", "LigerXxxFunction", cls)]
  2. Add two lines here:
       from .<op_name> import patches as _<op>_patches
       and append _<op>_patches to _OP_PATCH_FNS.

No changes to _cutile/__init__.py are ever needed.
"""

from .fused_linear_jsd import patches as _fljsd_patches
from .jsd import patches as _jsd_patches

# To add a new op: import its patches() function and append it here.
_OP_PATCH_FNS = [_jsd_patches, _fljsd_patches]

ALL_PATCHES = []
_IMPORT_ERRORS = []
for _fn in _OP_PATCH_FNS:
    try:
        ALL_PATCHES.extend(_fn())
    except ImportError as _e:
        _IMPORT_ERRORS.append(_e)

TILEGYM_AVAILABLE = bool(ALL_PATCHES)

__all__ = ["TILEGYM_AVAILABLE", "ALL_PATCHES"]
