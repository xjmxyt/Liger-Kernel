"""CuTile operator adapters — aggregates patch specs from all op modules.

To add a new CuTile op:
  1. Create ops/<op_name>.py that imports PatchSpec from this module and
     defines a patches() function:
       from . import PatchSpec
       def patches():
           if <unavailable>:
               raise ImportError(...) from _error
           return [PatchSpec(module_path=..., attr_name=..., replacement=cls)]
  2. Add two lines here:
       from .<op_name> import patches as _<op>_patches
       and append _<op>_patches to _OP_PATCH_FNS.

No changes to _cutile/__init__.py are ever needed.
"""

from typing import NamedTuple


class PatchSpec(NamedTuple):
    """Declares one attribute to swap in the transformers layer.

    module_path:  importable path of the transformers-layer module to patch
                  (e.g. "liger_kernel.transformers.jsd")
    attr_name:    name of the attribute to replace
                  (e.g. "LigerJSDFunction")
    replacement:  the CuTile class that replaces it
    """

    module_path: str
    attr_name: str
    replacement: type


from .fused_linear_jsd import patches as _fljsd_patches  # noqa: E402
from .jsd import patches as _jsd_patches  # noqa: E402

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

__all__ = ["PatchSpec", "TILEGYM_AVAILABLE", "ALL_PATCHES"]
