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
