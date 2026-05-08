import os
import subprocess
import sys

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_python(script: str):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in [str(_REPO_ROOT / "src"), str(_REPO_ROOT), env.get("PYTHONPATH", "")] if path
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_cutile_adapter_import_is_optional_without_activation():
    result = _run_python(
        """
import os
os.environ.pop("CUTILE_BACKEND", None)

import liger_kernel.transformers.jsd as jsd_layer
import liger_kernel.transformers.fused_linear_jsd as fljsd_layer

# Without CUTILE_BACKEND the class bindings stay as the default Liger classes
assert jsd_layer.LigerJSDFunction.__module__ == "liger_kernel.ops.jsd"
assert fljsd_layer.LigerFusedLinearJSDFunction.__module__ == "liger_kernel.ops.fused_linear_jsd"
"""
    )

    assert result.returncode == 0, result.stderr


def test_cutile_backend_activation_patches_transformers_layer():
    result = _run_python(
        """
import os
os.environ["CUTILE_BACKEND"] = "1"

try:
    import liger_kernel.transformers.jsd as jsd_layer
    import liger_kernel.transformers.fused_linear_jsd as fljsd_layer
    # Trigger transformers/__init__.py (which applies the patch)
    import liger_kernel.transformers
except ImportError as exc:
    assert "tilegym" in str(exc).lower()
else:
    # With CUTILE_BACKEND, class bindings should be tilegym's public classes
    assert jsd_layer.LigerJSDFunction.__module__ == "tilegym.suites.liger.cutile.jsd"
    assert fljsd_layer.LigerFusedLinearJSDFunction.__module__ == "tilegym.suites.liger.cutile.fused_linear_jsd"
"""
    )

    assert result.returncode == 0, result.stderr


def test_tilegym_enabled_context_manager():
    result = _run_python(
        """
import liger_kernel.transformers.jsd as jsd_layer
import liger_kernel.transformers.fused_linear_jsd as fljsd_layer

orig_jsd_fn = jsd_layer.LigerJSDFunction
orig_fljsd_fn = fljsd_layer.LigerFusedLinearJSDFunction

try:
    from liger_kernel.ops.backends._cutile import tilegym_enabled
    with tilegym_enabled():
        assert jsd_layer.LigerJSDFunction is not orig_jsd_fn
        assert fljsd_layer.LigerFusedLinearJSDFunction is not orig_fljsd_fn
    assert jsd_layer.LigerJSDFunction is orig_jsd_fn, "should be restored after context"
    assert fljsd_layer.LigerFusedLinearJSDFunction is orig_fljsd_fn
except ImportError as exc:
    assert "tilegym" in str(exc).lower()
"""
    )

    assert result.returncode == 0, result.stderr
