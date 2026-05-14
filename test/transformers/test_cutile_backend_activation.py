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


def test_cutile_backend_activation_patches_layer_norm_cross_entropy_geglu():
    result = _run_python(
        """
import os
os.environ["CUTILE_BACKEND"] = "1"

try:
    import liger_kernel.transformers.layer_norm as ln_layer
    import liger_kernel.transformers.cross_entropy as ce_layer
    import liger_kernel.transformers.geglu as geglu_layer
    import liger_kernel.transformers
except ImportError as exc:
    assert "tilegym" in str(exc).lower()
else:
    assert ln_layer.LigerLayerNormFunction.__module__ == "tilegym.suites.liger.cutile.layer_norm"
    assert ce_layer.LigerCrossEntropyFunction.__module__ == "tilegym.suites.liger.cutile.cross_entropy"
    assert geglu_layer.LigerGELUMulFunction.__module__ == "tilegym.suites.liger.cutile.geglu"
"""
    )

    assert result.returncode == 0, result.stderr


def test_tilegym_enabled_context_manager_layer_norm_cross_entropy_geglu():
    result = _run_python(
        """
import liger_kernel.transformers.layer_norm as ln_layer
import liger_kernel.transformers.cross_entropy as ce_layer
import liger_kernel.transformers.geglu as geglu_layer

orig_ln = ln_layer.LigerLayerNormFunction
orig_ce = ce_layer.LigerCrossEntropyFunction
orig_geglu = geglu_layer.LigerGELUMulFunction

try:
    from liger_kernel.ops.backends._cutile import tilegym_enabled
    with tilegym_enabled():
        assert ln_layer.LigerLayerNormFunction is not orig_ln
        assert ce_layer.LigerCrossEntropyFunction is not orig_ce
        assert geglu_layer.LigerGELUMulFunction is not orig_geglu
    assert ln_layer.LigerLayerNormFunction is orig_ln, "layer_norm should be restored"
    assert ce_layer.LigerCrossEntropyFunction is orig_ce, "cross_entropy should be restored"
    assert geglu_layer.LigerGELUMulFunction is orig_geglu, "geglu should be restored"
except ImportError as exc:
    assert "tilegym" in str(exc).lower()
"""
    )

    assert result.returncode == 0, result.stderr
