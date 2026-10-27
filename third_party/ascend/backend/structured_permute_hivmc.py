"""Lower compiler-owned custom operations before an older hivmc reads them.

Uses the installed CANN optimizer and leaves all SDK files unchanged. This
entrypoint is installed as a temporary hivmc launcher by the Ascend backend.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from structured_permute_ir import lower_private_ops


def main():
    real = os.environ["FLAGTREE_REAL_HIVMC"]
    env = os.environ.copy()
    original = env.pop("FLAGTREE_ORIGINAL_BISHENG_INSTALL_PATH", "")
    if original:
        env["BISHENG_INSTALL_PATH"] = original
    else:
        env.pop("BISHENG_INSTALL_PATH", None)
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        os.execve(real, [real, *args], env)
    source = Path(args[0])
    text = source.read_text()
    if "hivm.hir.custom" not in text or "custom_flagtree_" not in text:
        os.execve(real, [real, *args], env)
    with tempfile.TemporaryDirectory(prefix="flagtree-hivmc-") as tmp:
        generic = Path(tmp) / "kernel.generic.mlir"
        lowered = Path(tmp) / "kernel.calls.mlir"
        checked = Path(tmp) / "kernel.checked.mlir"
        dump = env.get("FLAGTREE_PERMUTE_DUMP_DIR")
        target = None
        if dump:
            target = Path(dump) / str(os.getpid())
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target / "before.mlir")
        cmd = [env["FLAGTREE_CANN_OPT"], str(source), "--mlir-print-op-generic",
               "--mlir-print-local-scope", "-o", str(generic)]
        subprocess.run(cmd, env=env, check=True)
        lowered.write_text(lower_private_ops(generic.read_text()))
        subprocess.run([env["FLAGTREE_CANN_OPT"], str(lowered), "-o", str(checked)], env=env, check=True)
        if target:
            shutil.copy2(checked, target / "after.mlir")
        subprocess.run([real, str(checked), *args[1:]], env=env, check=True)


if __name__ == "__main__":
    main()
