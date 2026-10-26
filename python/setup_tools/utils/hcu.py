# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from pathlib import Path
import re
import subprocess

from python.setup_tools.utils.default import FlagCXRegistrar, configure_flagcx_source

# PR #667 provides the DU Device API bitcode build. Keep this pin specific to
# HCU: the other backends still use their existing FlagCX revisions.
FLAGCX_HCU_URL = "https://github.com/MC952-arch/FlagCX.git"
FLAGCX_HCU_COMMIT = "2ca624ac84ed71bfed259d42400e39f2f6444e72"


def relocate_flagcx():
    configure_flagcx_source("hcu/third_party/flagcx", url=FLAGCX_HCU_URL, commit_id=FLAGCX_HCU_COMMIT,
                            source_dir=os.environ.get("FLAGCX_SOURCE_DIR"))


def _device_arch(sdk):
    requested = os.environ.get("DEVICE_BITCODE_ARCH", "").strip()
    if requested and requested != "native":
        if not re.fullmatch(r"gfx[0-9a-f]+", requested):
            raise RuntimeError(f"Invalid HCU DEVICE_BITCODE_ARCH: {requested!r}")
        return requested

    for tool in (sdk / "aillvm/bin/amdgpu-arch", sdk / "llvm/bin/amdgpu-arch", sdk / "bin/amdgpu-arch"):
        if tool.is_file():
            result = subprocess.run([str(tool)], text=True, capture_output=True, check=False)
            arches = set(re.findall(r"\bgfx[0-9a-f]+\b", result.stdout))
            if result.returncode == 0 and len(arches) == 1:
                return arches.pop()
    try:
        import torch
        if torch.version.hip and torch.cuda.is_available():
            arches = {
                torch.cuda.get_device_properties(i).gcnArchName.split(":")[0]
                for i in range(torch.cuda.device_count())
            }
            if len(arches) == 1:
                return arches.pop()
    except (ImportError, AttributeError, RuntimeError):
        pass
    raise RuntimeError("Cannot determine the HCU GPU architecture. Set DEVICE_BITCODE_ARCH explicitly "
                       "(for example gfx936 on BW1000); cross builds must name the deployment GPU.")


def _toolchain():
    sdk = Path(os.environ.get("DEVICE_HOME") or os.environ.get("HIP_PATH") or "/opt/dtk").expanduser().resolve()
    if not (sdk / "bin/hipcc").is_file():
        raise RuntimeError(f"HCU FlagCX requires {sdk / 'bin/hipcc'}. Set DEVICE_HOME to the DTK installation.")

    def llvm_major(executable):
        output = subprocess.check_output([str(executable), "--version"], text=True, stderr=subprocess.STDOUT)
        match = re.search(r"(?:clang|LLVM) version\s+(\d+)", output)
        if not match:
            raise RuntimeError(f"Cannot determine the LLVM version of {executable}")
        return int(match.group(1))

    compiler_major = llvm_major(sdk / "bin/hipcc")
    override = os.environ.get("FLAGCX_HCU_LLVM_BIN")
    candidates = [Path(override).expanduser()] if override else [sdk / "llvm/bin", sdk / "aillvm/bin", sdk / "bin"]
    compatible = []
    for candidate in candidates:
        tools = [candidate / tool for tool in ("opt", "llvm-as", "llvm-dis")]
        if not all(tool.is_file() for tool in tools):
            continue
        versions = {llvm_major(tool) for tool in tools}
        if len(versions) == 1 and next(iter(versions)) >= compiler_major:
            compatible.append((versions.pop(), candidate))
    if compatible:
        # LLVM bitcode readers support older producers. In DTK, hipcc can use
        # LLVM 17 while the complete opt/as/dis suite uses AOMP LLVM 18. Prefer
        # the nearest eligible version; actual readability is checked by the
        # DU build's llvm-dis and opt verifier, then by the kernel link check.
        _, selected = min(compatible, key=lambda entry: entry[0])
        return sdk, selected.resolve()
    raise RuntimeError(f"Cannot find a consistent DTK opt/llvm-as/llvm-dis suite with LLVM >= {compiler_major}. "
                       "Set FLAGCX_HCU_LLVM_BIN to its directory. Mixed tool versions and older readers are rejected.")


class HCUFlagCXRegistrar(FlagCXRegistrar):
    """Supply DTK build parameters and HCU bundle paths to the shared registrar."""

    always_rebuild = True
    require_header_manifest = True

    def get_compile_cmds(self):
        source = Path(self.flagcx_src_dir)
        if os.environ.get("COMPILE_KERNEL", "1") != "1":
            raise RuntimeError("HCU FlagCX device bitcode requires COMPILE_KERNEL=1.")
        if not (source / "bindings/ir/du/Makefile").is_file():
            raise RuntimeError(f"{source} lacks the Hygon bitcode build. Use FlagCX PR #667 "
                               f"at {FLAGCX_HCU_COMMIT}, or set FLAGCX_SOURCE_DIR to a compatible checkout.")
        sdk, llvm_bin = _toolchain()
        self.arch = _device_arch(sdk)
        json_dir = self._json_include_dir()
        build_dir = source / "build/hcu" / self.arch
        self.src_lib_dir = build_dir / "lib"
        self.cache_lib_dir = self.cache_root_dir / "hcu" / self.arch
        self._set_library_paths()
        cmd = [
            "make",
            "-j",
            os.environ.get("MAX_JOBS", "8"),
            "USE_DU=1",
            "COMPILE_KERNEL=1",
            f"DEVICE_HOME={sdk}",
            f"CCL_HOME={os.environ.get('CCL_HOME') or sdk}",
            f"DEVICE_BITCODE_ARCH={self.arch}",
            f"DTK_LLVM_BIN={llvm_bin}",
            f"BUILDDIR={build_dir}",
            f"JSON_INCLUDE_DIR={json_dir}",
        ]
        for key in ("HOST_COMPILER", "USE_UCX", "UCX_HOME", "USE_IBUC", "USE_MPI", "MPI_HOME"):
            if os.environ.get(key):
                cmd.append(f"{key}={os.environ[key]}")
        cmd.append("device_bitcode")
        # The common registrar groups equal commands and runs this target once.
        return {self.shared_lib_name: cmd, self.bitcode_name: cmd}

    def _get_runtime_path(self, lib_name):
        return Path(self.flagtree_dir) / "third_party/hcu/backend/lib/flagcx" / lib_name

    def _get_required_dst_files_paths(self):
        paths = super()._get_required_dst_files_paths()
        paths["include"] += (self._get_runtime_path(self.bitcode_name).parent / "include", )
        return paths

    def _get_install_metadata(self):
        return {"backend": self.backend_name, "arch": self.arch}


def handle_flagcx(*args, **kwargs):
    HCUFlagCXRegistrar(kwargs).run()


def get_package_data(backends):
    return {
        "triton.backends.hcu": ["lib/flagcx/*", "lib/flagcx/include/*.h"],
        "triton.experimental.tle.language": ["include/*.h"],
    }


def register_cache(cache, flagtree_backend, check_env, set_llvm_env):
    cache.store(
        file="hcu-llvm22-b0ca808-glibc2.35-glibcxx3.4.30-ubuntu-x86_64",
        condition=("hcu" == flagtree_backend),
        url=("https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/"
             "hcu-llvm22-b0ca808-glibc2.35-glibcxx3.4.30-ubuntu-x86_64_v0.5.0.tar.gz"),
        pre_hook=lambda: check_env("LLVM_SYSPATH"),
        post_hook=set_llvm_env,
    )


def install_extension(*args, **kargs):
    return
