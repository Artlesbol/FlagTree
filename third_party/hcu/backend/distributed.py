"""HCU FlagCX bundle discovery and device-library linking."""

import hashlib
import json
import os
from pathlib import Path


class FlagcxRuntimeConfig:
    """Resolve the matching host library, device bitcode and consumer headers.

    Ordinary kernels can still compile when FlagCX is absent. Explicit path
    overrides are checked so a misspelled path cannot silently select a bundled
    library instead. No GPU or host shared library is loaded at import time.
    """

    def __init__(self):
        self.bitcode_path = None
        self.shared_lib_path = None
        self.include_path = None
        self.arch = None
        self.is_available = False
        keys = ("USE_FLAGCX", "USE_DIST", "USE_DISTRIBUTED", "USE_TLE_DIST", "USE_TLE_DISTRIBUTED")
        self.enabled = not any(os.environ.get(key, "ON").lower() in ("off", "0", "false") for key in keys)
        if not self.enabled:
            return

        module = os.environ.get("FLAGCX_MODULE_PATH")
        bundle = Path(module).expanduser() if module else Path(__file__).parent / "lib/flagcx"
        self.bitcode_path = self._path("FLAGCX_BITCODE_PATH", bundle / "libflagcx_device.bc")
        self.shared_lib_path = self._path("FLAGCX_LIB_PATH", bundle / "libflagcx.so")
        self.include_path = self._path("FLAGCX_INCLUDE_PATH", bundle / "include", directory=True)
        if module and not all((self.bitcode_path, self.shared_lib_path, self.include_path)):
            raise RuntimeError(f"Incomplete HCU FlagCX bundle: {bundle}. Expected libflagcx_device.bc, "
                               "libflagcx.so and include/.")
        if self.bitcode_path:
            manifest = self.bitcode_path.parent / "metadata.json"
            if manifest.is_file():
                metadata = json.loads(manifest.read_text())
                if metadata.get("backend") != "hcu":
                    raise RuntimeError(f"FlagCX bundle is not built for HCU: {manifest}")
                self.arch = metadata["arch"]
        self.is_available = bool(self.bitcode_path and self.shared_lib_path and self.include_path
                                 and (Path(__file__).parent / "flagcx_wrapper.py").is_file())

    @staticmethod
    def _path(key, default, directory=False):
        override = os.environ.get(key)
        path = Path(override).expanduser() if override else default
        exists = path.is_dir() if directory else path.is_file()
        if override and not exists:
            raise RuntimeError(f"{key} does not point to an existing {'directory' if directory else 'file'}: {path}")
        return path.resolve() if exists else None


flagcx_rt_conf = FlagcxRuntimeConfig()


class Distributed:

    def get_extern_libs(self, arch=None):
        config = flagcx_rt_conf
        if not config.enabled or not config.bitcode_path:
            return {}
        if arch and config.arch and arch.split(":")[0] != config.arch:
            raise RuntimeError(f"FlagCX bitcode targets {config.arch}, but this kernel targets {arch}. "
                               "Rebuild FlagTree with DEVICE_BITCODE_ARCH set to the kernel GPU architecture.")
        # HCU's need_extern_lib checks whether a declaration contains this key.
        # flagcxDev* matches 'flagcx', whereas the NVIDIA key 'libflagcx' does not.
        return {"flagcx": str(config.bitcode_path)}

    @property
    def is_available(self):
        return flagcx_rt_conf.is_available


def link_flagcx(llvm_mod, extern_libs, llvm, hcu):
    """Import FlagCX before scanning its newly introduced DTK dependencies."""
    path = dict(extern_libs or ()).get("flagcx")
    if path and hcu.need_extern_lib(llvm_mod, "flagcx"):
        llvm.link_extern_libs(llvm_mod, [path])


def extern_libs_hash(extern_libs):
    """Include bitcode contents in the kernel cache key, not just its path."""
    result = []
    for name, filename in sorted(extern_libs):
        digest = hashlib.sha256()
        with open(filename, "rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        result.append((name, filename, digest.hexdigest()))
    return tuple(result)
