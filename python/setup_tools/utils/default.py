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

import subprocess
import os
from pathlib import Path
import shutil
import sys
from importlib.machinery import PathFinder

global registrar


def printinfo(msgs):
    print(f" [TLE-DIST-INFO]: {msgs}.")


class FlagCXRegistrar:

    cache_build = True
    require_header_manifest = False

    def __init__(self, external):
        self.bitcode_name = "libflagcx_device.bc"
        self.shared_lib_name = "libflagcx.so"
        self._set_path(external)

    def _is_flagcx_recompile_required(self):
        ENVS = ("FLAGCX_RECOMPILE", "DIST_RECOMPILE")
        for env in ENVS:
            env_value = os.environ.get(env, "0")
            if env_value in ("1", "true", "True", "ON"):
                printinfo(f"Recompiling FlagCX due to {env}={env_value}\n ")
                return env_value
        return False

    def _get_flagcx_wheel_path(self):
        spec = PathFinder.find_spec("flagcx", sys.path)
        return Path(spec.origin).parent if spec is not None else None

    def _is_cache_available(self):
        ENVS = ("FLAGCX_CACHE", "DIST_CACHE")

        for env in ENVS:
            env_value = os.environ.get(env, "1")
            if env_value in ("0", "false", "False", "OFF", "clean"):
                printinfo(f"Skipping using cache at {self.cache_lib_dir} due to {env}={env_value}\n ")
            if env_value in ("clean"):
                shutil.rmtree(self.cache_lib_dir, ignore_errors=True)
                os.makedirs(self.cache_lib_dir, exist_ok=True)
                printinfo(f"Cache {self.cache_lib_dir} cleaned due to {env}={env_value}\n ")
            return False
        return True

    def is_available(self):
        return Path(self.flagcx_src_dir).exists()

    def _set_path(self, external):
        submodule = external['backend']
        flagtree_cache = external['cache']
        flagtree_config = external['configs']
        backend_name = flagtree_config.flagtree_backend
        self.backend_name = "nvidia" if not backend_name else backend_name
        self.flagcx_src_dir = submodule.dst_path
        self.flagtree_dir = flagtree_config.flagtree_root_dir
        self.src_lib_dir = Path(self.flagcx_src_dir) / "build" / "lib"
        self.cache_lib_dir = Path(flagtree_cache.dir_path) / "flagcx"
        if self.cache_build:
            flagtree_cache._create_subdir(subdir_name="flagcx")
        self._set_library_paths()

    def _set_library_paths(self):
        """Refresh artifact paths after a backend selects its build/cache tree."""
        for lib_name in (self.bitcode_name, self.shared_lib_name):
            src_path = self.src_lib_dir / lib_name
            cache_path = self.cache_lib_dir / lib_name
            setattr(self, f"{lib_name.split('.')[0]}_src_path", src_path)
            setattr(self, f"{lib_name.split('.')[0]}_cache_path", cache_path)

    def _get_runtime_path(self, lib_name):
        return {
            self.shared_lib_name:
            Path(self.flagtree_dir) / "python" / "triton" / "_C" / self.shared_lib_name, self.bitcode_name:
            Path(self.flagtree_dir) / "third_party" / self.backend_name / "backend" / "lib" / self.bitcode_name
        }.get(lib_name)

    def get_compile_cmds(self):
        nproc = os.cpu_count()
        return {
            self.bitcode_name: ["make", "-C", "bindings/ir/nvidia"], self.shared_lib_name: ["make", "-j",
                                                                                            str(nproc)]
        }

    def _compile_and_cache(self):
        cmds = self.get_compile_cmds()
        # Preserve the existing cache policy for backends that use it. Without
        # a cache, always invoke make and let the build system track changes.
        reuse_cache = self.cache_build and not self._is_cache_available()
        rebuild = not self.cache_build or self._is_flagcx_recompile_required()
        # Backends may associate both outputs with one unified make target.
        # Treat them as a group, so the command runs once and partial caches
        # cannot mix host/device artifacts from different builds.
        groups = {}
        for lib_name, cmd in cmds.items():
            groups.setdefault(tuple(cmd), []).append(lib_name)
        for cmd, names in groups.items():
            sources = [getattr(self, f"{name.split('.')[0]}_src_path") for name in names]
            cached = [getattr(self, f"{name.split('.')[0]}_cache_path") for name in names]
            use_cached = not rebuild and reuse_cache and all(path.is_file() for path in cached)
            if use_cached:
                artifacts = cached
            else:
                if rebuild or not all(path.is_file() for path in sources):
                    printinfo(f"Compiling {', '.join(names)} in {self.flagcx_src_dir}")
                    subprocess.run(list(cmd), cwd=self.flagcx_src_dir, check=True)
                artifacts = sources
            for path in artifacts:
                if not path.is_file():
                    raise FileNotFoundError(f"Expected FlagCX build artifact not found: {path}")
            if self.cache_build:
                self.cache_lib_dir.mkdir(parents=True, exist_ok=True)
            for name, artifact, cache_path in zip(names, artifacts, cached):
                runtime_path = self._get_runtime_path(name)
                runtime_path.parent.mkdir(parents=True, exist_ok=True)
                if self.cache_build and not use_cached:
                    shutil.copy2(artifact, cache_path)
                shutil.copy2(artifact, runtime_path)

    def _get_required_src_files_paths(self):
        whl_path = self._get_flagcx_wheel_path()
        if whl_path is None:
            source = Path(self.flagcx_src_dir)
            wrappers = (source / "plugin/interservice/flagcx_wrapper.py", source / "src/flagcx/api.py")
            wrapper = next((path for path in wrappers if path.is_file()), wrappers[0])
            exported = self.src_lib_dir.parent / "include"
            if self.require_header_manifest or (exported / ".flagcx_device_headers").is_file():
                include = exported
            else:
                include = source / "flagcx/include"
            return {"wrapper": wrapper, "include": include}
        return {"wrapper": whl_path / "api.py", "include": whl_path / "include"}

    def _get_required_dst_files_paths(self):
        return {
            "wrapper":
            (Path(self.flagtree_dir) / "python" / "triton" / "experimental" / "tle" / "language" / "flagcx_wrapper.py",
             Path(self.flagtree_dir) / "third_party" / self.backend_name / "backend" / "flagcx_wrapper.py"), "include":
            (Path(self.flagtree_dir) / "python" / "triton" / "experimental" / "tle" / "language" / "include", )
        }

    def _copy_required_files(self):
        src_paths = self._get_required_src_files_paths()
        dst_paths = self._get_required_dst_files_paths()
        wrapper_src = src_paths["wrapper"]
        include_src = src_paths["include"]
        if not wrapper_src.is_file():
            raise FileNotFoundError(f"FlagCX Python wrapper not found: {wrapper_src}")
        manifest = include_src / ".flagcx_device_headers"
        headers = None
        if self.require_header_manifest or manifest.is_file():
            names = manifest.read_text().splitlines()
            if not names or any(not name or Path(name).name != name or not name.endswith(".h") for name in names):
                raise RuntimeError(f"Invalid or empty FlagCX header manifest: {manifest}")
            headers = [include_src / name for name in names]
            for header in headers:
                if not header.is_file():
                    raise FileNotFoundError(f"FlagCX consumer header not found: {header}")
        for wrapper_dst in dst_paths["wrapper"]:
            wrapper_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(wrapper_src, wrapper_dst)
            printinfo(f"flagcx_wrapper.py copied from {wrapper_src} to {wrapper_dst}")
        for include_dst in dst_paths["include"]:
            if include_dst.exists():
                shutil.rmtree(include_dst)
            if headers is None:
                include_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(include_src, include_dst)
            else:
                include_dst.mkdir(parents=True)
                for header in headers:
                    shutil.copy2(header, include_dst / header.name)
            printinfo(f"FlagCX headers copied from {include_src} to {include_dst}")

    def run(self):
        whl = self._get_flagcx_wheel_path()
        if whl is not None:
            printinfo("FlagCX is already installed, skipping compilation...")
        else:
            self._compile_and_cache()
        self._copy_required_files()


def handle_flagcx(*args, **kwargs):
    global registrar
    registrar = FlagCXRegistrar(kwargs)
    if not registrar.is_available():
        printinfo("FlagCX is not available, skipping compilation...")
    try:
        registrar.run()
    except Exception as e:
        printinfo(f"Error compiling FlagCX: {e}")
        printinfo("IF you are using AMD backend, please set the FLAGTREE_BACKEND=amd")
