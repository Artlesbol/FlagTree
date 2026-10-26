"""HCU FlagCX build and linking contracts; runnable without DTK or GPUs."""

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Tuple
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
ENV_KEYS = ("USE_FLAGCX", "USE_DIST", "USE_DISTRIBUTED", "USE_TLE_DIST", "USE_TLE_DISTRIBUTED",
            "FLAGCX_MODULE_PATH", "FLAGCX_BITCODE_PATH", "FLAGCX_LIB_PATH", "FLAGCX_INCLUDE_PATH",
            "DEVICE_HOME", "HIP_PATH", "DEVICE_BITCODE_ARCH", "FLAGCX_HCU_LLVM_BIN", "COMPILE_KERNEL",
            "CCL_HOME", "JSON_INCLUDE_DIR", "MAX_JOBS", "FLAGCX_CACHE", "DIST_CACHE", "FLAGCX_RECOMPILE",
            "DIST_RECOMPILE", "FLAGCX_SOURCE_DIR", "FLAGCX_ILUVATAR_ARCH")


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HCUFlagCXTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ)
        self.env.start()
        for key in ENV_KEYS:
            os.environ.pop(key, None)
        self.addCleanup(self.env.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.common = load_module("flagcx_common_test", "python/setup_tools/utils/default.py")
        # Loading setup_tools' package normally executes setup.py hooks. Import
        # the dependency directly so these host tests never download/build it.
        with patch.dict(sys.modules, {"python.setup_tools.utils.default": self.common}):
            self.build = load_module("hcu_flagcx_build_test", "python/setup_tools/utils/hcu.py")
        self.runtime = load_module("hcu_flagcx_runtime_test", "third_party/hcu/backend/distributed.py")

    def external(self, source, backend="hcu"):
        cache = self.root / "cache"
        return {
            "backend": SimpleNamespace(dst_path=str(source)),
            "configs": SimpleNamespace(flagtree_root_dir=str(self.root / "project"), flagtree_backend=backend),
            "cache": SimpleNamespace(dir_path=str(cache), _create_subdir=lambda subdir_name: (
                cache / subdir_name).mkdir(parents=True, exist_ok=True)),
        }

    def bundle(self):
        bundle = self.root / "bundle"
        (bundle / "include").mkdir(parents=True)
        (bundle / "libflagcx_device.bc").write_bytes(b"bitcode-v1")
        (bundle / "libflagcx.so").write_bytes(b"host-library")
        (bundle / "metadata.json").write_text(json.dumps({"backend": "hcu", "arch": "gfx936"}))
        os.environ["FLAGCX_MODULE_PATH"] = str(bundle)
        self.runtime.flagcx_rt_conf = self.runtime.FlagcxRuntimeConfig()
        return bundle

    def test_bundle_architecture_and_linker_key(self):
        bundle = self.bundle()
        self.assertEqual(self.runtime.Distributed().get_extern_libs("gfx936"),
                         {"flagcx": str(bundle / "libflagcx_device.bc")})
        with self.assertRaisesRegex(RuntimeError, "targets gfx936"):
            self.runtime.Distributed().get_extern_libs("gfx906")

    def test_explicit_missing_path_does_not_fall_back(self):
        self.bundle()
        os.environ["FLAGCX_BITCODE_PATH"] = str(self.root / "missing.bc")
        with self.assertRaisesRegex(RuntimeError, "FLAGCX_BITCODE_PATH"):
            self.runtime.FlagcxRuntimeConfig()

    def test_disabled_flagcx_needs_no_bundle(self):
        os.environ["USE_FLAGCX"] = "OFF"
        os.environ["FLAGCX_MODULE_PATH"] = str(self.root / "missing")
        self.runtime.flagcx_rt_conf = self.runtime.FlagcxRuntimeConfig()
        self.assertFalse(self.runtime.flagcx_rt_conf.is_available)
        self.assertEqual(self.runtime.Distributed().get_extern_libs("gfx936"), {})

    def test_reject_non_hcu_bundle(self):
        bundle = self.bundle()
        (bundle / "metadata.json").write_text(json.dumps({"backend": "nvidia", "arch": "sm_90"}))
        with self.assertRaisesRegex(RuntimeError, "not built for HCU"):
            self.runtime.FlagcxRuntimeConfig()

    def test_link_flagcx_before_resolving_sdk_dependencies(self):
        bundle = self.bundle()
        references = {"flagcxDevPut"}
        linked = []

        def link(module, paths):
            linked.extend(paths)
            references.remove("flagcxDevPut")
            references.add("__ockl_lane_u32")

        hcu = SimpleNamespace(need_extern_lib=lambda module, key: any(key in ref for ref in references))
        llvm = SimpleNamespace(link_extern_libs=link)
        libraries = {"flagcx": str(bundle / "libflagcx_device.bc"), "ockl": "ockl.bc"}
        self.assertFalse(hcu.need_extern_lib(None, "ockl"))
        self.runtime.link_flagcx(None, tuple(libraries.items()), llvm, hcu)
        self.assertEqual(linked, [libraries["flagcx"]])
        self.assertTrue(hcu.need_extern_lib(None, "ockl"))

    def test_bitcode_update_changes_active_compiler_options_hash(self):
        bundle = self.bundle()
        sdk = self.root / "sdk"
        for lib in ("ocml", "ockl", "hip", "opencl"):
            path = sdk / "amdgcn/bitcode" / f"{lib}.bc"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(lib.encode())
        source = ast.parse((ROOT / "third_party/hcu/backend/compiler_hcu.py").read_text())
        options_class = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "HIPOptions")
        scope = dict(dataclass=dataclass, Tuple=Tuple, Path=Path, hashlib=hashlib,
                     Distributed=self.runtime.Distributed, extern_libs_hash=self.runtime.extern_libs_hash,
                     HIPBackend=SimpleNamespace(path_to_rocm=lambda: str(sdk)))
        exec(compile(ast.Module(body=[options_class], type_ignores=[]), "HIPOptions", "exec"), scope)
        options = scope["HIPOptions"](arch="gfx936")
        old_hash = options.hash()
        (bundle / "libflagcx_device.bc").write_bytes(b"bitcode-v2")
        self.assertNotEqual(options.hash(), old_hash)
        self.assertIn("flagcx", dict(options.extern_libs))

    def test_native_architecture_query(self):
        tool = self.root / "aillvm/bin/amdgpu-arch"
        tool.parent.mkdir(parents=True)
        tool.touch()
        with patch.object(self.build.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "gfx936\ngfx936\n")):
            self.assertEqual(self.build._device_arch(self.root), "gfx936")

    def test_select_llvm_tools_matching_hipcc(self):
        sdk = self.root / "sdk"
        for tool in ("bin/hipcc", "llvm/bin/opt", "llvm/bin/llvm-as", "llvm/bin/llvm-dis",
                     "aillvm/bin/opt", "aillvm/bin/llvm-as", "aillvm/bin/llvm-dis"):
            path = sdk / tool
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        os.environ["DEVICE_HOME"] = str(sdk)

        def version(cmd, **kwargs):
            if cmd[0].endswith("hipcc"):
                return "HIP version: 6.1\nAMD clang version 17.0.0\n"
            return "LLVM version 17.0.0\n" if "/aillvm/" in cmd[0] else "LLVM version 22.0.0\n"

        with patch.object(self.build.subprocess, "check_output", side_effect=version):
            self.assertEqual(self.build._toolchain(), (sdk, sdk / "aillvm/bin"))
            os.environ["FLAGCX_HCU_LLVM_BIN"] = str(sdk / "llvm/bin")
            with self.assertRaisesRegex(RuntimeError, "matching hipcc's LLVM 17"):
                self.build._toolchain()

    def test_shared_source_configuration_preserves_backend_specific_pins(self):
        dependency = SimpleNamespace(url="original-url", commit_id="original-pin", dst_path="old")
        utils = ModuleType("python.setup_tools.utils")
        utils.flagtree_submodules = {"flagcx": dependency}
        tools = ModuleType("python.setup_tools.utils.tools")
        tools.flagtree_configs = SimpleNamespace(flagtree_submodule_dir=str(self.root / "third_party"))
        with patch.dict(sys.modules, {"python.setup_tools.utils": utils, "python.setup_tools.utils.tools": tools}):
            self.common.configure_flagcx_source("iluvatar/tle/third_party/flagcx")
            self.assertEqual((dependency.url, dependency.commit_id), ("original-url", "original-pin"))
            self.assertEqual(dependency.dst_path, str(self.root / "third_party/iluvatar/tle/third_party/flagcx"))
            self.build.relocate_flagcx()
            self.assertEqual(dependency.commit_id, self.build.FLAGCX_HCU_COMMIT)
            self.assertEqual(dependency.dst_path, str(self.root / "third_party/hcu/third_party/flagcx"))
            os.environ["FLAGCX_SOURCE_DIR"] = str(self.root / "pre-fetched")
            self.build.relocate_flagcx()
            self.assertEqual(dependency.dst_path, str(self.root / "pre-fetched"))

    def test_build_installs_matching_host_bitcode_headers_and_wrapper(self):
        sdk = self.root / "sdk"
        for tool in ("bin/hipcc", "aillvm/bin/opt", "aillvm/bin/llvm-as", "aillvm/bin/llvm-dis"):
            path = sdk / tool
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        source = self.root / "source"
        (source / "bindings/ir/du").mkdir(parents=True)
        (source / "bindings/ir/du/Makefile").touch()
        wrapper = source / "src/flagcx/api.py"
        wrapper.parent.mkdir(parents=True)
        wrapper.write_text("# matching wrapper\n")
        json_dir = self.root / "json"
        (json_dir / "nlohmann").mkdir(parents=True)
        (json_dir / "nlohmann/json.hpp").touch()
        project = self.root / "project"
        os.environ.update(DEVICE_HOME=str(sdk), DEVICE_BITCODE_ARCH="gfx936", JSON_INCLUDE_DIR=str(json_dir))
        calls = []

        def build(cmd, cwd, check):
            calls.append(cmd)
            output = Path(next(arg.split("=", 1)[1] for arg in cmd if arg.startswith("BUILDDIR=")))
            (output / "lib").mkdir(parents=True, exist_ok=True)
            (output / "include").mkdir(exist_ok=True)
            (output / "lib/libflagcx.so").write_bytes(b"HCU host")
            (output / "lib/libflagcx_device.bc").write_bytes(b"HCU bitcode")
            (output / "include/.flagcx_device_headers").write_text("flagcx.h\nflagcx_device_unified_ir.h\n")
            for name in ("flagcx.h", "flagcx_device_unified_ir.h"):
                (output / "include" / name).write_text("// matching header\n")

        registrar = self.build.HCUFlagCXRegistrar(self.external(source))
        with patch.object(registrar, "_get_flagcx_wheel_path", return_value=None), \
             patch.object(self.build, "_toolchain", return_value=(sdk, sdk / "aillvm/bin")), \
             patch.object(self.build.subprocess, "run", side_effect=build):
            registrar.run()
            registrar.run()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertIn("USE_DU=1", calls[0])
        self.assertIn("COMPILE_KERNEL=1", calls[0])
        self.assertIn("device_bitcode", calls[0])
        self.assertIn(f"DTK_LLVM_BIN={sdk / 'aillvm/bin'}", calls[0])
        installed = project / "third_party/hcu/backend/lib/flagcx"
        self.assertEqual((installed / "libflagcx.so").read_bytes(), b"HCU host")
        self.assertEqual((installed / "libflagcx_device.bc").read_bytes(), b"HCU bitcode")
        self.assertEqual(json.loads((installed / "metadata.json").read_text())["arch"], "gfx936")
        self.assertTrue((installed / "include/flagcx_device_unified_ir.h").is_file())
        self.assertTrue((project / "third_party/hcu/backend/flagcx_wrapper.py").is_file())
        self.assertTrue((project / "python/triton/experimental/tle/language/flagcx_wrapper.py").is_file())
        # Resolve the installed layout rather than explicit path overrides.
        self.runtime.__file__ = str(project / "third_party/hcu/backend/distributed.py")
        installed_config = self.runtime.FlagcxRuntimeConfig()
        self.assertTrue(installed_config.is_available)
        self.assertEqual(installed_config.bitcode_path, installed / "libflagcx_device.bc")

    def test_common_wrapper_supports_legacy_and_new_source_layouts(self):
        registrar = self.common.FlagCXRegistrar(self.external(self.root / "source", "nvidia"))
        source = Path(registrar.flagcx_src_dir)
        legacy = source / "plugin/interservice/flagcx_wrapper.py"
        modern = source / "src/flagcx/api.py"
        for wrapper in (legacy, modern):
            wrapper.parent.mkdir(parents=True, exist_ok=True)
            wrapper.touch()
        with patch.object(registrar, "_get_flagcx_wheel_path", return_value=None):
            self.assertEqual(registrar._get_required_src_files_paths()["wrapper"], legacy)
            legacy.unlink()
            self.assertEqual(registrar._get_required_src_files_paths()["wrapper"], modern)

    def test_common_grouped_build_rejects_partial_cache_and_runs_once(self):
        registrar = self.common.FlagCXRegistrar(self.external(self.root / "source", "nvidia"))
        registrar.src_lib_dir.mkdir(parents=True)
        registrar.libflagcx_cache_path.write_bytes(b"stale host-only cache")
        cmd = ["make", "device_bitcode"]
        calls = []

        def build(command, **kwargs):
            calls.append(command)
            registrar.libflagcx_src_path.write_bytes(b"new host")
            registrar.libflagcx_device_src_path.write_bytes(b"new bitcode")

        with patch.object(registrar, "get_compile_cmds", return_value={registrar.shared_lib_name: cmd,
                                                                      registrar.bitcode_name: cmd}), \
             patch.object(self.common.subprocess, "run", side_effect=build):
            registrar._compile_and_cache()
        self.assertEqual(calls, [cmd])
        self.assertEqual(registrar._get_runtime_path(registrar.shared_lib_name).read_bytes(), b"new host")
        self.assertEqual(registrar._get_runtime_path(registrar.bitcode_name).read_bytes(), b"new bitcode")

    def test_existing_cache_decision_is_preserved(self):
        registrar = self.common.FlagCXRegistrar(self.external(self.root / "source", "nvidia"))
        registrar.src_lib_dir.mkdir(parents=True)
        for name in (registrar.shared_lib_name, registrar.bitcode_name):
            getattr(registrar, f"{name.split('.')[0]}_cache_path").write_bytes(b"cached")
            getattr(registrar, f"{name.split('.')[0]}_src_path").write_bytes(b"current")
        os.environ["FLAGCX_CACHE"] = "OFF"
        with patch.object(self.common.subprocess, "run") as build:
            registrar._compile_and_cache()
        build.assert_not_called()
        # Preserve the pre-existing cache behavior; changing that policy is
        # independent of adding HCU's always-rebuild integration.
        self.assertEqual(registrar._get_runtime_path(registrar.shared_lib_name).read_bytes(), b"cached")

    def test_common_complete_cache_avoids_build(self):
        registrar = self.common.FlagCXRegistrar(self.external(self.root / "source", "nvidia"))
        for name in (registrar.shared_lib_name, registrar.bitcode_name):
            getattr(registrar, f"{name.split('.')[0]}_cache_path").write_bytes(b"cached")
        with patch.object(self.common.subprocess, "run") as build:
            registrar._compile_and_cache()
        build.assert_not_called()
        self.assertEqual(registrar._get_runtime_path(registrar.bitcode_name).read_bytes(), b"cached")

    def test_iluvatar_original_commands_and_paths(self):
        filename = ROOT / "python/setup_tools/utils/iluvatar.py"
        tree = ast.parse(filename.read_text())
        # Import the production class/helpers without running its setup() patch.
        tree.body.pop()
        scope = {"__file__": str(filename), "__name__": "iluvatar_registrar_test"}
        with patch.dict(sys.modules, {"python.setup_tools.utils.default": self.common}):
            exec(compile(tree, str(filename), "exec"), scope)
        source = self.root / "source"
        (source / "bindings/ir/iluvatar").mkdir(parents=True)
        legacy = source / "plugin/interservice/flagcx_wrapper.py"
        legacy.parent.mkdir(parents=True)
        legacy.write_text("# legacy wrapper\n")
        headers = source / "flagcx/include"
        headers.mkdir(parents=True)
        (headers / "flagcx.h").write_text("// legacy header\n")
        scope["relocate_flagcx"] = lambda: None
        scope["_resolve_flagcx_toolchain"] = lambda: ("/sdk/corex", "/sdk/corex/bin/clang", "/sdk/ccl")
        for relative in ("python/triton/experimental/tle/language", "third_party/iluvatar/backend"):
            (self.root / "project" / relative).mkdir(parents=True)
        registrar = scope["IluvatarFlagCXRegistrar"](self.external(source, "iluvatar"))
        calls = []

        def build(command, **kwargs):
            calls.append(command)
            registrar.src_lib_dir.mkdir(parents=True, exist_ok=True)
            if "USE_ILUVATAR=1" in command:
                registrar.libflagcx_src_path.write_bytes(b"iluvatar host")
            else:
                registrar.libflagcx_device_src_path.write_bytes(b"iluvatar bitcode")

        with patch.object(registrar, "_get_flagcx_wheel_path", return_value=None), \
             patch.object(self.common.subprocess, "run", side_effect=build):
            registrar.run()
        self.assertEqual(len(calls), 2)
        self.assertIn("ILUVATAR_ARCH=ivcore11", calls[0])
        self.assertIn("USE_ILUVATAR=1", calls[1])
        self.assertEqual(registrar._get_runtime_path(registrar.shared_lib_name),
                         self.root / "project/python/triton/_C/libflagcx.so")
        self.assertEqual(registrar._get_runtime_path(registrar.bitcode_name),
                         self.root / "project/third_party/iluvatar/backend/lib/libflagcx_device.bc")
        self.assertEqual((self.root / "project/third_party/iluvatar/backend/flagcx_wrapper.py").read_text(),
                         legacy.read_text())
        self.assertTrue((self.root / "project/python/triton/experimental/tle/language/include/flagcx.h").is_file())


if __name__ == "__main__":
    unittest.main()
