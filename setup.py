# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Modified from
# https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/setup.py
# https://github.com/facebookresearch/detectron2/blob/main/setup.py
# https://github.com/open-mmlab/mmdetection/blob/master/setup.py
# https://github.com/Oneflow-Inc/libai/blob/main/setup.py
# ------------------------------------------------------------------------------------------------

import glob
import os
import platform
import subprocess
import sys

from setuptools import find_packages, setup


def install_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])


IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine().lower().startswith("arm")

TORCH_AVAILABLE = False
CUDA_HOME = None
CppExtension = None
CUDAExtension = None
torch = None  # type: ignore[assignment]


def find_cuda_home():
    """Find CUDA installation directory with multiple fallback options."""
    # First check environment variables
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home and os.path.exists(os.path.join(cuda_home, 'bin', 'nvcc')):
        return cuda_home
    
    # Try to find nvcc in PATH
    try:
        nvcc_path = subprocess.check_output(['which', 'nvcc'], stderr=subprocess.DEVNULL).decode().strip()
        if nvcc_path:
            # Get parent directory of bin/nvcc
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
            if os.path.exists(os.path.join(cuda_home, 'bin', 'nvcc')):
                return cuda_home
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Try common CUDA installation paths
    common_paths = [
        '/usr/local/cuda',
        '/usr/local/cuda-12.6',
        '/usr/local/cuda-12.5',
        '/usr/local/cuda-12.4',
        '/usr/local/cuda-12.3',
        '/usr/local/cuda-12.2',
        '/usr/local/cuda-12.1',
        '/usr/local/cuda-12.0',
        '/usr/local/cuda-11.8',
        '/usr/local/cuda-11.7',
        '/usr/local/cuda-11.6',
        '/opt/cuda',
    ]
    
    for path in common_paths:
        if os.path.exists(os.path.join(path, 'bin', 'nvcc')):
            return path
    
    return None


if IS_APPLE_SILICON:
    try:
        import torch  # type: ignore[no-redef]
        from torch.utils.cpp_extension import CppExtension, CUDAExtension  # type: ignore[no-redef]

        TORCH_AVAILABLE = True
        CUDA_HOME = find_cuda_home()
        if CUDA_HOME:
            os.environ['CUDA_HOME'] = CUDA_HOME
    except ImportError:
        print("Torch not installed on Apple Silicon; skipping extension build.")
else:
    install_torch()
    import torch  # type: ignore[no-redef]
    from torch.utils.cpp_extension import CppExtension, CUDAExtension  # type: ignore[no-redef]

    TORCH_AVAILABLE = True
    # Find and set CUDA_HOME before torch tries to use it
    CUDA_HOME = find_cuda_home()
    if CUDA_HOME:
        os.environ['CUDA_HOME'] = CUDA_HOME
        print(f"Found CUDA at: {CUDA_HOME}")
    else:
        print("CUDA not found. Building without CUDA extensions.")

# groundingdino version info
version = "0.1.0"
package_name = "groundingdino"
cwd = os.path.dirname(os.path.abspath(__file__))


sha = "Unknown"
try:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode("ascii").strip()
except Exception:
    pass


def write_version_file():
    version_path = os.path.join(cwd, "groundingdino", "version.py")
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")
        # f.write(f"git_version = {repr(sha)}\n")


def get_extensions():
    if not TORCH_AVAILABLE:
        print("Torch not available; skipping C++ extensions")
        return None

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "groundingdino", "models", "GroundingDINO", "csrc")

    main_source = os.path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        os.path.join(extensions_dir, "*.cu")
    )

    sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    # Check if CUDA is available and properly configured
    cuda_available = torch.cuda.is_available() or "TORCH_CUDA_ARCH_LIST" in os.environ
    cuda_home_valid = CUDA_HOME is not None and os.path.exists(os.path.join(CUDA_HOME, 'bin', 'nvcc'))
    
    if cuda_available and cuda_home_valid:
        print(f"Compiling with CUDA (CUDA_HOME={CUDA_HOME})")
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        if cuda_available and not cuda_home_valid:
            print("=" * 80)
            print("WARNING: CUDA runtime detected but CUDA compiler (nvcc) not found!")
            print(f"CUDA_HOME={CUDA_HOME if CUDA_HOME else 'Not set'}")
            print("")
            print("This usually means you're using a CUDA runtime-only Docker image.")
            print("To build CUDA extensions, you need the full CUDA toolkit with nvcc.")
            print("")
            print("Solutions:")
            print("  1. Use a CUDA devel image (e.g., nvidia/cuda:12.4.1-devel-ubuntu22.04)")
            print("  2. Install CUDA toolkit: apt-get install cuda-toolkit-12-4")
            print("  3. Set CUDA_HOME to your CUDA installation directory")
            print("=" * 80)
            raise EnvironmentError(
                f"CUDA runtime available but nvcc compiler not found. "
                f"Cannot build CUDA extensions without CUDA toolkit. "
                f"See above for solutions."
            )
        print("Compiling without CUDA")
        define_macros += [("WITH_HIP", None)]
        extra_compile_args["nvcc"] = []
        return None

    # setuptools requires source paths to be relative to the project root
    def _relative_source(path: str) -> str:
        rel_path = os.path.relpath(path, start=this_dir)
        return rel_path.replace(os.path.sep, "/")

    def _as_setup_path(path: str) -> str:
        rel_path = os.path.relpath(path, start=this_dir)
        return rel_path.replace(os.path.sep, "/")

    # Setuptools expects source paths relative to the setup.py directory.
    sources = [os.path.relpath(src, this_dir) for src in sources]
    sources = [src.replace(os.sep, "/") for src in sources]
    include_dirs = [os.path.relpath(extensions_dir, this_dir)]

    ext_modules = [
        extension(
            "groundingdino._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


def parse_requirements(fname="requirements.txt", with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists

    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith("-r "):
            # Allow specifying requirements in other files
            target = line.split(" ")[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {"line": line}
            if line.startswith("-e "):
                info["package"] = line.split("#egg=")[1]
            elif "@git+" in line:
                info["package"] = line
            else:
                # Remove versioning from the package
                pat = "(" + "|".join([">=", "==", ">"]) + ")"
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info["package"] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ";" in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip, rest.split(";"))
                        info["platform_deps"] = platform_deps
                    else:
                        version = rest  # NOQA
                    info["version"] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info["package"]]
                if with_version and "version" in info:
                    parts.extend(info["version"])
                if not sys.version.startswith("3.4"):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get("platform_deps")
                    if platform_deps is not None:
                        parts.append(";" + platform_deps)
                item = "".join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


if __name__ == "__main__":
    print(f"Building wheel {package_name}-{version}")

    with open("LICENSE", "r", encoding="utf-8") as f:
        license = f.read()

    write_version_file()

    build_cmdclass = {}
    if TORCH_AVAILABLE:
        build_cmdclass = {"build_ext": torch.utils.cpp_extension.BuildExtension}

    setup(
        name="groundingdino",
        version="0.1.0",
        author="International Digital Economy Academy, Shilong Liu",
        url="https://github.com/IDEA-Research/GroundingDINO",
        description="open-set object detector",
        license=license,
        install_requires=parse_requirements("requirements.txt"),
        packages=find_packages(
            exclude=(
                "configs",
                "tests",
            )
        ),
        ext_modules=get_extensions(),
        cmdclass=build_cmdclass,
    )
