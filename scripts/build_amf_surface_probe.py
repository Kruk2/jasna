"""Build the isolated AMF Vulkan-to-HIP bridge outside the source tree."""

from __future__ import annotations

import argparse
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Distribution, Extension


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyav-source", required=True)
    parser.add_argument("--amf-include", required=True)
    parser.add_argument("--ffmpeg-include", required=True)
    parser.add_argument("--ffmpeg-lib", required=True)
    parser.add_argument("--vulkan-include", required=True)
    parser.add_argument("--rocm-include", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source = Path(__file__).with_name("amf_surface_probe.pyx").resolve()
    pyav_source = Path(args.pyav_source).resolve()
    output = Path(args.output_dir).resolve()
    temporary = output / "build-temp"
    output.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)

    extension = Extension(
        "_jasna_amf_surface_probe",
        [str(source)],
        define_macros=[("__HIP_PLATFORM_AMD__", "1")],
        include_dirs=[
            str(Path(args.amf_include).resolve()),
            str(Path(args.ffmpeg_include).resolve()),
            str(Path(args.vulkan_include).resolve()),
            str(Path(args.rocm_include).resolve()),
        ],
        library_dirs=[str(Path(args.ffmpeg_lib).resolve())],
        libraries=["avcodec", "avutil"],
        runtime_library_dirs=[str(Path(args.ffmpeg_lib).resolve())],
    )
    distribution = Distribution(
        {
            "name": "jasna-amf-interop-core",
            "ext_modules": cythonize(
                [extension],
                compiler_directives={"language_level": "3"},
                build_dir=str(temporary / "cython"),
                include_path=[str(pyav_source / "include"), str(pyav_source)],
            ),
        }
    )
    command = distribution.get_command_obj("build_ext")
    command.build_lib = str(output)
    command.build_temp = str(temporary / "objects")
    command.inplace = False
    distribution.run_command("build_ext")

    modules = sorted(output.glob("_jasna_amf_surface_probe*.so"))
    if not modules:
        modules = sorted(output.glob("_jasna_amf_surface_probe*.pyd"))
    if not modules:
        raise RuntimeError("AMF interop bridge build produced no extension module")
    print(modules[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
