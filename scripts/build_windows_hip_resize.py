"""Offline, reproducible bundle for the admitted Windows HIP resize backend.

Requires the matching HIP SDK but imports no Torch and performs no GPU work.
Run native compilation under the usual resource guard. This does not install
drivers, change environments permanently, or modify the NVIDIA CUDA source.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile

REPO=Path(__file__).resolve().parents[1]
SOURCE=REPO/'jasna/media/resize_normalize.cu'
SOURCE_SHA='206391fa046d3a96b725eaa14e66b9bbfe8e95b903421fe4be74ac270fef63ab'
CANDIDATE_SHA='1475514686fcf32b82803f9000254db33d5e4190bee13a36bd532044d3c2fdfd'
ARTIFACT_SHA='3c93e066930ad74bfa90f50ffdc059b230bc3ccfcc1ead3b1aa40034d29d0a10'
SDK_PINS={
    'bin/hipcc.exe':'6e3c41e378c96bdfe3febd935b0a1f916b246a2e70d9f295db8d64261b54fc07',
    'lib/llvm/bin/clang++.exe':'91e17e5fc56e408f1517526da1eba2bb501375fa8ba496fdcfd0d4146efc77a0',
    'include/hip/hip_runtime.h':'d8f2980c757faef2045c58d36774a1af26a93c9de8be5558c616fa9231f01cd2',
    'include/hip/hip_fp16.h':'db25167fdda886bcedfb3bb9758e3890fd6c7e036668f424c0fb39ce9afd2e07',
    'lib/llvm/amdgcn/bitcode/ocml.bc':'2400b4e558fc9cb73cc97deb6b4ec7db33cc6141cba8c5a938687d74666dced6',
    'bin/amdhip64_7.dll':'37c40daa884de68bb7b5ee1b0575ab97b96866adf304fd76fdbccba772830f5f',
}

def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

def transformed_source(source):
    if source.count(' / 255.0f)')!=4:
        raise RuntimeError('expected four shared source taps')
    transformed=source.replace(' / 255.0f)',' * (1.0f / 255.0f))')
    if hashlib.sha256(transformed.encode('utf-8')).hexdigest()!=CANDIDATE_SHA:
        raise RuntimeError('four-tap derived source differs from accepted input')
    return transformed

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-root',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args()
    sdk=args.sdk_root.resolve(strict=True)
    output=args.output_directory.resolve(strict=True)
    if os.name!='nt' or not output.is_dir(): raise RuntimeError('Windows/existing output directory required')
    if sha(SOURCE)!=SOURCE_SHA: raise RuntimeError('shared CUDA source drift')
    for relative,digest in SDK_PINS.items():
        if sha(sdk/relative)!=digest: raise RuntimeError('matched SDK drift: '+relative)
    spec=importlib.util.spec_from_file_location('resize_bundle_contract',REPO/'jasna/media/windows_hip_resize_contract.py')
    contract=importlib.util.module_from_spec(spec); spec.loader.exec_module(contract)
    destinations=(output/contract.CODE_OBJECT,output/contract.MANIFEST)
    if any(path.exists() for path in destinations): raise RuntimeError('refusing to replace an existing bundle')
    env=dict(os.environ,HIP_PATH=str(sdk),ROCM_PATH=str(sdk),HIP_PLATFORM='amd',HIP_CLANG_PATH=str(sdk/'lib/llvm/bin'))
    # Temporary build-only sources are derived artifacts, never edits to CUDA.
    with tempfile.TemporaryDirectory(prefix='jasna-hip-resize-build-') as temporary:
        build=Path(temporary)
        source=build/'resize_normalize_scalar_reciprocal.cu'
        source.write_text(transformed_source(SOURCE.read_text(encoding='utf-8')),encoding='utf-8',newline='\n')
        include=build/'include'; include.mkdir()
        (include/'cuda_fp16.h').write_text('#pragma once\n// Research-only compatibility shim; the shared CUDA algorithm is unchanged.\n#include <hip/hip_fp16.h>\n',encoding='utf-8',newline='\n')
        binary=build/contract.CODE_OBJECT
        command=[str(sdk/'bin/hipcc.exe'),'--genco','--no-gpu-bundle-output','-fuse-cuid=none',
            '--offload-arch=gfx1100',f'--rocm-device-lib-path={sdk / "lib/llvm/amdgcn/bitcode"}',
            '-O3','-std=c++17','-ffp-contract=fast','-fno-fast-math','-x','hip',
            '-include','hip/hip_runtime.h','-I',str(include),str(source),'-o',str(binary)]
        result=subprocess.run(command,cwd=build,env=env,capture_output=True,text=True,timeout=120)
        if result.returncode: raise RuntimeError(f'HIP build failed {result.returncode}: {result.stderr[:8192]}')
        actual_digest=sha(binary)
        if actual_digest!=ARTIFACT_SHA:
            raise RuntimeError(f'rebuilt artifact {actual_digest} is not byte-identical to accepted resize kernel; no bundle published')
        # Publish only independently reproducible accepted bytes. Exclusive
        # creation means even a late concurrent publisher cannot be overwritten.
        with destinations[0].open('xb') as stream: stream.write(binary.read_bytes())
        with destinations[1].open('x',encoding='utf-8') as stream:
            json.dump(contract.accepted_manifest(),stream,indent=2)
    print(json.dumps(dict(status='build_pass',artifact=str(destinations[0]),sha256=ARTIFACT_SHA,
        manifest=str(destinations[1]),source_sha256=SOURCE_SHA,derived_sha256=CANDIDATE_SHA,
        scope='byte-identical offline artifact, still requires product runtime acceptance'),indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())
