# v0.11.0 release source correspondence

This file records the custom and patched inputs used by official v0.11.0
packages. The release tag is the source revision for Jasna itself. Release
packages also include `LICENSE`, `LICENSING.md`, `NOTICE`, the third-party
notices, and full license texts.

## Source revisions

| Component | Revision or source |
| --- | --- |
| Jasna | Git tag `v0.11.0`; the tag and package version must match |
| Protection component | The private gitlink revision recorded by the `v0.11.0` tag; source is not published |
| PyAV | Unmodified PyPI `av` 18.1.0, tag [`v18.1.0`](https://github.com/PyAV-Org/PyAV/commit/7e3d950a8b72062502c1a60d672f8ca565313af5) |
| VALI | [`0e5c01ee57222a8ef4f9c7591284c4324bb3fba5`](https://codeberg.org/Kruk2/vali/commit/0e5c01ee57222a8ef4f9c7591284c4324bb3fba5) in the public Kruk2 fork |
| RF-DETR | [`1.8.3` / `3bd6bffbcb13cac3a5b1c37da5a0fd5453b50c86`](https://github.com/roboflow/rf-detr/commit/3bd6bffbcb13cac3a5b1c37da5a0fd5453b50c86) |
| MMagic compatibility patch | [`patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff`](patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff) |

The VALI fork contains the complete modified source, including Jasna's decode
status, corrupt-packet handling, timestamp, and decoder-construction changes.
It is licensed under Apache-2.0.

## FFmpeg in the PyAV and VALI wheels

The PyAV wheels and the Linux VALI wheel link against the
[`pyav-ffmpeg` 8.1.2-1](https://github.com/PyAV-Org/pyav-ffmpeg/releases/tag/8.1.2-1)
build of FFmpeg 8.1.2. The source is FFmpeg tag `n8.1.2`, commit
[`38b88335f99e76ed89ff3c93f877fdefce736c13`](https://github.com/FFmpeg/FFmpeg/commit/38b88335f99e76ed89ff3c93f877fdefce736c13).
The exact build scripts are in the pyav-ffmpeg tag. They request x264/x265 and
`--enable-version3`; because the wheel vendors and links the GPL-licensed x264
library, Jasna distributes the combined wheel payload under GPLv3. FFmpeg's
own runtime metadata identifies its libraries as LGPLv3-or-later.

Linux x86-64 vendor archive:

```text
https://github.com/PyAV-Org/pyav-ffmpeg/releases/download/8.1.2-1/ffmpeg-manylinux-x86_64.tar.gz
SHA-256 665b4903251dc753725e9cf64615c1af7708b932fb26bf16356acc77d76e95f8
```

## Bundled FFmpeg command-line tools

Linux releases use the BtbN `linux64-gpl-8.1` build
`ffmpeg-n8.1.2-34-g9b6c8969e0`. Its FFmpeg source is commit
[`9b6c8969e05b4f0b29f0f85cd501be6b3e582e6b`](https://github.com/FFmpeg/FFmpeg/commit/9b6c8969e05b4f0b29f0f85cd501be6b3e582e6b).
The BtbN GPL variant uses `--enable-gpl --enable-version3 --disable-debug`;
the complete build scripts are in
[BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds).

```text
https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-07-31-14-10/ffmpeg-n8.1.2-34-g9b6c8969e0-linux64-gpl-8.1.tar.xz
SHA-256 09fc77be269c7053e438b7e96548e4af97604faf96a42c4a3c56a1ad74c22c0a
```

Windows packages use the GPLv3 FFmpeg 8 installation selected by the release
builder. Before publishing, its `ffmpeg -version` output, source revision,
configure line, and archive hash must be added to the release notes if they
differ from the Linux build above.

## Python and build environment

Linux releases build CPython 3.13.14 from the official source archive:

```text
https://www.python.org/ftp/python/3.13.14/Python-3.13.14.tar.xz
SHA-256 639e43243c620a308f968213df9e00f2f8f62332f7adbaa7a7eeb9783057c690
```

The application dependencies and exact direct version constraints are in
`pyproject.toml`. Binary packages copy the license files from every installed
Python distribution into `licenses/python-packages/`.

## Reproducibility limit

The public tag contains Jasna's public application source, public patches, and
the source correspondence above. The final whole-application build driver and
the protection implementation remain in the private protection project.
Therefore a public checkout can reproduce the free workflows but cannot
reproduce the supporter-enabled official executable bit for bit.
