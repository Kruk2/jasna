# Linux and Windows feature review index

English (default) | [中文](../zh/feature_reviews.md)

This index covers 26 feature-scoped changes based on upstream main `81dc8b053fb317c063390daab1dab8289c2094df`. Every feature has an English guide and a Chinese counterpart. English is the default review/documentation entry point; historical Chinese engineering records remain supplemental evidence, not a substitute for these guides.

## Features and direct prerequisites

Prerequisite numbers refer to this table, not GitHub PR numbers. An empty dependency list does not imply hardware certification.

| Order | Feature | Direct prerequisites |
| --- | --- | --- |
| 1 | [Windows ROCm import and vendor compatibility](changes/03-windows-rocm-compat.md) | None |
| 2 | [HIP color conversion kernels](changes/04-hip-colour.md) | 1 |
| 3 | [Shared native job and diagnostic contracts](changes/00-shared-native-job-contracts.md) | 1, 2 |
| 4 | [Pinned unified media runtime and installer](changes/01-runtime-contract.md) | None |
| 5 | [Reproducible FFmpeg and PyAV builds](changes/02-runtime-build.md) | 4 |
| 6 | [Opt-in Windows HIP resize normalization](changes/05-windows-hip-resize.md) | 2 |
| 7 | [Bounded Windows D3D11–HIP resident media](changes/07-windows-resident-media.md) | 4, 2 |
| 8 | [AMF native decoding and frame ownership](changes/06-amf-native-decode.md) | 3, 4, 2, 7 |
| 9 | [AMD encoder contracts and source-rate Peak VBR](changes/08-encoder-source-rate.md) | 2, 7 |
| 10 | [Smart Render seams and durable resume](changes/13-smart-render-resume.md) | 8, 9 |
| 11 | [Bounded Linux HEVC dual-GOP encoding](changes/09-dual-gop.md) | 9, 10 |
| 12 | [Validated RF-DETR MIGraphX selection](changes/10-rfdetr-migraphx.md) | 6 |
| 13 | [Validated BasicVSR++ MIGraphX B1 restoration](changes/11-basicvsrpp-migraphx.md) | 1, 12 |
| 14 | [Public-source optional license boundary](changes/19-public-source-license.md) | None |
| 15 | [Shared pipeline resource and failure safety](changes/12-pipeline-resource-safety.md) | 3, 8, 7, 9, 11, 13, 10, 14 |
| 16 | [Adaptive automatic pre-scan and missed-range fixes](changes/14-automatic-prescan.md) | 8, 12 |
| 17 | [Preserved input folders and output resume validation](changes/15-preserved-folder-outputs.md) | 10 |
| 18 | [Isolated native video jobs and durable outputs](changes/16-isolated-video-jobs.md) | 3, 4, 2, 8, 11, 15, 10, 16, 17 |
| 19 | [GUI controls, diagnostics, and reliable progress](changes/17-gui-settings-diagnostics.md) | 3, 18, 14 |
| 20 | [Exact VR studio and projection routing](changes/18-vr-projection-studios.md) | None |
| 21 | [CPU-only SD 1.5 regression isolation](changes/20-cpu-regression-isolation.md) | 19 |
| 22 | [Explicit AMD performance and capacity probes](changes/21-performance-probes.md) | 15 |
| 23 | [Identity-gated Windows AMD Math SDPA](changes/23-windows-sdpa-compat.md) | 1, 12 |
| 24 | [Quarantine after specific Windows AMF transfer failures](changes/24-native-context-quarantine.md) | 8, 15, 18 |
| 25 | [Permutation-aware RF-DETR precision diagnostics](changes/25-rfdetr-precision-probe.md) | 23 |
| 26 | [Integration, dependency, and acceptance records](changes/22-integration-record.md) | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 |

## Submission and shared architecture

Shared job/GUI/scan/restoration/output orchestration stays common across vendors and operating systems. Native decode/encode, GPU kernels, identity checks, and failure recovery use their corresponding adapters. rocDecode is not restored.

The source review stack is cumulative. A cross-fork upstream PR cannot use a base branch that exists only in the fork. Submit independently scoped changes against upstream main in dependency order, rebasing and retesting after prerequisites land; do not silently publish a cumulative suffix as an independent feature. Independent root features are Windows ROCm compatibility, the runtime contract, the public-source license boundary, and VR studio routing.

## Evidence and limitations

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

Private protection source, credentials, weights, generated media, installed runtimes, and user configurations are not part of the public PRs. A passing mocked license test does not unlock paid functionality. The production GUI launcher is not switched by documentation/submission work.
