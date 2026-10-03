# Integration, dependency, and acceptance records

English (default) | [中文](../../zh/changes/22-integration-record.md)

Feature: `22-integration-record`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Document the complete feature union against latest upstream main, the shared orchestration/backend split, dependency order, exact CPU evidence, historical Linux native evidence, and Windows limits.

## Usage and default behavior

Start with the English feature index and use the Chinese counterpart when needed. Review and merge narrow features in dependency order; rerun affected checks after rebasing to updated main. The local combined source is not a replacement for individually scoped PRs.

## Direct prerequisites

- [Windows ROCm import and vendor compatibility](03-windows-rocm-compat.md)
- [HIP color conversion kernels](04-hip-colour.md)
- [Shared native job and diagnostic contracts](00-shared-native-job-contracts.md)
- [Pinned unified media runtime and installer](01-runtime-contract.md)
- [Reproducible FFmpeg and PyAV builds](02-runtime-build.md)
- [Opt-in Windows HIP resize normalization](05-windows-hip-resize.md)
- [Bounded Windows D3D11–HIP resident media](07-windows-resident-media.md)
- [AMF native decoding and frame ownership](06-amf-native-decode.md)
- [AMD encoder contracts and source-rate Peak VBR](08-encoder-source-rate.md)
- [Smart Render seams and durable resume](13-smart-render-resume.md)
- [Bounded Linux HEVC dual-GOP encoding](09-dual-gop.md)
- [Validated RF-DETR MIGraphX selection](10-rfdetr-migraphx.md)
- [Validated BasicVSR++ MIGraphX B1 restoration](11-basicvsrpp-migraphx.md)
- [Public-source optional license boundary](19-public-source-license.md)
- [Shared pipeline resource and failure safety](12-pipeline-resource-safety.md)
- [Adaptive automatic pre-scan and missed-range fixes](14-automatic-prescan.md)
- [Preserved input folders and output resume validation](15-preserved-folder-outputs.md)
- [Isolated native video jobs and durable outputs](16-isolated-video-jobs.md)
- [GUI controls, diagnostics, and reliable progress](17-gui-settings-diagnostics.md)
- [Exact VR studio and projection routing](18-vr-projection-studios.md)
- [CPU-only SD 1.5 regression isolation](20-cpu-regression-isolation.md)
- [Explicit AMD performance and capacity probes](21-performance-probes.md)
- [Identity-gated Windows AMD Math SDPA](23-windows-sdpa-compat.md)
- [Quarantine after specific Windows AMF transfer failures](24-native-context-quarantine.md)
- [Permutation-aware RF-DETR precision diagnostics](25-rfdetr-precision-probe.md)

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

Keep historical v0.10 records explicitly historical. Do not call waived or skipped Windows/NVIDIA/LTX checks PASS, or describe missing official paid components as implemented.

## Validation and reproduction

No dedicated test module is owned by this feature; check the complete collection and the integrated regression.

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [docs/MAIN_INTEGRATION_ROCM10_20261001_CN.md](../../../docs/MAIN_INTEGRATION_ROCM10_20261001_CN.md)
- [docs/STACKED_PR_LINUX_ACCEPTANCE_CN.md](../../../docs/STACKED_PR_LINUX_ACCEPTANCE_CN.md)
- [docs/V010_PR_REBUILD_20261001_CN.md](../../../docs/V010_PR_REBUILD_20261001_CN.md)
- [docs/WINDOWS_ROCM10_COMPATIBILITY_CN.md](../../../docs/WINDOWS_ROCM10_COMPATIBILITY_CN.md)
- [docs/en/development.md](../../../docs/en/development.md)
