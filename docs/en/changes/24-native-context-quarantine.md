# Quarantine after specific Windows AMF transfer failures

English (default) | [中文](../../zh/changes/24-native-context-quarantine.md)

Feature: `24-native-context-quarantine`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Classify the specifically observed Windows AMD AMF hardware-to-host transfer failure with a typed error. Preserve the original cause, quarantine the current queue, and leave pending jobs recoverable.

## Usage and default behavior

Classification requires the active Windows AMD AMF decode route and matching errno/native transfer evidence. After quarantine, restart into a fresh process rather than continuing with a potentially poisoned native GPU context.

## Direct prerequisites

- [AMF native decoding and frame ownership](06-amf-native-decode.md)
- [Shared pipeline resource and failure safety](12-pipeline-resource-safety.md)
- [Isolated native video jobs and durable outputs](16-isolated-video-jobs.md)

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

Do not hide unrelated media/I/O failures or label every error as a context fault. This is containment and diagnosability, not proof that TDR, driver resets, or illegal memory accesses have been eliminated.

## Validation and reproduction

```bash
python -m pytest -q tests/test_windows_amf_context_quarantine.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [jasna/gpu_context_errors.py](../../../jasna/gpu_context_errors.py)
- [jasna/gui/processor.py](../../../jasna/gui/processor.py)
- [jasna/media/video_decoder.py](../../../jasna/media/video_decoder.py)
- [tests/test_windows_amf_context_quarantine.py](../../../tests/test_windows_amf_context_quarantine.py)
