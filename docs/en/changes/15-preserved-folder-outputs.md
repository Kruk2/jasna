# Preserved input folders and output resume validation

English (default) | [中文](../../zh/changes/15-preserved-folder-outputs.md)

Feature: `15-preserved-folder-outputs`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Resolve each job's output relative to its selected input root, preserve requested subfolders, create directories before publication, and validate existing outputs before treating a resumed batch item as complete.

## Usage and default behavior

Enable the GUI preserve-input-subfolders option and select the output root. Full, Smart Render, source copy, and resume use the shared output-path rules; Linux and Windows do not duplicate these rules.

## Direct prerequisites

- [Smart Render seams and durable resume](13-smart-render-resume.md)

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

Reject path escape, absolute-path injection, symlink escape, and output collisions. A file's existence alone is not completion; cancellation/failure must retain useful diagnostics and not mark pending jobs successful.

## Validation and reproduction

```bash
python -m pytest -q tests/test_batch_resume_output.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [docs/PRESERVED_FOLDER_OUTPUTS_CN.md](../../../docs/PRESERVED_FOLDER_OUTPUTS_CN.md)
- [jasna/gui/output_paths.py](../../../jasna/gui/output_paths.py)
- [jasna/gui/queue_panel.py](../../../jasna/gui/queue_panel.py)
- [jasna/gui/resume_validation.py](../../../jasna/gui/resume_validation.py)
- [tests/test_batch_resume_output.py](../../../tests/test_batch_resume_output.py)
