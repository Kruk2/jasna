# Legacy full-video benchmarks

Full-length files, Linux, RTX 5090, i9-13900K, the listed clip size per input. Lada 0.10.1 to Jasna 0.6.2 are the
older README results; v0.7.2 and v0.9.0 were run on 2026-07-26 (details in `2026-07-26_release_matrix.md`).

| File | Clip (s) | Lada 0.10.1 | Jasna 0.3.0 | Jasna 0.5.0 | Jasna 0.6.2 | Jasna 0.7.2 | **Jasna 0.9.0 (7d9cc8c)** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **ABF-017** (4K, 2h 25min) | 60 | 02:56:26 | 01:20:49 (2.2x faster) | 01:10:00 (2.5x faster) | — | — | **42:17 (4.2x faster)** |
| **HUBLK-063** (1080p, 3h 10min) | 180 | 01:34:51 | 44:21 (2.1x faster) | 37:57 (2.5x faster) | 30:58 (3.1x faster) | 23:38 (4.0x faster) | **18:01 (5.3x faster)** |
| **DASS-570_2m** | 30 | 01:08 | 00:30 (2.3x faster) | 00:24 (2.8x faster) | 00:20 (3.4x faster) | 01:05 (1.0x faster) | **00:22 (3.1x faster)** |
| **NASK-223_Test** | 30 | 03:12 | 01:18 (2.5x faster) | 01:02 (3.1x faster) | 00:58 (3.3x faster) | 01:07 (2.9x faster) | **01:01 (3.1x faster)** |
| **test-007** | 30 | 01:16 | 00:41 (1.9x faster) | 00:28 (2.7x faster) | 00:22 (3.5x faster) | 00:27 (2.8x faster) | **00:27 (2.8x faster)** |

Raw data for the v0.7.2 and v0.9.0 runs: [2026-07-26_legacy_videos.csv](2026-07-26_legacy_videos.csv).
