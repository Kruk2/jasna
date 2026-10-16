# 全路径剖析：瓶颈已从稳态转移到启动；GUI 会话复用修复（2026-10-08）

## 分段实测（RX 7900 XT，20 秒 1080p 含马赛克片段，602 帧）

```
启动（每个会话一次）：约 20 秒
  BasicVSR++ 权重加载              0.7 s
  首次 eager 前向（MIOpen 内核）    7.7 s   ← 每个进程都要重付（见下）
  HIP 计算图捕获（T=90 + T=74）     1.8 s
  检测模型 + MIGraphX 会话          3.0 s
  YUV→RGB torch.compile（首帧）     2.3 s
稳态（602 帧）：9.4 秒（端到端 64 fps）
  decode-detect：decode 3.6 s + detect-track 5.8 s
  主修复：2.7 s（HIP 图回放；eager 时为 11.4 s）
  blend-encode：decode 2.9 s + write 0.9 s
```

结论：**稳态已接近地板**（检测裸测 5.5 ms/帧 = 183 fps，流水线里 9.6 ms/帧，
差额来自与修复/编码共享 GPU），**启动成了大头**（短片场景占墙钟 60%）。

## 7.7 秒首次前向：排查过的死路（避免重复劳动）

| 手段 | 实测 |
|---|---|
| `MIOPEN_CACHE_DIR` 指向持久目录 | 无效（8.1s / 7.7s，缓存目录保持空）|
| 显式 `MIOPEN_USER_DB_PATH` + `MIOPEN_SYSTEM_DB_PATH` | 无效（8.0s）|
| `MIOPEN_FIND_MODE=NORMAL` | **更慢 14.3s** |
| `MIOPEN_FIND_MODE=IMMEDIATE` | 直接失败 |

即：这是本 ROCm/MIOpen 构建在 Windows 上**每个进程一次**的内核选择/加载成本，
无法通过环境变量或磁盘缓存规避；也不能靠"捕获时顺带预热"绕过
（捕获期间禁止 hiprtc 分配/编译）。

## 真正有效的修复：GUI 会话跨批次复用（-24.7 s / 批次）

批处理器本来就在一次批次内复用会话，但 `_run()` 的 `finally` 会关闭它
——于是**每导入一个视频并开始，就重付一次约 20 秒启动**。

改动（`jasna/gui/`）：
- `processor._ensure_video_session`：按 `video_session_key(settings)` 缓存，
  设置变化才重建（此前设置变了还会返回旧会话）；
- 会话在批次结束后保持热态，窗口关闭时由 `Processor.release_sessions()` 释放；
- `video_session_key` 加入 `temporal_overlap`（决定预捕获的片段长度）。

实测（同一 Processor 实例，连续两批）：

| | 墙钟 |
|---|---:|
| 第一批 | 32.5 s |
| 第二批 | **7.8 s（省 24.7 s，快 4.2 倍）** |

两批输出一致；"Restoration models loaded" 仅出现一次；
`tests/test_gui_segments.py` 6 passed，GUI 设置测试通过。

## 剩余已知成本（本次未动）

1. 每进程一次 MIOpen 预热 7.7 s —— 该构建固有问题；
2. `blend_encode_loop` 的第二遍解码（600 帧约 2.9 s）—— 消除需在两条循环
   之间加有界帧共享队列（架构级改动，约占稳态 5-10%）。

## 状态

- 已推送 PR #475（`34a5460`），评论：`pull/475#issuecomment-6050457211`；
- 便携版补丁包更新为 12 文件 + 中文说明（`Jasna-AMD-Windows-补丁-20261008.7z`）。
