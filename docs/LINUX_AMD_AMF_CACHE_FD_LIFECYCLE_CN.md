# Linux AMD AMF dma-buf cache FD 生命周期

更新时间：2026-08-30

## 功能边界

本 PR 只收紧已经启用的 Linux AMD AMF stable dma-buf identity cache 的文件描述符
所有权与验收日志，不改变任何解码选择顺序或产品默认：

- Linux AMD AV1 Main NV12/P010 的 `auto` 仍按上一 PR 使用 cache；
- H.264/HEVC cache 仍不作为产品默认；
- native gate 外输入仍使用普通 PyAV；不存在第二套 Linux AMD 解码 fallback；
- Windows、NVIDIA、共享 Pipeline、RF-DETR、BasicVSR++、B4、CQ 和编码器均不变。

本 PR 不包含已经裁决为 `CLOSED_UNSUPPORTED` 的 HIP→AMF/Vulkan encode surface
ring、source digest、observer/reuse/ownership 探针或 GPU-only encoder 接入。

## FD 所有权合同

Vulkan 每次导出一个 opaque dma-buf FD 后，bridge 按 stable `(st_dev, st_ino)`
身份处理：

1. cache miss：`hipImportExternalMemory` 成功后由 Jasna 立即且只调用一次
   `close()`；session 继续持有并在关闭时销毁 HIP import/mapping。ROCm 7.2.1
   不会替调用方关闭该 FD，不能再按“所有权转交”处理。
2. cache hit：本次新导出的 FD 不再需要，立即且只调用一次 `close()`。
3. import 前或其他失败路径仍持有 FD 时，cleanup 只关闭一次。
4. `close()` 失败记录返回值和 `errno`，reader 在 session close 审计时 fail closed；
   Linux 上不盲目重试，避免已释放后复用的整数被误关。

reader 的最终 JSON transport stats 现在包含：

- `cache_fd_close_calls` / `cache_fd_close_failures`；
- `cache_last_fd_close_errno`；
- `cache_fd_ownership_transfers`（兼容字段；新合同中必须为 0）。

成功门要求每次 export（hit 与 miss）都恰有一次 close，ownership transfer 为 0，
且 close failure 与最后 errno 都为 0。最终 stats 会以
`AMF interop transport stats reader=` 前缀写入 run log，供媒体事务独立解析。

## 已有真实 Linux 验证

旧合同下的 H.264 High/NV12 4096×2048、120 帧 canary 曾记录每个 reader
`36 miss + 84 hit`、84 次 close 与 36 次 ownership transfer。该结果只作为发现
合同错误前的历史证据，不能再作为当前成功门；新合同必须得到 120 次 close、0 次
ownership transfer。原三轮 lifecycle 的媒体、PTS、framemd5、资源与 kernel 结果
仍可证明 cache 外壳未改变成片，但 FD 验收必须按新合同重新执行。

新合同已用 AV1 Main10/P010 实际素材完成 B4 decode-only 回归：调用者取得 1400
帧，cache uploader 合法预取下一批后共有 1404 次 native copy，其中 36 miss、
1368 hit。1404 次 export/FD close 全部配平，close failure/errno 和 ownership
transfer 均为 0；36 个 import/map 在 reader close 时全部释放/销毁。每 100 帧
采样的总 FD 恒为 9、dma-buf FD 恒为 0，退出后回到总 FD 6、dma-buf FD 0；PTS
严格递增，junction 峰值 70°C、显存峰值 2,053,324,800 bytes，运行窗口无相关
kernel 错误或残留进程。当前聚焦测试为 `140 passed, 1 skipped`（AMF core、backend
和 AMD path）；旧 seek 测试在 AMD host 上固定 B24 与产品 AMF B1/2/4/8 门不兼容，
属于既有测试隔离问题，本修复未修改共享 seek/auto 逻辑。

证据目录：

```text
/mnt/D/AI/amf-unified-work/transactions/jasna-ubuntu-amf-h264-cache-ring-canary-b1-attempt003-20260829
/mnt/D/AI/amf-unified-work/transactions/jasna-ubuntu-amf-h264-cache-ring-lifecycle3-b1-aggregate-20260829
```

聚焦单元测试覆盖 cache hit/miss 计数、FD close error fail-closed 和最终 JSON stats
序列化。Linux 累计验收结果统一记录在 `docs/STACKED_PR_LINUX_ACCEPTANCE_CN.md`。
