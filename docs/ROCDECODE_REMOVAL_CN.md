# Linux AMD rocDecode 删除与迁移记录

更新时间：2026-08-30

## 最终裁决

本 PR 从跨平台累计验收 HEAD `bdd00ec3d9154a7aaf1ebe23122d242dbf8dba2f`
独立删除 rocDecode。它只收口 Linux AMD 解码后端，不调整共享 Pipeline、自动 VR/2D
识别、NVIDIA VALI/NVDEC、Windows AMD、RF-DETR、Tracker、BasicVSR++、CQ、编码器或
产品默认 B4。

删除前，Linux AMD AV1 Main NV12/P010 已经由产品 `auto` 进入通过验收的 AMF Vulkan
→ HIP stable dma-buf cache。10,000 帧 B4 产品吞吐为 `924.696 fps`，旧 rocDecode
同口径基准为 `1137.729 fps`；用户已明确接受 `18.72%` 差距。引用审计还确认产品没有
实例化 `ReusableRocDecoder`，旧后端剩余用途只有显式诊断和 PyAV 失败后的兼容 fallback。

## 当前产品路由

| 平台/输入 | `auto` 路由 | 失败语义 |
|---|---|---|
| Linux AMD H.264 Main/High 8-bit | AMF Vulkan → HIP `private-deferred` | native fail closed |
| Linux AMD HEVC Main/Main10 NV12/P010 | AMF Vulkan → HIP `private-deferred` | native fail closed |
| Linux AMD AV1 Main NV12/P010 | AMF Vulkan → HIP stable dma-buf cache | native fail closed |
| Linux AMD native gate 外输入 | 既有普通 PyAV 路由 | 保留 PyAV 原始明确错误 |
| Windows AMD | 既有 AMF host 或受控软件解码 → ROCm upload | 既有策略不变 |
| NVIDIA | 既有 VALI/NVDEC/PyAV 顺序 | 既有策略不变 |

本删除不扩大 AV1 profile、pixel format、位深或动态重配 gate。native gate 外 AV1
仍先走原本的普通 PyAV；打开或读取失败时不再切换第二套 AMD 解码实现，也不新增 CPU
fallback。显式 `JASNA_DECODE_BACKEND=rocdecode` 现在与其他未知值一样 fail closed，
由 reader 报出 `Unknown decode backend` 及当前允许值。

## 删除面

- 删除 `jasna/media/rocdecode.py` 与运行时编译的 `rocdecode_bridge.cpp`；
- 删除 wheel/package-data 中的 bridge 源文件；
- 删除 `_RocDecodeFrameSource`、`ReusableRocDecoder`、显式 backend、auto fallback、
  fallback 后的软件重试及对应 close/PTS 分支；
- 删除 Pipeline 中没有产品调用方的 `reusable_rocdecoder` 透传参数；
- 删除旧 bridge/reuse 单元测试，增加旧环境值明确拒绝的迁移回归；
- 自动粗扫继续只构造共享 reader，不获得 scan-specific backend。

## 验收门

提交前必须完成：

1. `compileall`、`git diff --check` 和全部受影响单元测试；
2. Linux AMD AV1 Main 8-bit NV12 与 Main10 P010 使用未设置 decode backend 的产品
   `auto` 完整读取，核对帧数、PTS 严格递增与 AMF lifecycle/cache stats；
3. 旧 `rocdecode` 环境值明确拒绝，普通 PyAV 打开/读取错误不发生第二后端重试；
4. Windows AMD 与 NVIDIA 路由隔离回归；
5. 源码、打包配置和测试中除迁移拒绝用例外不再存在 rocDecode 实现引用。

真实媒体、模型、运行时 cache 和生成输出只写事务目录，不进入 git。

## 2026-08-30 最终 Linux 验收

反向引用审计确认删除 `ReusableRocDecoder`、fallback 专用 `after_pts`、旧 backend
状态和 `threading` import 后没有悬空调用者；本 PR 没有新增替代 backend、兼容层或
产品分支。源码、打包配置和测试中只在“旧环境值应被拒绝”的迁移回归中保留
`rocdecode` 字符串。受影响回归为 `104 passed, 1 skipped`，累计改动测试集合为
`874 passed, 2 skipped`；额外 decoder 环境集合在当前分支和基线均为
`22 passed, 2 failed, 1 skipped`，两个相同失败来自本机 AMD 环境不符合既有
NVIDIA/software-path 测试假设，不是本删除引入的差异。提交前再次独立执行直接路由/
Pipeline/粗扫回归得到 `230 passed, 1 skipped`，相邻 frame-rate、pipeline、decoder、
runtime/installer 集合得到 `85 passed`。

RX 7900 XTX 上以产品默认 `auto`、B4、未设置 decode/cache/copy-stream 环境变量
分别完整读取：

- AV1 Main8 4096x2048 120 帧：120/120，PTS 严格递增，PTS/sample RGB hash
  通过；cache hit/miss `84/36`，FD close/ownership `84/36`；峰值结温
  `67 C`、GPU busy `51%`、VRAM `3816124416` bytes；
- AV1 Main10 8192x4096 120 帧：120/120，PTS 严格递增，PTS/sample RGB hash
  通过；cache hit/miss `84/36`，FD close/ownership `84/36`；峰值结温
  `77 C`、GPU busy `91%`、VRAM `15820898304` bytes。

两次 session create/close 均为 `1/1`，host transfer、CPU Map、staging、D2H、
non-D2D 和 failed bridge copy 均为 0；运行窗口 kernel journal 没有 OOM、GPU reset、
ring timeout 或 page fault，D 盘验收后可用 `41 GiB`，没有残留 runner 或 partial 文件。
温控门保持 `100 C` 持续 10 秒、`120 C` 立即停止，两次均未触发。

第一次 4K 探针在完整读取 120 帧后由 close audit 正确 fail closed。原因是本机已安装
bridge SHA256 `c10f91669ff724a028fd3f64e175431e2bfee4bc99abe92f9dcd286a7711f88f`
早于当前 lifecycle schema，缺少 `cache_fd_close_failures` 和
`cache_last_fd_close_errno`。改用已经独立验收、SHA256
`46970233e77ac088faa7a4360e3d3a1b6e05d5417bffddcd005b23487097a6cd` 的当前 bridge
后两项实片验收通过。产品不会为旧 bridge 放宽 audit；runtime 原子安装应确保代码与
bridge 版本匹配。

同一产品代码提交 `de2143d` 随后完成 Windows 聚焦验收：150 项测试通过、1 项跳过；
H.264 Main8、HEVC Main8、HEVC Main10、AV1 Main10 四种真实媒体均以 `auto`、B4
通过帧数、FPS、位深、严格 PTS 和既有 oracle hash。Windows H.264/HEVC 8-bit 仍为
AMF host，HEVC Main10/AV1 Main10 仍为软件解码后 ROCm upload；没有 CPU-only 输出、
fallback、TDR、AMF/ROCm crash、OOM、残留进程或 partial/tmp。本次追加只更新文档，
不改变 Windows 已验收的产品源码、打包配置或测试。

Linux 正式 unified runtime 随后通过原子安装器更新。安装输入只组合既有合同哈希已接受的
PyAV/FFmpeg 与上述新 bridge，不重编译或修改二进制；先在独立 test-runtime 通过 ABI/
加载路径 preflight 和 4K AV1 Main8 `auto`/B4 120/120 帧，再以 `--force` 原子发布到
`~/.local/share/jasna/unified-runtime/linux-amd`。旧 runtime 保留为
`linux-amd.backup-20260830-153248`。正式目录再次通过 preflight 与同一实片 smoke：
PTS/hash 一致、cache hit/miss `84/36`、FD close/ownership `84/36`、session `1/1`，
host/CPU Map/staging/D2H/non-D2D/failed bridge 全 0；结温检查为 `54→57 C`，运行窗口无
OOM、GPU reset、ring timeout 或 page fault，且无残留进程/partial/tmp。

完整结果和受控运行记录位于：

```text
/mnt/D/AI/amf-unified-work/transactions/
  jasna-ubuntu-rocdecode-removal-20260830/
  jasna-windows-final34-rocdecode-removal-20260830/
  jasna-ubuntu-final34-runtime-install-20260830/
```
