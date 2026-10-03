# AMD 8K 管线优化审计（2026-09-05）

本轮只使用独立测试入口和事务目录，没有修改产品导入图、GUI 默认值、
RF-DETR 产品 manifest、花屏修复、显存回收或 AMF 编码路线。全程未使用
rocDecode。

## RF-DETR MIGraphX static B2

当前产品检测器已经是 MIGraphX，但产物为 static B1。本轮根据当前产品 B1
的 `mixed-dot-projector-convolution-fp16` 拓扑重新导出、兼容重写并编译了
static B2，而不是复用 2026-08-24 的旧 B2。

- B1 两次提交与 B2 一次提交的微基准：19.921 ms 对 17.756 ms/两帧；
  B2 吞吐提高 12.19%。
- 真实 8K HEVC Main/NV12 20 秒正反矩阵顺序：B1、B2、B2、B1。
- `detect-track` 中位数：B1 42.80 秒，B2 39.95 秒；B2 快 7.13%。
- 产品 wall 中位数：B1 143.790 秒，B2 141.777 秒；B2 只快 1.42%。
- B2 两次检测覆盖一致，且与 B1 共识覆盖相同，均覆盖 1202 帧。
- 四个输出均为 1202 帧，严格软件解码通过，PTS/DTS 和 250 帧 GOP
  关键帧/参数集检查通过。
- B2 两次输出 framemd5 相同；B1/B2 因浮点检测输出不同而不要求互相
  bit-exact。
- B2 的整卡峰值显存中位数比 B1 约增加 392 MiB。

结论：B2 对检测阶段有效，但整条产品路线只有 1.42% 收益，低于值得承担
额外 artifact、显存和维护成本的幅度。保持产品 static B1，不集成 B2。

证据目录：

`/mnt/D/AI/amf-unified-work/transactions/jasna-rfdetr-migraphx-b2-dot-projector-attempt002-20260905/`

## 单次解码共享原帧容量

测试入口没有替换现有两个 AMF reader，只记录 DecodeDetect 产生某 PTS 与
BlendEncode 消费同一 PTS 之间的积压，因此输出行为不变。审计输出的
framemd5 与未加审计的 B1 基线完全相同。

- 生产/消费：1202/1202 帧，结束时无残留。
- 实测峰值积压：348 帧。
- 实际 RGB uint8 8K 帧为 96 MiB；直接共享需要 32.625 GiB。
- 即使保留原生 NV12（48 MiB/帧），348 帧也需要 16.313 GiB。
- 帧驻留时间：中位数 24.42 秒，P95 32.70 秒，最大 38.71 秒。
- 当前样本从第 0 帧开始有连续双眼 track；每个首段必须达到
  `max_clip_size=180` 才会送入修复。因此带背压的共享 ring 若低于约
  180 帧会形成“检测等空槽、混合等修复结果”的闭环等待。
- 180 帧下限仍需 16.875 GiB RGB，或 8.438 GiB NV12；P010 原生帧则需
  16.875 GiB。它会耗尽当前安全显存余量，并破坏不同显存容量显卡的适配。

结论：当前 BasicVSR++ 整段修复架构下，完整解码帧共享不是可接受的显存
优化。保留双 reader；第二次解码本质上是用 VCN 重算换取十几到几十 GiB
的帧缓存，不能直接删除。

证据目录：

`/mnt/D/AI/amf-unified-work/transactions/jasna-single-decode-capacity-20260905/`

## 后续更合理的研究方向

若继续优化 blend/encode，优先研究“第二 reader 保留原生 NV12/P010，
只把修复 ROI 转换并贴回 YUV，再直接提交 AMF”，目标是减少全帧
YUV→RGB→YUV 往返，而不是缓存整段 RGB 原帧。该方向会影响色彩、色度
采样和 ROI 边缘，必须保持默认关闭，并分别验证 Main/NV12 与 Main10/P010
的画质、接缝、严格解码和性能后才能考虑集成。

### 原生 YUV 第一阶段探针

新增 `scripts/probe_amd_native_yuv_roundtrip.py`，只在独立探针进程内运行，
不注册产品 backend，也不修改 GUI/CLI 默认。探针保留第二个 AMF reader，并
用同一个产品 `NvidiaVideoEncoder` 的 AMF 参数、PTS 缓冲、host-native 输入和
输出校验做同源 A/B：

- `baseline-rgb`：当前完整帧 `NV12/P010 -> RGB -> NV12/P010`；
- `native-yuv`：AMF Vulkan surface 仍经现有审计桥 D2D 到 HIP packed YUV，
  跳过两次完整帧色彩转换，直接进入已验收的 pinned-host AMF 提交。

第一阶段故意不做 ROI 修复，也不恢复旧的 encode-side Vulkan/HIP bridge。
这样可以先隔离测量完整帧色彩往返本身的成本；若这一层收益不明显，就没有
必要承担 YUV ROI 色度边缘和 fisheye 投影的额外复杂度。只有无修复 passthrough
在真实 Main/NV12 与 Main10/P010 上均通过帧数、PTS、格式、严格软件解码和
同源正反性能矩阵，才进入局部 ROI YUV 混合。

### Main/NV12 实测与停止结论

固定样本为此前保留的 20 秒 8K HEVC Main/NV12 片段：

`/mnt/D/AI/amf-unified-work/transactions/jasna-hevc-main8-nv12-20260903/inputs/source-8k-main8-20s.mp4`

没有使用用户明确排除的 `#3` CQ18 样本。

单 AMF 会话 300 帧 ABBA 结果：

- baseline-rgb 中位 51.387763 秒；
- native-yuv 中位 51.020922 秒；
- native-yuv 只快 0.719%；
- 60 帧同源质量检查中，baseline 为 PSNR 48.294489 dB、SSIM 0.995749，
  native-yuv 为 PSNR 50.274785 dB、SSIM 0.995866。原生 YUV 避免额外的
  8-bit RGB 量化和第二次 4:2:0 采样，因此质量略好，但编码吞吐几乎不变。

随后给同一探针增加 `--writer dual`，基线调用产品
`AmdDualGopFrameWriter`，实验臂只覆盖 `_prepare()`，直接把 packed NV12
复制到相同的共享 pinned-host 池。两臂使用完全相同的两个持久 AMF 会话、
250 帧 GOP、VBR Peak、async depth 4、PTS 规则和最终拼接校验；产品类和默认
值没有改动。

600 帧 ABBA（baseline、native、native、baseline）结果：

- baseline-rgb：70.252723 / 69.957529 秒，中位 70.105126 秒；
- native-yuv：69.772019 / 100.676760 秒，中位 85.224389 秒；
- 第一组相邻正常配对仅快约 0.689%，与单会话的 0.719% 一致；第二次
  native-yuv 出现约 30 秒离群慢速，因此整组中位数反而慢 17.74%；
- 系统日志没有 GPU reset、VCN timeout、OOM 或热故障证据，离群值的精确
  来源未定位。它不改变“正常情况下收益仍不足 1%”这一停止判断；
- 四臂均为 600 帧、10.010 秒、8192x4096 HEVC Main；packet PTS/DTS 严格
  递增，范围均为 0..899399；
- 四臂均通过 `/usr/bin/ffmpeg -xerror -err_detect explode` 软件严格解码，
  双 GOP writer 的独立 VPS/SPS/PPS access-point 校验也通过；
- 每臂 600 次 AMF→HIP 复制、1200 次 plane D2D、600 次 external-memory
  import/destroy、map/release 和 export-FD close 全部配平，禁止的 host map、
  staging、D2H、software hwframe transfer 均为 0；
- pinned pool 每臂结束 `in_use=0`，峰值 16..18/25；
- 两次 baseline 输出逐字节 SHA-256 相同；两次 native-yuv 输出也逐字节
  SHA-256 相同。

结论：去掉完整帧 `YUV→RGB→YUV` 的确改善纯转码画质，但在当前 host-native
AMF 提交和双 GOP 编码瓶颈下，稳定速度收益仍只有约 0.7%，同时长测还出现
一次明显性能离群。按实验门槛停止该路线，不继续 Main10/P010，不实现高风险
的 ROI YUV pasteback/fisheye 局部投影，也不恢复旧 encode-side bridge。
生产代码、花屏修复、显存策略、Smart Render 和双 GOP 默认保持不变。

证据报告：

- 单会话：
  `/mnt/D/AI/amf-unified-work/transactions/jasna-native-yuv-roundtrip-20260905/main8-abba-300/REPORT.json`
- 双 GOP：
  `/mnt/D/AI/amf-unified-work/transactions/jasna-native-yuv-roundtrip-20260905/main8-dual-abba-600/REPORT.json`

## 测试入口

- `scripts/probe_rfdetr_migraphx_b2.py`
- `scripts/probe_rfdetr_migraphx_b2_product.py`
- `scripts/probe_single_decode_capacity.py`
- `scripts/probe_amd_native_yuv_roundtrip.py`
- `tests/test_probe_rfdetr_migraphx_b2.py`
- `tests/test_probe_single_decode_capacity.py`
- `tests/test_probe_amd_native_yuv_roundtrip.py`

聚焦测试：

```bash
/home/user/vr_toolbox_jasna_linux/.venv/bin/python -m pytest -q \
  tests/test_probe_rfdetr_migraphx_b2.py \
  tests/test_probe_single_decode_capacity.py \
  tests/test_probe_amd_native_yuv_roundtrip.py
```
