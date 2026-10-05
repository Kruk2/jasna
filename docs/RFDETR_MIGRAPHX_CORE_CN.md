# Linux AMD RF-DETR MIGraphX Core

更新时间：2026-08-29

## 范围与非目标

本次只重建 RF-DETR 的 Linux AMD MIGraphX artifact/runner 核心，代码边界为：

- `jasna/migraphx_artifact.py`：已编译 MXR 与 JSON sidecar 的严格验证和 pointer-only 执行器；
- `jasna/mosaic/rfdetr_migraphx_runner.py`：RF-DETR v6 medium 的专用 host/model/tensor 合同；
- `jasna/mosaic/rfdetr.py`：显式传入 sidecar 时的窄接入，以及静态 batch 的保序拆分；
- `tests/test_migraphx_artifact.py`：上述合同的 mock/结构回归。

它不是产品自动选择功能。没有恢复或新增 session factory/config/CLI/GUI 接线、artifact 自动发现、runtime 安装器、
共享 Pipeline/Tracker、BasicVSR++、decoder/encoder、NVIDIA 或 VR/2D 逻辑。没有显式 sidecar 时，AMD 仍走既有
`RfDetrTorchRunner`；该核心不主动改变任何默认路线。

## 显式入口与 host 合同

调用方只能通过 `RfDetrMosaicDetectionModel(..., amd_migraphx_manifest_path=Path(...))` 选择该核心。显式路径在
非 Linux 或非 AMD 设备上会在导入 runner/MXR runtime 前拒绝；一旦进入 MIGraphX runner，失败不会回退到 PyTorch。

接受范围固定为：Linux、PyTorch ROCm `cuda` device、可用 GPU、`gfx1100`、FP16、RF-DETR v6 `medium`、
576×576。sidecar 还必须声明 purpose
`jasna-rfdetr-v6-medium-segmentation`、`ihipStream_t` stream ABI，且逻辑输出顺序只能是
`dets`、`labels`、`masks`。输入为 contiguous `float32` `(B, 3, 576, 576)`；三个输出分别为
`(B, 200, 4)`、`(B, 200, 3)`、`(B, 200, 144, 144)`，stride、parameter name 和 dtype 都逐项精确比较。

核心沿用 checkpoint `e49161f` 中已知的静态 batch 合同集合 `{1, 2}`，但本工作树安装并实测的 artifact 是 static
B1。未来产品选择层若要限制为 B1，必须在那里以独立证据和测试明确收窄，不能由自动发现隐式改变本 core。

## Sidecar、ABI 与失败语义

sidecar 使用严格的 schema v1：未知键、空输出、重复逻辑/parameter 名称、无效 SHA-256、非正维度或 shape/stride
rank 不一致都会被拒绝。加载前会验证 MXR 的存在、字节数与 SHA-256，并验证源 checkpoint SHA-256。加载后继续验证：

- 当前平台、ROCm device architecture、PyTorch、HIP、MIGraphX、torch-migraphx 版本；
- 已编译 program 的 parameter 集合、MIGraphX type、shape、stride、输出数量及顺序；
- 运行输入的 device/dtype/shape/stride；
- `run_async` 返回的每个输出必须是预先分配的自有 GPU buffer，禁止接受外部 pointer。

`MigraphxArtifactRunner` 只将 input 的 data pointer 传给 `mgx_argument_from_ptr`，不复制输入，也没有 PyTorch
或其他 backend fallback。`close()` 会清除 program、arguments 和输出 buffer；此后的推理立即报错。MIGraphX Python
binding 若未在当前环境中安装，才只从带当前解释器 `EXT_SUFFIX` 的官方 ROCm `lib` 目录查找；导入 torch-migraphx
pointer bridge 时仅临时把当前解释器的 `bin` 放到 `PATH` 首位，随后恢复原 `PATH`。

## B4 到 static B1 的保序与所有权

上游可以继续传 B4 tensor。若显式 artifact 是 static B1，model 内部将其拆为四个 B1 dispatch，并在每次下一次
dispatch 前对有效输出 slice 立刻 `clone()`，最后按输入顺序 `cat()`。这一步不可省略：MXR 输出由 runner 持有的
pointer-backed buffer 复用，延后复制会让早先 batch 的结果被后一次 dispatch 覆盖。该逻辑不修改 score threshold、
postprocess、Tracker 或共享 Pipeline。

## 来源与 artifact 证据

实现以 checkpoint `e49161f` 的下列文件为重建参考，而不是整包 cherry-pick：

- `jasna/migraphx_artifact.py`
- `jasna/mosaic/rfdetr_migraphx_runner.py`
- `jasna/mosaic/rfdetr.py`
- `tests/test_migraphx_artifact.py`

参考研究记录 `docs/UNIFIED_PYAV_FFMPEG_AMF_RESEARCH_CN.md` 第 112 节（2026-08-27）记载的已验收 B1 资源为：
`rfdetr-v6.pt` SHA-256
`f10bedc4d105c2721e4259b8680203d51f344f73e55e85710d915619f5731b55`、MXR SHA-256
`174a02d30ebff31956cc111a1ffbf2b5d399d5de92d5aba61ffac68a1b32fb88`、sidecar SHA-256
`7c03e58f0406968ad3e3c1ac9e24e1e6f343f55dedc0de0afa3494bc3f60a076`。这些值是 artifact 兼容边界，不表示本 core
已接入产品自动选择。

## 本次验证与限制

在当前 Linux ROCm `gfx1100`（Radeon RX 7900 XTX）环境完成：

```bash
/home/user/vr_toolbox_jasna_linux/.venv/bin/python -m pytest -q \
  tests/test_migraphx_artifact.py
/home/user/vr_toolbox_jasna_linux/.venv/bin/python -m compileall -q \
  jasna/migraphx_artifact.py \
  jasna/mosaic/rfdetr_migraphx_runner.py \
  jasna/mosaic/rfdetr.py \
  tests/test_migraphx_artifact.py
git diff --check
```

结果为 `24 passed`，`compileall` 与 `git diff --check` 均通过。额外的真实 smoke 已验证安装 sidecar、MXR 与
`rfdetr-v6.pt` 的 SHA/大小，成功加载 B1 MXR 并经 `MigraphxArtifactRunner` 和
`RfDetrMigraphxRunner` 输出三组有限的 `cuda:0 float32` tensor。显式 model 的 B4 输入实际产生四次 B1 dispatch，
marker 顺序为 `0, 1, 2, 3`，拼接后的三个输出均为 B4 且不再指向 runner 的自有输出 buffer。

这不是完整产品验收：尚未在本次范围内接入 session/product 自动选择，也未运行真实视频的端到端质量、帧数/PTS、
关闭/取消或性能对照。那些工作必须在后续独立接线任务中完成，并且不能把 artifact 加载失败静默降级为 PyTorch。
