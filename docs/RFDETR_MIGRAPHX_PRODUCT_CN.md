# Linux AMD RF-DETR MIGraphX 产品选择

更新时间：2026-08-29

本功能把已经验收的 RF-DETR MIGraphX core 接到检测模型注册边界。只在 Linux、AMD/ROCm、`gfx1100`、FP16、
`rfdetr-v6` 且模型目录存在 `rfdetr-v6.migraphx-gfx1100.json` 时自动选择该 artifact；不满足条件或没有安装
sidecar 时保持原有 PyTorch/ROCm RF-DETR。

“未选择”和“选择后失败”严格区分：一旦发现已安装 sidecar，文件大小、SHA、源模型、runtime 版本、架构或 tensor
ABI 任一检查失败都会直接报错，不会静默回退 PyTorch。显式 manifest 同样只允许 AMD `rfdetr-v6`。

接入点仍是 RF-DETR 自有的 `build_detection_model` 后端选择，不修改检测阈值、Tracker、共享 Pipeline、
BasicVSR++、NVIDIA、VR/2D 或产品 batch。GUI 完整处理、自动粗扫和图片恢复只要使用同一注册入口，就会得到同一
选择规则；解码路线另行收口，不由本功能修改。

聚焦验证：

```bash
python -m pytest -q \
  tests/test_rfdetr_migraphx_product.py \
  tests/test_migraphx_artifact.py \
  tests/test_detection_registry.py
```
