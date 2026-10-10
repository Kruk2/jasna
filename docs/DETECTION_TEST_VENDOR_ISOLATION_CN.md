# 检测测试的显卡厂商隔离

更新时间：2026-08-29

检测模型注册、权重发现和 TensorRT 预编译的 NVIDIA 单元测试现在显式模拟 NVIDIA 厂商，避免在 Linux AMD/ROCm
开发机上根据本机显卡误走 `.pt` 权重或跳过 TensorRT 分支。RF-DETR 编译委派测试用本地假模块隔离可选
TensorRT 依赖，因此无需在 AMD 开发环境安装 NVIDIA runtime。

该改动只修正测试环境边界，不修改产品的 NVIDIA/AMD 路由、模型格式、检测阈值或运行时 fallback。聚焦验证：

```bash
python -m pytest -q \
  tests/test_detection_registry.py \
  tests/test_rfdetr_postprocess.py \
  tests/test_model_weights_dir.py
```

当前 Linux AMD 环境结果：`61 passed`。
