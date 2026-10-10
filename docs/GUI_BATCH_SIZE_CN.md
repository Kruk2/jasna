# GUI B4/B8 批大小合同

GUI 的共享检测/处理 batch 默认固定为 B4，不按显存容量自动切换。需要显式使用 B8 时，在“编码设置 →
自定义参数”填写：

```text
--batch-size 8
```

同时传递编码器参数时必须使用英文逗号分隔：

```text
--batch-size 8,rc-lookahead=32
```

只接受 4 或 8。该标记在 GUI 设置边界被提取，绝不会传给 FFmpeg/AMF encoder；留空等同 B4。此功能不修改
NVIDIA/AMD 路由、`auto`、VR/2D、max clip、overlap、检测阈值、编码质量或其他产品默认。

跨供应商载入工厂预设时，CQ 及双 GOP 的修改基准采用当前界面的实际值。
NVIDIA 或不支持双 GOP 的平台不会因为预设中可移植的开关值而自动显示为已修改。
`test_gui_settings_sections.py` 分别验证 Linux NVIDIA 和 AMD 的有效默认值。
