# 媒体单测环境隔离

统一 AMD 产品路线让 encoder 设置校验和 `NvidiaVideoEncoder` 的内部 vendor 选择真正跟随
当前 PyTorch runtime。旧 NVENC 单测此前依赖“测试机默认是 NVIDIA”的隐含前提，因此在
ROCm 主机上会错误装配 AMF 默认值，造成与产品行为无关的失败。

本阶段只修正测试边界，不修改产品代码：

- 旧 NVENC encoder/settings 测试默认显式固定 `AcceleratorVendor.NVIDIA`；原有 AMD 用例
  仍在单个测试内显式覆盖为 AMD，因此两类合同不会互相污染。
- session factory 测试只导入可用的 primary/pipeline 模块；三个 secondary heavy module
  由测试专用 `ModuleType` stub 通过 `sys.modules` 临时注入。它不再要求
  `jasna.restorer.__init__` 为仅供测试的模块名增加伪导出，也不会为了创建 mock 先加载
  TensorRT/protection 等产品依赖，因此不破坏产品启动时的 lazy import。
- GUI job ordering 与 Stop 同步测试屏蔽每个伪 job 结束后的 GPU cache/synchronize 清理。
  这些测试不验证 GPU 回收；如果继承完整测试进程已加载的 ROCm 状态，5 秒 join 会把正常
  cleanup 延迟误判为只处理了第一个 job，或误判 Stop worker 没有退出。

验收要求是在同一 Linux ROCm 环境中完整通过 `test_media_init.py`、
`test_video_encoder_unit.py`、`test_session_factory.py`、`test_gui_job_ordering.py` 和
`test_gui_processor_stop.py`，同时保留 AMD 专属用例覆盖。
