# 公共源码与可选私有授权模块

公共源码 checkout 不包含 `jasna.protection` 的私有实现。`jasna.license_api`
只在该包本身缺失时提供免费模型流程所需的无授权 store：`load_license()` 返回
`None`、`is_licensed()` 返回 `False`，激活请求明确失败。不伪造授权、不解密
受限权重，也不将 supporter 功能转换成免费功能。

包含私有模块的正式发行仍使用真实的 store 和异常类型；私有包内部缺失依赖时
必须继续抛错，不能被公共源码的 shim 掩盖。编译与图像恢复入口使用这个统一边界。
加密模型的测试使用独立伪模块，不把私有代码或密钥提交进公共仓库。

验证：`tests/test_license_api.py`、`tests/test_engine_compiler.py`，其中 TensorRT
相关单元测试显式模拟 NVIDIA，不依赖测试宿主的真实显卡。
