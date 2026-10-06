# Windows D3D11/HIP resident 媒体组件

此组件显式使用 `JASNA_WINDOWS_D3D11_HIP_RESIDENT=1`，默认关闭。Python
协调器、Cython 包装、C++ 原生实现、构建器和独立探针随同一功能提交；解码器、
编码器及流水线的调用点在对应集成 PR 中。

当前准入尺寸是 1920×1080（原生面为 1920×1088）及 3840×2160。8192×4096
虽然通过单 reader interop 检查，但双 reader 产品拓扑越过主机提交余量保护，
因此明确拒绝。不能将该组件描述为 Windows 8K 默认 GPU 解码优化。

每个编码输出池固定四个 surface；AMF 的 `async_depth=4`、`g=60`、`bf=0`、
`preanalysis=0` 必须与池容量匹配。资源导入/销毁、surface/fence 配平及关闭错误
均需审计，不能只以输出文件存在作为成功依据。

构建入口为 `scripts/build_amf_d3d11_hip_resident.py`，组件探针为
`scripts/probe_windows_d3d11_hip_resident_product.py`。统一 runtime 安装器校验
桥二进制与源码身份；普通 GUI 自动选择及冻结包完整分发仍需 Windows 实机验收。
本次重建只重跑 CPU 契约回归，不将早期组件记录当作当前组合的完整视频验收。
