# Windows 匹配运行时 GUI 验收边界（2026-09-07）

本记录仅覆盖 GUI 空闲启动、系统检查、资源显示和正常关闭。它不是完整视频、
播放器、停止/恢复或新版 AI 后端的产品验收，不应据此启用研究后端。

## 已落地的后端相关界面修正

- Windows HIP 暖机在原后台线程内执行 `torch.cuda.synchronize()` 后再退出，
  避免已复现的未完成 HIP 工作引发的退出挂起；Linux/NVIDIA 保持原行为。
- Windows 系统内存回退检查只对已确认的 NVIDIA CUDA 构建读取 NVIDIA DRS。
  AMD/ROCm 明确显示不适用；未知供应商显示未检查。检查本身不导入 Torch，
  不初始化 GPU，也不改动驱动策略。
- 这些是 Python GUI/诊断修改，没有新增运行时二进制或修改运行时 ABI，
  因此没有更新媒体 runtime manifest，也没有替换产品默认 Python 环境。

## 独立主会话验收

Terra 提供供应商分流修正和假对象测试；主会话审阅实际 diff，部署前后分别运行：

- 10 个明确指定 node ID 的原产品策略测试，均通过。
- 5 个从实际源码抽取函数的 AST/假对象测试，均通过。

测试禁止原生 Torch/UI 导入，禁用插件自动加载及产品 conftest，完成后断言
`torch` 未出现在模块表。NVIDIA 现有值、未知值、读取失败、AMD 优先级、
供应商缺失和非 Windows 均覆盖。

之前误用 `pytest -k sysmem`：暂存目录名称本身含 sysmem，导致选择整份测试文件，
结果 11 failed / 31 passed / 8 errors，并误导入旧 Torch。该失败保留，不能作为
CPU 检查通过证据；其并发时段的首次 8K 资源测量存在干扰。

真实 GUI 记录：
`D:\AI\jasna_windows_amd_dev\Temp\matched-gui-overlay-a1-20260907-5c11a7\MATCHED_GUI_SYSMEM_A2`

- 核心运行时：Torch 2.14.0+rocm10.1.0a20260906，TorchVision
  0.29.0a0+rocm10.1.0a20260906，Triton 3.8.0.post28，HIP 7.16.26354。
- 三个 GUI 包来自独立审计 overlay；原核心 site-packages 和 application overlay 未变。
- 真实窗口 AMD 项显示 `N/A (AMD/ROCm)`，GPU/VRAM/RAM/CPU 数值可见。
- 点击开始使用后以 Alt+F4 正常关闭；guard completed / exit0 / active0，
  Job 峰值 641429504 B，最小系统提交余量 15020691456 B。
- 采用隔离设置和缓存，没有处理视频，没有改动用户设置。

供应商修正原文件备份：
`D:\AI\amf-unified-work\transactions\jasna-windows-triton38-matched-stack-20260906\gui-sysmem-vendor-backup-686f271a02ae463c82b22e656765289d`

部署后 SHA-256：

- `jasna/os_utils.py`: C6B9528125D81FF99DC93723177823F7519436FD70EE017886E0A54AC3B201DA
- `tests/test_os_utils.py`: E1E6C9BAD69FAE733FD342EA4447636ABC8D81C0105582FAE107541DC5580CEF

## 仍不能外推的部分

真实 GUI 没有通过配置选择研究用 B1 bounded-leaf 检测器的路径：界面批量为
4/8，普通 Windows AMD 检测器仍调用原 PyTorch 核心。研究注入和剩余块编译
不是 SessionConfig 的产品后端选项。关闭“编译 BasicVSR++”不会自动选中它们。

Windows Main10 当前合同是单线程 SLICE 软件解码加 HIP 上传；模型修复仍在 GPU。
两个 reader、检测、修复、混合等共享流程没有改成 CPU 修复。D3D11/HIP resident
及 HIP 颜色内核实验开关继续关闭。空闲 GUI 成功不能替代该视频路径的安全验收。
