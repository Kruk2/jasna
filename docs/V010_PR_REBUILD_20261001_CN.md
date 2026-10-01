# Jasna 0.10 全功能 PR 重建

固定产品基线为 `v0.10.0` / `93d0584`。本次重建汇总 Linux 桌面开发树、Windows
性能开发树及早期未合入桌面树的跨平台测试夹具修正，不把旧混合研究历史提交给上游。
Linux/Windows 原始源码均有本地归档引用；模型、媒体、缓存、日志、临时验收输出和
机器配置不进入 PR。文档中的本机路径改为匿名示例，历史测量数值与适用边界保留。

## 功能划分

| 编号 | 范围 |
| --- | --- |
| 01 | 统一 runtime 的 ABI、原子安装与启动预检 |
| 02 | Linux/Windows FFmpeg/PyAV 可复现构建与补丁 |
| 03 | Windows ROCm/MMEngine、可选 TensorRT 导入与供应商检查 |
| 04 | Linux 与显式 Windows HIP 色彩内核 |
| 05 | 显式 Windows HIP resize/normalize |
| 06 | Linux AMF Vulkan→HIP D2D、Windows 解码正确性及源帧生命周期 |
| 07 | 默认关闭的 Windows D3D11/HIP resident 媒体组件 |
| 08 | AMD 编码格式/码率合同及源码率 Peak VBR |
| 09 | 有界 Linux HEVC 双 GOP 编码器 |
| 10 | RF-DETR MIGraphX artifact 校验与选择 |
| 11 | BasicVSR++ MIGraphX B1 与扩展加载 |
| 12 | 流水线资源预算、失败传播、native watchdog 与会话接线 |
| 13 | Smart Render 时间轴/接缝与持久断点工作区 |
| 14 | 自动粗扫/精扫、断点与检测覆盖率修正 |
| 15 | 保持输入子目录结构与已有成片验证 |
| 16 | 隔离视频任务、原子发布及显式 Windows guarded backend |
| 17 | GUI 设置、本地日志、遥测、缩放与进度/终态修正 |
| 18 | 鱼眼片商精确识别及投影规则 |
| 19 | 无私有 license store 时的免费源码工作流边界 |
| 20 | CPU/NVIDIA/Windows 测试环境与模块缓存隔离 |
| 21 | 显式性能研究探针及已有研究结论 |
| 22 | 本重建记录及跨平台验收索引 |

## Git 与依赖

每个功能分支均直接以 0.10 tag 为根，仅包含其所属文件的最终改动；一个文件只归属
一个功能 PR。共用入口的接线集中在流水线、Processor 与 GUI 功能中，其正文明确列出
依赖。因此组件 PR 与调用点 PR 需要组合验收，不能将单个依赖未合入的分支当作完整产品。

本地 `codex/v010-all-prs-20261001` 按上述功能分支逐项合并。重建脚本检查所有改动都有
唯一归属，并核对集合 tree 与汇总源码 tree 一致；本地产品只使用集合树。

上游截至本次重建为 `main` / `81dc8b0`，已经加入新的 GUI/模型/媒体抽象，并非 0.10。
PR 目标是原项目 `Kruk2/jasna`，以 Draft 提供功能差异及合并顺序；不能把 0.10 成功
运行当作最新 main 的兼容性证明。上游接受前还需结合其新架构处理重叠修改并复验。
旧关闭 PR 不自动重开；本次不自动合并上游或发布 release。

## 验收边界

本次新运行使用 GPU 隔离环境进行 CPU 集成回归及源码编译/空白检查，保留失败记录，
修复测试夹具的真实环境、模块缓存及配置假设。92 个改动的 CPU 回归文件最终得到
**2056 passed、81 skipped、178 subtests passed**（142.61 秒）。另外 6 个依赖
真实 GPU/TensorRT/媒体解码的测试文件没有在本轮执行。所有 `jasna/`、`scripts/`
与 `tests/` Python 源码编译通过，`git diff --check`、统一 runtime 预检及正式
启动器的 CLI help 通过。这些是集合树的验收，不是各个未组合分支的独立验收。
源帧提前关闭的回归检查要求 close 后释放预取组和 pinned staging，当前组合保留
Linux 的同步/显存保护和 Windows 的 reader ownership 接线。

历史 Linux/Windows 真机记录仍保留在各功能文档中；其中日期、SDK、bit depth、
分辨率、窗口/片段时长及未通过范围均是验收边界。当前没有新增 Windows/GPU 实机
或用户真实视频重跑，不将 CPU 检查冒充这类验收。

Windows HIP colour、resize、resident transport、原生日志策略及 guarded backend
保持既有显式入口/默认关闭边界。Windows split-frame 仍是待办；没有引入 rocDecode。
被研究否决的 B2 检测、全帧共享解码等只保留探针，不进入默认流水线。

桌面入口在集合验证通过后指向独立集成工作树；旧 GUI 进程继续使用原工作树，
用户正常结束后再打开桌面图标即可使用集合版本。权重和设置在本机保留，不进入 Git。
