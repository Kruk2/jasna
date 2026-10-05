# 最新上游 main 整合与 ROCm 10 验收

状态（2026-10-03）：Linux 最新 main / ROCm 10 集合已获用户授权替换桌面 GUI。
Windows 最新兼容增量已纳入本地功能审查栈；新入口的 Windows/NVIDIA 真机验证
未冒称通过。未推送、发布或重开上游 PR。

## 2026-10-03 最新审查栈

固定上游仍为 `81dc8b053fb317c063390daab1dab8289c2094df`。保留原栈前 22 个
确切功能提交，新增三个窄功能，最后重新整理验收文档，共 26 项：

| 新增范围 | 功能分支 | 边界 |
| --- | --- | --- |
| Windows AMD RF-DETR SDPA | `feature/main-r4-23-windows-sdpa-compat` | 精确 Torch/HIP/DLL/gfx1100 身份门；FP16 独立，不改 Linux/NVIDIA |
| 原生上下文队列隔离 | `feature/main-r4-24-native-context-quarantine` | 只匹配 Windows AMF host-transfer 故障；保留原错及未启动项，不宣称修好 TDR |
| 检测精度诊断 | `feature/main-r4-25-rfdetr-precision-probe` | 显式合成探针，一对一 proposal 匹配；不是默认流水线或画质认证 |
| 更新验收文件 | `feature/main-r4-22-integration-record` | 原四份整合文档及 Windows 兼容记录，不带机器设置、私有模块或生成媒体 |

集合目标为 `integration/main-all-prs-20261003`。每项正文必须列出确切基分支及
依赖；累计前缀可组合验证，不把后面的累计差异直接当成独立 upstream/main PR。
补丁逐项回放、普通合并及最终集合必须得到同一 tree。先保留本地审查材料，
发布时重新核对 upstream main，并经用户确认后才推送或创建 PR。

本次只重新组织提交和更新文档，不改生产处理代码。最终审查集合的 `jasna/`、
`scripts/`、`tests/` 与已部署冻结 GUI 树 `c9d6acdb877aaead64cc287d1a611e6b3db8d25e`
必须逐字节相同；桌面入口仍使用已部署集合，不因整理分支切换用户正在使用的目录。
实际前缀/全量测试次数、输出及 SHA 以随审查材料保存的新 manifest/report 为准，
下文 3109 项是原 23 项栈的历史结果，不能冒充新栈测试。

用户明确豁免本轮 Windows SDK 原生资产重建、全卡遥测及实机 A/B，记录为
`WAIVED_BY_USER_NOT_RUN`，不是 PASS。旧资产身份不改写；Windows resident、
colour/resize、guarded worker 等显式优化保持既有能力门和默认关闭边界。
其余 Windows 新入口、NVIDIA/LTX及付费模型兼容性未认证；公共源码没有官方
可导入的 protection 组件，不能把授权测试需求写成模块已实现或激活已成功。

## 源码基线与取舍

- 当前固定上游：`81dc8b053fb317c063390daab1dab8289c2094df`，版本 0.11.0。
- 旧 Linux/Windows 全功能集合：`c45bd038e55b882ffd9cf4f2db81841236b81b3e`。
- 独立集合分支：`integration/main-all-features-20261001`。最终发布前重新确认上游
  main 是否有新提交；功能 PR 名使用 `feature/main-*`，不使用研究工具名称。
- 59 个文本冲突已逐项整合，但文本冲突消失不等于运行正确。保留上游新的
  Session/共享检测器、SegmentRestoration 模型与 seed、LTX/NVIDIA 限制、媒体
  probe/encoder_settings 模块、队列分页和主线程回调；各平台只替换其有证据的后端。
- 保留上游 SAR、AAC、codec-less stream、AV1 CQ/qindex、full hvc1/Smart hev1
  等新增正确性修复，不用旧版本整个文件覆盖它们。
- 保留已有 AMF D2D、HEVC 双 GOP、MIGraphX、内存预算、断点扫描/工作区、输入
  子目录、隔离任务、Windows 显式优化和鱼眼片商规则。绝不重新接入 rocDecode。

历史 0.10 重建和 22 个功能分组见 `V010_PR_REBUILD_20261001_CN.md`。本轮需要
重新验收，不能把旧功能文档中的真机通过结果写成最新集合的通过结果。

### 最新 main 的本地功能分支拆分

此前 22 个按文件独占的规划组存在循环依赖，不能当成无依赖、单独可运行的 PR。
本轮抽出原生任务/设置/诊断共用契约，将 AMD 平台测试基础随兼容层落入，再按
明确依赖顺序拆成 23 个本地候选分支，名称为 `feature/main-r3-*`。消费端接线测试
与对应 Pipeline/Processor/GUI 一起加入，后端和工具的单元测试先加入；最终测试
文件逐字节恢复，不删除或降低最终断言。

每个确切分支前缀已独立 checkout，编译、完整测试收集及对应功能测试全部通过。
最终集合完整 CPU 回归为 3109 passed、225 skipped、178 subtests passed
（首次精确功能栈 119.87 s，随后纯文档头复验同样通过）。CPU 测试以空 GPU
visibility 运行；TensorRT 专用模块明确待 NVIDIA
环境，不以跳过项冒充通过。造片使用具备 lavfi/软件编码器的系统 FFmpeg，精简统一
运行库另走真实 GPU 验收；CPU 合成造片工具不是正式路线的解码回退。

从上述 upstream base 依次普通 Git merge 23 个功能头，每次 merge tree 与该前缀
精确一致，所有功能头都是集合的祖先。测试时的全部代码/文档树为
`3c70598944bd6fac59e31928a6c155b8bf5aefde`，集合分支
`integration/main-all-prs-20261001`；本节是随后补充的纯文档验收记录，不改变处理代码。
模型、媒体、运行库、机器设置和研究归档历史不进入这些功能提交。

这是有明确基分支的本地审查栈，不是 23 个可直接对 upstream main 提交的累计 diff。
上游贡献需要依赖先落地，再对届时最新 main rebase/复验，确保每个 PR 只显示自己的
功能差异。首轮验收时没有发布/重开上游 PR，也未替换桌面运行库；后续授权部署
与 26 项审查栈以本文开头的 2026-10-03 状态为准。

生产默认 epoch 下、跨 8K Main10/4K 与 5K Main8 的真实 GUI 长批次已完成：
13 个任务全部成功，wall=9427.04 s（2 h 37 m 7 s）。处理源码与上述
`3c70598944bd6fac59e31928a6c155b8bf5aefde` 完全相同；后续差异仅本文件的
验收记录。Windows/NVIDIA 新环境实机仍待外部硬件，本机 CPU 模拟、旧 SDK
记录不能替代。此本地审查栈仍不得作为累计差异直接发布到 upstream main。

## 独立候选环境

候选使用 Python 3.12、Torch 2.12.0+rocm10.0.0、torchvision 0.27.0+rocm10.0.0、
MIGraphX 2.17.0+rocm10.0.0、从 ROCm 10 对应源码重建的 Torch-MIGraphX 1.2。
MIGraphX 源码固定 `becdb3da862f2297041b746b90bc6130e2b1d1f7`，Torch-MIGraphX
扩展源码固定 `e551a861cf8fc0865d81920fa6f53db210763eed`。

ROCm 10 是发行版本；本环境 Torch 报告的 HIP ABI 为 7.15.26333，不能把发行号
当作 HIP 动态库 major。加载顺序先 Torch 后 MIGraphX。Linux 启动器选用 venv 的
wheel SDK 路径，HIP module API 复用 Torch 已载入的运行库，拒绝混用系统运行库。
色彩内核构建显式使用 `--no-gpu-bundle-output` 生成 ELF code object。

PyAV/FFmpeg 使用独立安装且通过原 SHA/ABI/source 合同的统一运行库；AMF bridge
源哈希与已验收资产一致。不能因为 preflight 通过就推断新 HIP 真机兼容：正式
产品路径仍需逐项验证。此为首次建立候选环境时的记录；随后验收和授权切换状态
以本文开头为准，原环境与回滚入口保留。

新 RF-DETR 和 B1 模型 artifact 在独立目录重建，不修改旧 artifact 或 manifest。
版本、扩展 SHA、checkpoint/语义源码 SHA、GPU、静态 ABI、28-node/3-partition
检查不放宽。上游两处语义源文件变化已审查：删除未用 import 以及 NVIDIA 子引擎
路径/编译辅助代码调整，不改变 AMD 传播主体；产品和探针采用审查后的精确新哈希。

## 当前证据

- 最新完整 CPU 回归：3109 passed、225 skipped、178 subtests passed；已包括新增
  SDK/静态批次/双 GOP 接口/离线编译器用例。跳过项不记为真机通过。
- 真实 ROI、相同 RGB 哈希、三轮：B1 31/60 帧旧版 0.2250/0.4237 s，候选
  0.1946/0.3724 s；数值与本环境 eager 的原有 allclose 门通过。
- RF-DETR 四个相同真实帧、20 轮：10.019 → 7.533 ms；0.35 阈值下每帧有效
  检测数一致，合并遮罩 IoU 0.9937–1.0。不比较无效查询的逐位置最大差来判断漏检。
- 正式 3840×2160 HEVC Main/NV12 300 帧完整路径：候选成功，实际 restore=8.6 s；
  两个 AMF reader 各 300/300 D2D，禁止 host/map/staging/D2H/bridge 项全部为零。
- 正式 8192×4096 HEVC Main10/P010 1202 帧双 GOP 路径：候选成功，五个独立
  VPS/SPS/PPS 接入点与产品 PTS 检查通过。初次运行暴露旧 EncoderSpec 属性访问，
  已改为新版共享 AMF_SMART_FRAGMENT_OPTIONS，并用真实 EncoderSpec 补两个回归。

正式新旧 A/B 已补软件严格解码、实际解码帧数、唯一 PTS/递增 DTS 和显示时间轴：

| 范围 | 旧环境 wall | 候选 wall | 验收边界 |
| --- | ---: | ---: | --- |
| 4K Main8，300 帧、10 s | 26.30 s | 20.86 s | 同一 CQ18 完整恢复路径，单轮，不外推长片 |
| 8K Main10 双 GOP，1202 帧 | 183.56 / 156.09 s | 124.43 / 113.39 s | 两轮交替，平均 wall 减少 30.0%，20.103411 s、五个 GOP |
| 5K Main8 双 GOP，1201 帧 | 55.46 s | 46.00 s | 此样片没有有效检测，只验证媒体路径，不宣称恢复模型提速 |
| H.264 Main8 → HEVC，300 帧、10 s | 26.60 / 23.90 s | 18.84 / 18.84 s | 真实源派生、含 B 帧重排，两对反序 A/B，平均 wall 减少 25.4% |

以上输入、模型权重、范围、VR、码控与硬件相同，没有并行跑新旧 GPU 任务。
8K 新旧最终大小 36,985,381 / 36,852,022 bytes（差约 0.36%），各自两轮一致。
额外 1024 宽下采样的新旧完整输出 SSIM=0.998619；这是差异诊断，不能作为原始
8K 像素或人工效果验收的替代。

额外正式 8K Smart Render 使用新公开离线编译器的独立 artifact，实际 restore=16.7 s。
copy/render 接缝、三个编码 GOP/独立参数集、最终 1202 解码帧、全片严格解码和
显示 PTS 序列通过；没有现场 JIT，也没有软件/host 解码回退。两向色彩与 seek
真机回归合计 52 passed，涵盖 8/10-bit、矩阵/range、pitch、重复/非默认 stream、
双 reader 和非零 stream start。RGB Torch oracle 显式关闭 HIP，避免强制 HIP
测试设置误传给 CPU-selector reference；不改变或降低原生转换的数值断言。

该 Smart Render 样片的容器 duration=20.074667 s，不能声称与 full 的
20.103411 s 容器字段完全相同。逐包显示跨度为 20.103411 s，和 full 一致；
源样片自身的 duration 元数据也不同于其实际包显示跨度。最终判定以实际解码
帧数和完整显示 PTS 为依据，同时保留这些头部字段差异供后续播放器验收。
补跑旧源码/旧运行库的同范围 Smart Render 后，旧版容器字段也为 20.074667 s，
新旧均严格解码为 1202 帧、相对源 PTS 最大差 6 微秒；该差异不是本轮新增回归。

补充 H.264 full 的格式语义验证：同一 3840×2160、yuv420p、300 帧/10 s、含
B 帧重排的真实源派生样片，经 `h264_amf` → Vulkan/HIP D2D → 实际 B1 恢复
→ GUI/CLI 所选 HEVC 输出；新环境两轮 primary restore=7.2/7.3 s，两路 reader
均 300/300 D2D，无禁止传输。四份新旧输出分别独立严格软件解码为 300 帧，
10.000000 s、唯一 PTS/递增 DTS、相对源 PTS 差为零，实际 codec 均 HEVC Main。
两轮新输出逐字节一致；20,381,018 bytes 对旧 20,224,864 bytes（约 +0.77%）。
该短片只能证明此格式/设置的正确性和 A/B，不代表长 H.264 批次稳定，耗时改善
属于整套源码/SDK/模型 artifact 的变化，不能全部归因于 ROCm 版本号。

### 连续五分钟与真实 GUI 工作队列

另用历史保存的真实连续五分钟 8K Main10 输入（不是重复短片），正式 full/
双 GOP 路线完成 17,985 帧、72 个独立 GOP；实际 primary restore=1111.8 s，
总 wall=1228.29 s。两路 reader 均 17,985/17,985 D2D，禁止传输项为零；whole-card
峰值 21,976 MiB，最低余量 2,584 MiB，没有 offload、pressure episode 或 critical
reclaim。输入 457,509,510 bytes，输出 456,086,213 bytes，视频 duration
300.049722 s、容器 duration 300.053333 s。没有同设置的当日旧版五分钟 A/B，
所以这项只证明该跨度的正确性/稳定性，不外推新的长片加速百分比。

独立软件严格解码和实际解码帧数为 17,985，唯一 PTS/递增 DTS 通过。逐整数 PTS/
Fraction 比较，归一化时间轴最大差 22.222 微秒；验收使用由源/输出 time-base
推导的固定双时钟量化界 55.556 微秒（上限 1 ms），不随片长/GOP 数累积放宽。
最初本地辅助脚本用六位小数 pts_time 与任意 20 微秒界拒绝该样片，失败证据保留；
更正的是独立辅助测量方法，产品 1 ms 时间轴安全门与媒体代码没有因此放宽。

真实 GUI Processor/isolated-worker 队列依次完成 4K Main8 300 帧和 8K Main10
1202 帧，保留中文多级子目录；测试独立设置 6 s epoch，使后者经历四个分段/
三个进程回收。此项验证 GUI 生产调用链，不是用户窗口/播放器的人工验收，也不将
有并行 CPU 验证负载的 wall 作为性能 A/B。实跑暴露并修复：

- `mark_completed` 仍发旧五参数回调；迁移到上游六参数（含 stage）并补严格签名/
  weighted JobProgress 测试，不给 GUI callback 加默认参数掩盖漏迁移。
- 正常 typed AMF session recycle 被当作失败 workspace 打 ERROR；只将明确的
  `amf_session_limit` 控制流记 DEBUG，真实 HIP/其他失败仍 ERROR 并保留工作区。
- 回收后重播已完成前缀和 FPS 校准会清零速度/ETA；现在不发相同前缀进度，新采样
  尚未充分时沿用本视频上一有效 FPS 估算剩余时间，收到新样本立即替换，不跨任务/
  LTX stage 继承。此变更只是显示估算，不改变 GPU 调度或实际处理速度。

修复后真实双任务重跑：restoring 帧进度不倒退、有效 FPS 后不再归零、队列完整成功，
无 ERROR/CRITICAL；两份输出 SHA-256 与仅修回调后的前一轮逐字节相同，说明显示/
日志修复没有改变成片。新增失败保持可见的单元测试与最终完整 CPU 回归均通过。

真实 GUI 在独立配置/Xvfb 下建立并正常关闭。另完成生产默认 epoch 的多小时
Processor/native-worker 佇列：七项真实连续五分钟 8K Main10，穿插三项 4K Main8
与三项 5K Main8，合计 13 项、27 个恢复片段、wall=9427.04 s，实际 primary
restore 合计 7533.3 s。中文子目录、单调恢复帧进度、有效样本后 FPS 不归零、
队列正常完成全部通过，无 ERROR/CRITICAL。54 条双 reader D2D 审计的禁止
host/map/staging/D2H/bridge 传输均为零；没有 offload、pressure episode 或
critical reclaim。

各片段全卡遥测日志最高 23,767 MiB，最低余量 793 MiB；不能仅凭末段的较低
峰值宣称显存始终宽裕。独立进程树监控在队列启动后约 59 s 才开始，采样范围内
树 RSS 峰值约 7,991 MiB、全卡峰值约 22,440 MiB、系统 available 最低约
19,291 MiB；它不能代替全片段日志中的峰值。此轮无 OOM/压力事件不等于任意
显卡和任意长片都无显存风险。

三个不同格式的首份成片分别独立严格软件解码/计帧；其余十份先对整份输入和整份
成片逐字节 `cmp`，仅在与对应直接验收样本完全一致、文件未被修改时复用解码
证据，同时独立检查各成片包数、PTS/DTS、关键帧位置。不是声称十三份重复内容
都重新独立软件解码。8K 七份输出 SHA-256 完全相同、17,985 帧、归一化 PTS
最大差 11.111 微秒，低于固定双时钟量化界 55.556 微秒；4K/5K 分别 300/1201
帧，PTS 最大差 0/5.556 微秒。

按用户要求，此轮不生成或查看成片截图；严格解码、帧数、时间轴与接缝检查是
坏码流/坏帧的验收依据。编码合法但内容已异常的画面不一定产生解码错误；数值
一致性和 D2D/色彩测试提供另外的证据，但不宣称替代任意画面的人眼判断。
该批次之后 Windows/NVIDIA 新运行库真机仍未认证；Windows 三项用户豁免和
Linux 已授权 GUI 切换见本文开头。仍不提交或重开上游 PR。

额外负例：旧中段切片 Main8 5K 含两个 `discard` 前导 B 包，1417 包只解出
1415 幅画面，仍被原有最终帧数安全门拒绝。该限制在此前运行库的历史验收中已有
准确记录（见 HEVC_SMART_RENDER_ENCODER_CN.md），不是本轮放宽安全门的理由。
成功/性能矩阵采用原片 IDR 开始、无 discard 前导包的同一真实 1201 帧样片。

新 B1 artifact 可用显式离线编译器重建：

```bash
python scripts/build_basicvsrpp_migraphx_b1.py \
  --checkpoint /path/to/lada_mosaic_restoration_model_generic_v1.2.pth \
  --output-dir /path/to/new-b1-artifacts
```

必须先安装该环境已构建的 `_torch_migraphx` 扩展。编译器检查精确语义源哈希、EMA、
gfx1100、28-node/3-partition 和零输入数值，拒绝覆盖已有目录；失败中间物保留诊断。
它不自动安装、不改变默认选择，也不代替单独的严格 cold-load 和真实 ROI/影片验收。

## 复验与限制

CPU 隔离新 SDK 时使用空的 ROCR_VISIBLE_DEVICES、HIP_VISIBLE_DEVICES 与
CUDA_VISIBLE_DEVICES；旧式 `-1` 在该 SDK 的 ordinal parser 中会 abort，不能继续
用它作为可靠的 CPU 测试屏蔽方式。图形测试在独立 Xvfb 上执行，不占用用户 GUI。

正式 CLI 通过 `scripts/run_jasna_unified.py --runtime-root /path/to/runtime --`
启动；同一真实输入、相同模型/范围/max-clip/VR/编码设置，分别选择旧环境和候选。
Main10 双 GOP 使用 `--amd-dual-gop-encode`；4K Main8 不强行放宽已有双 GOP 几何门。
候选目录由 JASNA_BASICVSRPP_MIGRAPHX_B1_DIR 和 JASNA_TORCH_MIGRAPHX_EXTENSION
明确选择，RF-DETR sidecar 放在候选检测权重旁；首次必须强制 B1=1 并检查
fallback=False，不能误测 eager。所有生成媒体、NPZ、模型/扩展、日志和机器路径
只保留在本地验收目录，不进入 Git。

Windows 新 SDK 的 colour/resize、D3D11-HIP resident、guarded worker、主流程与
native DLL 生命周期需 Windows 真机重建/验收；Linux CPU 模拟测试及 Windows
旧运行库记录不能替代它。NVIDIA/LTX 真机也不在本机可用范围内。Windows 显式
优化保持既有默认关闭和精确 manifest 门，不通过改版本号绕过二进制身份检查。
