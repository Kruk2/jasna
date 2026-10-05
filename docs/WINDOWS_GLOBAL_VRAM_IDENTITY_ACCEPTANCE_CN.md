# Windows 全卡显存：无 HIP 父进程采样验收

2026-09-08。产品采样器组件及真实跨进程采样通过；尚未接入默认管线、
GUI恢复重试，不是视频性能或完整输出验收。

## 产品实现与分工

Terra worker 拥有新增 `jasna/windows_global_vram.py` 和
`tests/test_windows_global_vram.py`，除此之外未修改文件。新建worker因会话数量
上限失败后，复用既有 Terra worker 完成，不更换低一档模型、不让worker跑GPU。

研究版已验证的 HIP LUID/node → DXGI/PDH Dedicated Usage 算法进入产品模块。
`WindowsGlobalVramReader(device_index, hip_library)`保留既有构造方式。
新增不可变 `WindowsGpuIdentity` 及 `WindowsGlobalVramReader.from_identity()`，
后者只构造独立PDH采样器，不调用HIP、不导入Torch。身份必须来自当前实际
选定的worker，不能跨启动持久化。匹配的是该HIP设备的精确LUID/node，不是
取任意最大显卡、GUI百分比换算或观察进程自己的HIP预算。

保持精确字节、缺失/重复/错卡拒绝、RLock串行read/close、关闭幂等且终态、
关闭失败可重试。device_index 在进入ctypes.c_int前拒绝bool、负值及溢出。
identity只接受规范小写LUID字符串、plain int且0..31的node。

Terra运行19项纯CPU假实现测试通过；主线程完整阅读实际两个文件、核对所有权
和工单，独立重跑19项通过。没有改动共享显存水位或默认Windows平台选择。

产品模块SHA：
`09bd4cfb037a41d49f303cc97a9f1368f2cf2baf49d01a8456d072452eae7def`。
测试SHA：
`f8ed269c5e2a40413cf9c354f0a34445875b47e1b01c2042a701a8b74886093e`。

## 主线程真实硬件验证

脚本及运行根目录：
`D:/AI/jasna_windows_amd_dev/Temp/windows-guard-exit-adapter-20260908-a1`。
脚本：`probe_product_vram_identity.py`。
通过运行：`product-vram-identity-runs/2fcb251a6b9244baba617c17a07f9dfc`。

实际生产guard固定180秒、Job6144MiB、host commit reserve10240MiB、
physical reserve8192MiB、process limit16、poll1秒；原媒体开关仍0。
仅两个串行64MiB HIP分配子进程，复用既有SHA钉住的HIP DLL加载/分配帮助函数。
没有视频输入、AI模型或GUI窗口。

CPU观察进程在子进程报告实时HIP LUID/node之后，通过产品from_identity创建
独立PDH reader，采样baseline、分配后及子进程退出后各8次、间隔200ms；
最后4次中位数用于原V2门槛。分配前要求全卡余量≥1GiB+64MiB。子进程精确
句柄等待退出后再采样，绝不将hipFree成功视为显存归还。

| 周期 | 分配后全卡读数增量 | 子进程退出后回落 | 退出后最小全卡余量 |
| --- | ---: | ---: | ---: |
| 0 | 64 MiB | 315.1484375 MiB | 22.3667 GiB |
| 1 | 64 MiB | 316.3984375 MiB | 22.3692 GiB |

两次均满足原门槛：增量56..96MiB、退出回落≥56MiB；最后两次全卡余量均≥1GiB。
观察进程开始、创建reader后、子进程退出后及最终均未加载已选amdhip64_7.dll
（同时检查amdhip64.dll、amdhip64_6.dll），且Torch不在sys.modules。
reader关闭两次验证幂等；每轮allocator退出0，所有读线程退场。

guard completed/child0/active0，11秒左右，Peak Job97,275,904字节。
主线程另用PowerShell检查原始parent/observer报告、当前各源码SHA、精确PID、
退出码、资源门槛、两轮数值、余量及HIP/Torch缺席状态，通过；最终进程清单为空。
这是所选HIP运行时的无HIP观察者证据，不声称枚举所有可能GPU API上下文。

## 失败记录及脚本修正

- `a8d154f2b8da4d76a9c9a39979bae0a4`：脚本调用streams.join(2)，而已验收的
  共用流接口只允许0..1秒；finally错误遮住了最初断言。保留原报告及
  `probe_product_vram_identity.pre_join_fix.py`。
- `36bc1734acb24c0d8ea0ce2a68e900a3`：修正join(1)后仍在ready阶段断言失败。
  独立CPU小探针证实venv转发启动器Popen PID与实际Python PID不同（5108/27424）。
  保留原报告及`probe_product_vram_identity.pre_base_python.py`。
  后续探针增加ready/完整traceback记录，并使用已查询确认的匹配venv
  `sys._base_executable=C:/Program Files/Python312/python.exe`直启ctypes-only
  子进程。仍使用同一钉住HIP DLL，不更换GPU栈、不放宽精确PID校验。

两次失败guard均普通child_failed且active0；失败标记未改写为通过。

## 下一步与未完成边界

实时身份通过有界worker协议交给父进程、完全退场后进入共用恢复循环的接线，
以及无视频合成重试信号的真实运行时验证，见下节。下一步在子进程内将同一
产品采样器接到既有VramOffloader策略，并验证所有异常/Stop关闭路径。当前
默认管线仍未接入该reader，不能声称全卡预算缺口已修复。

Linux 4GiB启动预算与约1GiB运行时全卡压力预留是不同概念，本轮均未改动。
还需实际GUI恢复重试、完整视频及连续60秒原生8K Main10验收。Linux是成熟
AMD/ROCm流程参考，不要求Windows FPS相同。

## 实时身份协议与共用恢复循环（后续同日验收）

主线程新增 `jasna/gui/gpu_recovery.py`，提取同一个有界恢复循环。Processor
保留Linux读取器、4GiB启动恢复门槛及None读数的既有策略；Windows注入后端
使用全卡1GiB恢复门槛、至少两次连续样本，None/异常/非法值明确失败。
两者仍用60秒、250ms、Stop事件，不改变VramOffloader运行时水位。

Windows选定运行时builder为共用worker启用身份报告。每次backend.run复制
环境并生成新的32位十六进制attempt token；子进程通过实际HIP cuda:0取得
身份，先关闭自己的临时PDH reader，再发送精确字段的windows_gpu_identity
事件。重复、错token、多字段、非法identity、未请求的事件均拒绝。父进程
仅在guard普通退出码验证、流/进程清理和on_finished全部通过后提交身份。
0/75/86需要身份，校验请求阶段失败1可没有；强制终止不能提交身份。

`open_recovery_reader()`与运行尝试共用锁，身份单次消费；只有进程完全退场
才创建HIP-free reader，在所有退出路径关闭。构造或关闭失败无法证明资源
完整退场，标记后端失效并禁止继续启动。下一次attempt开始前清除旧身份。
自动Windows隔离依旧关闭；产品没有事务目录硬编码。

Terra仅新增 `tests/test_windows_gpu_recovery.py`。主线程审阅实际文件，发现
原先稳定计数重置用例只测low/high/high，要求改成high/low/high/high，并补上
guard拒绝、on_finished异常、下一轮清除旧身份；17项最终CPU测试由Terra及
主线程分别运行通过。主线程还跑16项guard、9项worker启动/身份发送、19项
reader测试通过。旧guard测试的env对象身份断言改为值相等且不同对象，因为
环境复制是防止attempt token污染调用者的刻意变化。

主线程真实CPU guard回归5路径通过，最新记录位于EXIT/product-parent-cpu-runs：
complete_590ac687572441c78f6d74b40fdd5bc9、recycle_03b0313fcbe7488aa9838ae7ed873d09、
error_296c127892fe498787c868e72e6e583f、stop_89a4a7668ef44fe5b98c3bc5dc3c9b79、
timeout_b067b3d8d009471db663b83448b42f80。选定runtime预检82ac4093c97d4770b82bcffe62840e34
及实际无效请求9295a1e74f7a45529605e766fbe5eb15亦通过；预检不进入worker，
故该探针显式移除身份报告要求，实际无效请求仍保留身份报告标志。

### 真实 HIP 子进程退场 → 实际 Processor 恢复 → 再启动

脚本EXIT/run_product_gpu_recovery.py；通过运行：
`product-gpu-recovery-runs/a9ea018bf4bd4d1a815fec6030428b13`。
CPU父进程按原样AST提取实际Processor attempt/control/recovery方法，以免
GUI父端测试初始化Torch。两个串行匹配运行时子进程各自校验统一runtime、
通过产品身份报告helper实际初始化所选Torch/HIP、发出合成native_pressure
重试信号并退出75。它们没有进入视频管线、没有制造真实显存压力或运行AI。

两个子进程均guard child_failed/child75/outer1/active0，经严格验证恢复为75；
实际父方法消费身份、调用原样PDH reader得到两个连续样本、关闭一次，然后
再启动下一轮。恢复耗时0.625秒和0.25秒，末样本全卡余量22.3668和22.3631GiB。
两个guard Peak Job818.15/818.98MiB，固定180秒/原资源额度均满足。CPU父端
每轮恢复后仍无已选HIP DLL/Torch，观察器原绑定已恢复，结束进程清单为空。
主线程独立PowerShell复核原始报告、当前源码SHA、两轮退出/资源/采样/关闭/
无HIP状态通过。观察器只记录PDH返回值，不修改采样结果或重试门槛。

前一次64a68fe6832a4cd5b248ac6efa0b0ffe失败记录保留：主探针错误地嵌套guard，
内层被机器级单实例互斥正常拒绝（205/launcher_error，HIP子进程未启动），
外层child_failed/active0。保留`run_product_gpu_recovery.pre_nested_guard.py`，
改为与既有产品父端验收一致的CPU父进程+逐次受guard保护的HIP子进程。没有
修改guard、移除互斥或放宽限制；CPU父端只做已验证的HIP-free PDH读取。

这是实际运行时/父恢复方法的合成重试测试，不是整个GUI窗口、视频触发重试、
新路径Stop或完整60秒8K视频验收。后续仍须完成这些验收和管线采样器接线。

### 产品管线持有采样器：真实 8K Stop 验收（2026-09-08）

`VramOffloader` 新增可选 reader factory/错误回调。仅 Windows AMD 且受控
worker 身份标志开启时，管线创建产品 reader；Linux 和借用 reader 的旧策略
不变。首次读数在工作线程启动前验证；后续读数失败取消管线并保留首错；
reader 在线程 finally 关闭，停止超时不跨线程强关。管线启动和等待路径用
finally 停止采样器。没有修改水位、恢复门槛、媒体默认标志或自动隔离开关。

Terra 新增 `tests/test_owned_vram_reader.py`，末次返回遇到服务 429，不能声称
worker 已完成最终验收。主线程检查实际文件及生产 diff，独立运行 11 项通过；
另独立重跑 19 reader + 17 recovery + 16 guard + 9 worker，共 72 项 CPU 测试通过。

真实运行：SHARED/native-runs/`GUI600_20260907T175833Z_5d0affedc5ac`。
使用固定 600 帧副本、既有研究媒体后端，产品父 attempt/control/event 方法
和真实共用 GUI worker/session/pipeline；未打开 GUI 窗口。第 128 帧发 Stop，
2.891 秒完成退出，全程 91.719 秒。guard exit0/active0，Peak Job 6207156224B；
主机 commit/physical 最小余量 21523668992/17062252544B，原 180 秒保护未变。
产品 sampler 375 次全卡读数，最低余量 19294236672B，正常关闭、无错误、
无存活采样线程；identity 事件恰好一次，父后端在退出验证后取得身份。
未触发 offload 或压力回收，不能据此声称真实高压力恢复已验证。

主线程独立 PowerShell 核对输入/当前源码/日志 SHA、原始 guard 与 child
报告及终态，进程清单为空。Stop 输出不计作完整视频；完整 600 帧、GUI 窗口
和连续 60 秒 8K 仍需分别验收。

### 同一产品采样器：完整 600 帧及独立输出验证

SHARED/native-runs/`GUI600_20260907T180124Z_5c4f68c5aa85` 完整处理通过。
研究父脚本 `EXIT/run_product_owned_full600.py` 复用既有 bounded transport 和
fixed600 guard runner，仅接入已验收 product-owned sampler wrapper；600 秒
是用户既有固定 600 帧特批，产品父端 180 秒上限未改。不是自动 GUI 隔离验收。

包含准备/退出总计 189.297 秒（600/总时长约 3.17 fps）；子报告准备 43.905 秒，
产品调用 143.419 秒（约 4.18 fps），末段进度约 4.858 fps。口径不同不可混用；
此轮没有同条件 A/B，不能声称采样器提速或导致回退。真实正常任务超过 180 秒，
支持“固定短超时会误杀正常冷启动任务”的判断。

产品采样器 1275 个全卡样本，最低全卡余量 18246639616B，reader 正常关闭，
无存活线程/监控失败/压力回收/offload。guard exit0/active0、无强制终止，
Peak Job 6214221824B，host commit/physical 最低 20389814272/16610324480B。
协议无丢记录或清理错误，身份事件一次且 token 匹配。两个媒体默认标志仍为 0。

主线程另以原 180 秒/原内存保护串行运行 `validate_gui600_video.py`，46.16 秒
完成；严格 CPU 解码 600 帧、8192×4096、Main10/yuv420p10le、10.010 秒通过，
PTS/DTS 均为 2100+1001*n、timebase 1/60000，无缺帧/重复/解码错误。输出 SHA
`6ad604631b8f77535bb0bd39d80f36236caa52fd0fe9d9569b5834f48a1600cf`。
独立 PowerShell 复核 launch source 和 validator identities 当前 SHA，通过且
进程清单为空。这是结构/时间轴验收，不是画质像素一致性、音频或连续 60 秒验收。

并行 CPU-only Terra 新增 `tests/test_windows_vram_reader_factory.py`，7 项覆盖
Windows/HIP/device 门槛、索引选择、异常传播及无 eager Torch/HIP 导入。
主线程逐行检查、独立重跑 7 项通过，git diff --check 通过；本阶段主线程合计
79 项 CPU 测试通过。Terra 恢复后也确认此前 owned suite 自跑 11 项通过。

下一步性能诊断参考本次重叠阶段计时：detect-track 99.7 秒、primary restore
83.2 秒、primary queue-wait 46.0 秒；这些是并发阶段计时，不可相加或当纯 GPU
kernel 时间。仍需继续共享流程产品化、实际 GUI、完整 60 秒 8K 与画质验收。

### 连续 60 秒跟进（2026-09-08，非总体完工）

用户明确批准固定60秒副本1800秒超时，其它保护保持原值。独立长测目录
`D:\AI\jasna_windows_amd_dev\Temp\windows-continuous60s-20260908-a1`，运行
`native-runs/GUI60S_20260907T181600Z_7687e0f2adc0` 已完整完成3600帧，
总722.328秒（端到端4.98fps）、产品调用705.111秒（5.11fps）。无新性能候选，
仍用既有研究媒体后端与共享GUIworker；产品默认路径没有自动启用。

产品采样6731次、最低全卡余量17651847168B，正常关闭、无压力卸载；guard
completed/exit0/active0，Peak Job6238306304B，原资源保护满足。
随后独立串行CPU验收294.032秒通过：完整3600帧严格解码、8K/Main10、精确
PTS/DTS，AAC2816包内容及绝对时间戳与固定输入一致、音频解码通过。主线程
独立核对输出/源码/validator SHA、原始报告和空进程清单，通过。

完整记录见长测目录 `ACCEPTANCE_CN.md`。这证明60秒连续处理及结构/音频
验收，不等于修复像素画质一致性、普通GUI窗口或全部后端产品化已完成。
