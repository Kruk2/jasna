# Windows 文件 worker 原生日志策略：限定验收

2026-09-08。显式 `JASNA_WINDOWS_NATIVE_FFMPEG_LOGS=1`，默认关闭。
必须为 Windows 隔离视频 worker、请求 GPU 身份事件且有 32 位小写十六进制
attempt token。它不是独立的安全沙箱证明；资源边界仍由可信 launcher/Job guard 负责。

## 修改与边界

`jasna/windows_native_logs.py` 在 worker 导入 Processor 前安装一次。当前只准入
PyAV 18.1.0 和已检查的 API/alias 结构。保留 FFmpeg 原生 stderr 回调到进程退出，
后续 set_level/set_libav_level 不再安装 Python/GIL 或无日志回调。级别限制为
ERROR(16)..WARNING(24)，None 恢复 WARNING；这是限定 worker 策略，不是通用
PyAV API 替代。Python logging.ERROR(40) 被夹到 WARNING，避免误当 FFmpeg ERROR。

实际 PyAV import 经 AudioResampler 的 C import 间接加载 av.filter/loudnorm。
最初的“拒绝已加载 filter”设计在原生 A1/A2 正常失败，未触发资源保护；失败证据
保留。因此最终实现不禁止正常 filter 导入，而是验证并重绑定 loudnorm 缓存的
get/set_level，同时拒绝 av.filter.stats 和 av.filter.loudnorm.stats 两个入口。

不支持 Capture 驱动的设备枚举/滤镜统计。splice 的 Capture 目前仅抑制输出，
未消费捕获结果；原生模式下这些警告可能进入 stderr。PyAV 异常的附加日志文字
可能减少，需依赖有界 stderr 诊断。无法拦截此前被未知调用方缓存的 callable，
也无法检测任意 C 库直接替换回调；assert_active 检查 Python 绑定，不是原生指针查询。
本策略仅适用于已审查的文件 worker 导入顺序与能力范围。

## 真实原生日志实验

证据目录：
`D:\AI\jasna_windows_amd_dev\Temp\windows-product-native-logs-20260908-a1`。
最终运行 PRODUCT_LOGS_probe_A3，使用 probe_native_logs_v3.py；A1/A2 不计成功。

通过 PyAV 自带 logging.log 的固定 `%s` 包装实际调用 FFmpeg av_log，而非假日志。
先确认基线 Python Capture 收到唯一标记，再安装产品策略。10 次后续级别修改
以及 loudnorm 的缓存 setter，共 11 条 ERROR 标记全部到独立 OS stderr；Python
Capture/handler 均为空，stdout 不污染，INFO 被过滤。共享 IsolatedWorkerStreams
丢失记录和错误均为零。实际已加载 avutil 路径与 SHA 对应准入 runtime。

Guard completed/exit0/active0/forced0，4.457 秒，peak Job 50085888 bytes。
未加载 Torch，未打开媒体，不是视频性能、GUI 面板或 run-log 验收。

## 真实 8K Main10 解码与关闭

证据目录：
`D:\AI\jasna_windows_amd_dev\Temp\windows-product-native-frame2-20260908-a1`。
PRODUCT_NATIVE_FRAME2_probe_A1，主线程独立 accept_native_frame2.py 通过。

使用唯一获准的 60 帧 8192×4096 Main10 副本，真实产品 decoder 与产品日志策略；
安装后导入实际 video_encoder，其原有晚到的 set_level(40) 确实被重定向一次。
无研究 restore_default_callback 补丁。逐元素 GPU RGB 与精确 PTS 对照 SLICE1 和
仅测试对象上显式设置的 FRAME2：完整 60、seek/stride 10、尾部 B4 3、提前关闭 4、
重复提前关闭 4，共 81 帧全部一致。两次提前关闭 0.156/0.141 秒；全部 reader
原生 owner 清空，全卡采样 reader 关闭、GPU 同步完成。

Guard completed/exit0/active0/forced0，38.987 秒，peak Job 4054233088 bytes；
全卡最低可用 20964257792 bytes，始终超过 1 GiB。Job 6144 MiB、主机提交余量
10240 MiB、物理余量 8192 MiB、180 秒、16 进程和 1000 ms 轮询均未放宽。

未改变产品默认解码线程配置。这是日志策略下的解码/关闭兼容性，不是完整流水线
Stop、恢复像素质量、FPS 提升或最终 60 秒产品组合验收。

主线程独立运行 CPU 回归：日志策略 11、共享源释放 8、失败详情 6、guard 16、
显存恢复 17，共 58 项通过。Terra 提供源码审查（包括上述间接导入纠正），所有
原生/GPU 实验由主线程串行执行。普通 Windows GUI 自动隔离仍未接线，不能将
本页的默认关闭后端视为普通 GUI 已完成验收。

## 独立审计与真实流水线 Stop

Terra 实现只读 accept_native_logs.py 及 8 项故障注入测试；主线程检查实际文件，
要求补足精确 guard 上限/主机余量/重复标记检查后，独立复跑 8 项及原始证据审计通过。
审计同时保留 A1/A2 的失败状态，核对 A3 全部 launch/child 源 SHA、已加载 avutil、
独立 stderr 和级别修改次数。没有把失败运行改写成成功。

真实共享 GUI worker/Processor/session/pipeline 的新 Stop 运行：
`D:\AI\jasna_windows_amd_dev\Temp\windows-product-native-pipeline-20260908-b1\native-runs\PRODUCT_NATIVE_LOGS_20260907T203936Z_22e2a2b91224`。
第 128 帧经真实 stdin 控制通道发送 Stop；2.891 秒后整个 guard 进程树正常退出。
两 reader、四工作线程、writer、全卡 sampler 全部清理。Guard completed/exit0/
active0/forced0，peak Job 6210465792 bytes，全卡最低余量 19335921664 bytes，
无 CPU offload。主线程 accept_native_stop.py 对原始事件、源码、guard、身份 token、
清理和策略最终状态独立验收通过；5 项 CPU harness 合同测试通过。

这里直接使用产品原生日志策略，研究 runner 没有 restore_default_callback 补丁。
但 FRAME2 decoder、RF-DETR bounded-leaf、小 YUV tile 仍是既有研究适配器，不是
普通 GUI 全产品后端。停止后的任务为 pending，不是完成视频，也不作质量验收。

此前 a1 的 Stop 正常返回，但继承的完成测试错误要求 1200 次 resize，导致 child
exit1；修正为区分 Stop/完成的有界调用数合同后在 b1 重新运行。a1 原始失败报告
保留，不计通过。两个版本的产品代码相同；修改仅为研究验收脚本。

## 同组合完整 600 帧输出

b1/native-runs/PRODUCT_NATIVE_LOGS_20260907T204224Z_d9e578f3aa30：完整 600 帧
Main10 修复完成，产品 resize 1200 次；日志策略最终保持原生且重定向 1 次 setter。
进程树总耗时 164.313 秒（约 3.65 E2E FPS），pipeline.run 147.342 秒（约 4.07 FPS）。
没有提速结论；比历史 product-resize600 的 149.766 秒总耗时更长，非受控交错 A/B，
不能用作因果回归判定，也不能隐去较慢结果。当前路线仍使用研究 FRAME2/检测/tile。

独立 CPU 全解码/600 帧/精确 PTS 验证通过，45.406 秒；首 PTS 2100、末 601699，
步长 1001，time base 1/60000，8192×4096 HEVC Main10，video-only。输出
69404807 bytes，SHA256 4a93730622cb99a8fd862f51544f21819257d05553f9b479de2135aa5389cb49。
验证 guard completed/exit0/active0/forced0；主线程再次核对原始验证报告、全部
身份 hash、guard 和时间轴通过。新增 6 项 CPU 固定600合同/元数据测试通过。
不是恢复像素质量或最终连续60秒含音频/可见GUI验收。

全卡峰值 7091 MiB，Torch allocated 峰值 4739 MiB，零 offload；显存低占用不是
性能优化成果。主线程复核阶段计时：detect-track 111.8s、主修复 95.8s且等队列47.1s，
blend-encode 等修复83.4s且解码57.5s。这些是并发阶段墙钟，不能相加或当成GPU kernel时间。
用户指出 Linux 常见16–22GiB显存且GPU更忙；对照其日志确认 Linux 也用B1，但有
RF-DETR/BasicVSR++ MIGraphX 以及 AMF Vulkan→HIP D2D，Windows本次则AI eager/
bounded-leaf、CPU软件解码上传。不能简单归因B1，或以填满显存替代消除阶段等待。
