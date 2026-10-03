# 共享软件解码源生命周期验收

2026-09-08，目标仍是完整 Windows AMD 路线与 GUI；本文仅验收源帧释放改动。

## 产品改动

`jasna/media/video_decoder.py` 使用共享路径，不新增 Windows 解码算法副本：

- `_decoded_frames` 在 yield 前从 packet 解码结果列表移除对应引用，避免已被下游
  消费的帧继续由整个 packet list 持有。seek 丢弃、PTS 和错误处理规则不变。
- `_frames_software` 的 CPU 源平面已同步复制入 pinned staging 后，释放最后一组
  PyAV frame、normalized frame、frombuffer 视图及 group，再预取下一批。
- pinned/device staging 保持存活至 GPU 同步；硬件 mapped surface 路径不提前释放。
- 不改变线程策略、解码后端、YUV数学、native callback策略或GUI默认选择。
  FRAME2依赖的日志/worker隔离产品化仍待完成。

改动前实际产品源SHA256：
`51bce9fef9639d54ad5d5568096ac1ec9fad0f57646fb338370cfdaf26856543`。
本次产品源SHA256：
`9cfbd410fe832157ede8314c763668ddd7767af05d023e16853b892ae546be6f`。

## 主线程验收

8项纯stdlib测试直接从实际产品源抽取方法，覆盖：AMD/NVIDIA软件回退的10bit
同步CPU复制、H2D/prefetch/sync顺序、weakref源释放、pinned及下一组存活、部分批次、
空输入、packet列表消费释放、seek/flush、无PTS帧和生成器关闭。

真实证据目录：
`D:\AI\jasna_windows_amd_dev\Temp\windows-product-decoder-lifetime-20260908-a1`。
`PRODUCT_DECODER_probe_A1` 对比上述两个精确SHA对应的解码器，用同一个已准入的
60帧/1.001秒 native8K Main10副本；不是原始长视频，也不是60秒片段。

两边都为 pyav-sw、SLICE1、相同native默认日志回调，保留产品当前YUV转换数学。
逐批 `torch.equal` 比较GPU RGB，没有把整帧复制回CPU进行比对：

- 完整60帧，B1，PTS严格为2100+1001*n；
- 从第30帧seek、stride3，10帧；
- 从第57帧seek、B4，尾部实际3帧；
- B1处理4帧后主动关闭生成器/reader。

全部77帧RGB逐元素与PTS精确一致，所有reader的container/video_stream/decoder_ctx
关闭后均为空。GPU同步、全卡reader关闭通过。
主线程独立 `accept_decoder_lifetime.py` 核对报告与日志一致、所有当前源pin、
严格样本/时间戳序列、逐批资源采样位置、原始guard命令/额度/结果通过。
首次核验器把B4尾部一次采样误计为3次，已改为按实际批次逐项核验标签序列，
没有修改native报告、放宽数据/资源要求或重跑native任务。

child耗时43.5秒；guard完成44.314秒，退出0、active0、forced0；峰值Job
4109815808字节，小于6144MiB，最小主机提交/物理余量24728670208/19038756864字节。
全卡显存最小空闲20833972224字节，大于1GiB；180秒上限、1秒poll与16进程上限未变。

本项由主线程实现并独立运行全部检查。Terra同时只读审计GUI隔离缺口，未参与
此次native运行。它证明源生命周期改动的行为一致，不证明整片FPS、修复像素画质、
FRAME2默认安全性或可见GUI；后续完整60秒验收必须记录本次新源，不能沿用旧pin。
