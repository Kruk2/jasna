# GUI 自动预扫描路由

## 作用

GUI 的“预扫描策略”提供 `自动`、`始终扫描` 和 `关闭` 三种模式。默认
`自动` 会先按约 4 秒间隔粗扫，再按约 0.5 秒间隔精扫候选区间，并根据覆盖率选择：

- 没有可信马赛克：直接校验并复制源视频；
- 覆盖率达到默认 85%：进入完整修复；
- 其余情况：把精确区间交给现有 Smart Render 路线。

显式手工区间和显式“整片处理”优先于自动扫描，不会被预扫描覆盖。

## 一致性约束

- 扫描复用产品共享的视频 reader、检测模型注册入口和用户检测阈值，不复制
  NVIDIA/AMD 后端判定。
- Linux AMD 因而自然继承统一 PyAV/AMF 解码和 RF-DETR MIGraphX 产品选择；
  不增加 CPU fallback，也不恢复 rocDecode。
- 扫描、复制、完整修复和 Smart Render 都保留现有输出验证、隐藏暂存和
  Stop 原子提交语义。
- GUI 码率模式也是路由无关的任务合同：Linux AMD HEVC 选择“自动（匹配源码率）”后，
  预扫描无论最终选择 `full` 还是 Smart Render 都必须保留源码率 `vbr_peak`；选择手动
  CQ 时两条路线都显式使用 `rc=cqp`。CLI 完整编码及其他平台/编码格式不自动扩展。
- 自适应粗扫在同一媒体内复用一个产品 reader，并在每个 GOP 之间关闭上一条
  frame generator 后重新 seek。不能为 8K 长片的每个采样点重复创建 AMF/Vulkan
  decoder epoch；否则驱动延迟回收的 decode surface 会在粗扫结束前耗尽可导入显存。
- 精扫仍复用共享 `auto` reader，不传 scan-specific backend。仅当共享能力门确认是
  Linux AMD 原生 AMF D2D、输入不少于 3000 万像素且时长不少于 10 秒时，时间轴由
  两条 reader 分段并行；4K、短片、不支持原生 AMF 的格式和 Windows AMD 仍为单
  reader，NVIDIA 原有 4K 并行门不变。
- 精确 PTS、检测签名、执行策略和已完成分数写入 checkpoint；中断后可复用
  已完成检测结果。媒体解码仍从头开始，不宣称保存了解码器内部位置。

## 边界处理

短、低置信度候选会被过滤；持续高置信度区间保留。精扫结果按配置补边并限制
在首尾帧范围内。Windows AMD HEVC Main10 粗扫使用已验证的固定网格 reader，
避免部分读取后销毁 PyAV reader 的历史挂起。

59.94 FPS 等媒体的采样 PTS 会因时间基量化在名义间隔附近产生约毫秒级抖动。
连续命中合并允许采样步长 1%、且最多 10 ms 的邻接容差；该范围足以吸收正常
PTS 抖动，但不会跨过真正缺失的一个采样点。旧版使用近乎严格相接判断，会把
连续高置信度命中拆成不足 1 秒的孤岛，再被短候选过滤。扫描算法签名已升级为
`jasna-pre-scan-v5-pts-jitter-merge`，旧错误 checkpoint 会自动失效并重新扫描。

自动路线还会在加载修复模型之前预检 H.264 源 GOP 是否能由当前硬件编码器复现。
例如 Linux AMD AMF 最多支持 3 个连续 B 帧；遇到使用 4 个连续 B 帧的源片时，
自动路线不再于 Smart Render 启动阶段报错，而是记录不兼容原因并切换到 Full。
Full 会重编码完整时间轴，但只在扫描确认的 PTS 区间运行修复模型。手工指定区间
仍保持严格 Smart Render 语义，不会静默改变用户明确选择的路线。

## 验证范围

单元测试覆盖 Auto/Scan/Off、全空/近全/部分路由、checkpoint 签名与恢复、
短区间过滤、首尾限制、停止行为、源复制失败回退、H.264 GOP 不兼容时的自动
Full 回退，以及与最终输出原子发布的组合。真实视频验收仍需按相同素材分别检查
三种路由的时长、帧数和输出可解码性。
Linux AMD 还需检查长 GOP 计划只打开一个 AMF reader、`failed_bridge_copies=0`，
并确认正常 INFO 不再为每个采样点重复输出完整 transport JSON；完整 counters 仅在
DEBUG 诊断日志保留。

2026-08-30 的 60 秒 8K Main10 产品验收中，自动粗扫用同一 reader 完成 12 个 GOP
采样点，覆盖率 91.7% 后正确进入 full；粗扫只有一条 12/12 D2D、12/0 FD close
摘要，没有重复 decoder epoch 或 INFO JSON 刷屏。随后 3600 帧完整修复和独立输出
验收通过。完整证据见事务目录
`jasna-gui-8k60s-prescan-amf-reuse-20260830/VERIFICATION.txt`；1856 秒全片仅粗扫压力
验证仍作为独立后续范围，不与本次 60 秒完整修复混为一项。

同日的 60 秒 8K Main10 精扫公平 A/B 中，单 AMF reader 为 86.745 原始帧等效 fps，
双 AMF reader 两次为 147.231/145.624 fps，中位数 146.428 fps，耗时减少 40.76%。
两次双 reader 的 120 个有序时间、checkpoint PTS、检测分数和 mask 均与单 reader
逐字节一致；两条 reader 的 D2D、FD、fixed-context session、stream/event 全部配平，
所有 host/Map/staging/D2H/failed bridge 计数为 0。证据见事务目录
`jasna-linux-amf-dual-reader-scan-ab-20260830/VERIFICATION.txt`。

## Linux AMD 队列生命周期收口（2026-08-31）

GUI 的 Linux AMD 视频队列现在把每条视频的自动预扫描和后续 copy/full/smart 路由
放入同一个独立 child 进程。这样扫描模型、两条 AMF reader、修复模型和编码器不会
跨视频留在长驻 GUI，也不会改变 `DEFAULT`/`MANUAL`/`FULL` 的既有优先顺序。扫描
checkpoint、用户检测阈值、B4/B8 和 VR auto 解析仍按完整任务设置序列化给 child；
父进程只在 child 退出、输出路径/新鲜度/媒体校验通过后提交完成状态。

120 帧 8K Main10 实片连续双任务中，两次都保留 1 个可信精扫区间、覆盖率 100% 并
选择 full；child PID 不复用，退出后显存均回到约 1.65 GiB。四条 reader audit/child
分别为扫描 12/16 和完整处理 120/120，cache 默认关闭，host/Map/staging/D2H/failed
bridge 全为 0。两个输出严格解码、120 帧、PTS 和 hash 验收一致。详细成功证据和一次
独立记录的随机启动停顿见
`transactions/jasna-8k-vram-fix-20260831/VERIFICATION.txt`，不能用成功的双任务记录
覆盖该已知观察项。
