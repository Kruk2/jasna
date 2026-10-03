# Linux AMD AMF 私有 deferred event-pool 实验记录

本文只记录 `scripts/amf_surface_probe.pyx` 的 decode-side 私有 HIP stream
实验。该提交当时不是产品默认解码策略，也不改变 `auto`、`pyav-hw`、`pyav-sw`、
VALI 或当时的兼容后端 policy；后续产品 PR 已接入合格 H.264/HEVC，独立删除 PR
又移除了旧兼容后端。

## 显式入口与回退边界

只有同时设置以下两个变量才会走 deferred 路径：

```bash
JASNA_DECODE_BACKEND=amf-interop
JASNA_AMF_INTEROP_DECODE_COPY_STREAM=private-deferred
```

显式 `amf-interop` 未设置第二个变量或设为 `null` 时，仍使用已有的
`null-stream-source-release` 路径。后续产品路由只对 Linux AMD 合格的 H.264/HEVC
在 `auto` 下固定选择 `private-deferred`；AV1 和其他格式不在这项自动迁移内。bridge
缺少 dependency probe、session 方法、非空 Torch HIP stream handle 或任何必需 HIP
entry point 时，都会 fail closed，不切到 CPU、host map、D2H、staging 或软件回退。

`JASNA_AMF_INTEROP_RESOURCE_CACHE` 仍默认关闭；显式 true 仍会在打开前被拒绝。
这里的每帧 Vulkan external-memory import/map 没有变成 cache，只是其释放时刻从 copy
函数返回前延后到 consumer event 确认以后。

## 三槽 ownership 图

每个 reader 固定创建一个 `hipStreamNonBlocking` producer stream，并预创建三个 slot。
每个 slot 持有两个可复用且禁止 timing 的 HIP event：producer copy 完成 event 与
consumer dependency 完成 event。因此完整 pool 是 3 × 2 = 6 个 event；不是每帧创建/销毁
event，也绝不能扩大到第四个 retained source。

```text
AMF source Acquire + per-frame import/map
             |
             v
private producer stream --[copy-complete event]--> Torch current consumer stream
             |                                         |
             |                                  YUV -> RGB conversion
             |                                         |
             +----------------[consumer-complete event]-+
                                                       |
                              event query / forced event synchronize
                                                       |
                       release AMF source + free map + destroy import
```

正常提交先用 `hipEventQuery` 回收已经完成的最旧 slot；若仍达到三个 retained source，
只对最旧 consumer event 做一次 `hipEventSynchronize` 形成有界 backpressure。关闭时先
force-drain 所有 retained slot，再销毁六个 event 和 producer stream。正常每帧提交不调用
`hipStreamSynchronize` 或 `hipDeviceSynchronize`。错误路径会先证明 queued work 已结束；
若无法证明安全，则保留 AMF source/mapping ownership 并报错，绝不在 work in flight 时释放。

## B8 reader 边界

reader 打开时创建并用 dependency probe 验证一个 **非默认** Torch stream；
`AmfInteropUploader` 在该 stream context 中完成分配、copy 和 conversion。bridge 让
这个同一 stream 等待 producer event，再立刻在该 stream 记录 consumer event，随后
`YuvToRgbConverter.convert_into` 才排队。不能重新读取默认 Torch stream：HIP 的合法
legacy/default stream handle 是 `0`。组末的现有 stream synchronize 仍只是 B8 batch
交给下游前的 conversion handoff，不是 deferred source-release 同步。

上一组 Python AMF frame 引用仍在读取下一组之前清空；不跨 yield 预取未消费的 native
surface。这一点避免 decoder surface pool 在 B8 batch 下停滞。

## 明确拒绝的变体

- 每帧创建/销毁 event：会把 3,600 帧变成大量 HIP allocation，且没有改进依赖图。
- null stream 或 device-wide synchronization 作为 deferred 实现：会破坏该模式的目标。
- timeout 轮询作为 slot 回收：不是有界 producer/consumer 依赖证明。
- 四个或更多 retained source：会侵占 AMF decoder 为 B8/submit 预留的 surface。
- cache-on、host transfer、CPU map、D2H 或 staging：超出本实验边界。

## 验证范围与后续接口

焦点测试在 `tests/test_amf_interop_core.py`，覆盖 env 显式性、桥接/probe fail-closed、
非空 consumer stream、每帧 telemetry、三槽 event-pool 计数、禁用 host/non-D2D
transport、B8 handoff 以及 teardown 不覆盖原始 copy 异常。

目标 3,600 帧会话的正常闭环计数是：stream create/destroy `1/1`，event
create/destroy `6/6`，event record `7200`，device wait `3600`，source acquire/release
`3600/3600`，max/final in-flight `3/0`。显式 mode 继续保留为诊断接口；产品 `auto`
迁移的范围和失败边界见 `docs/LINUX_AMD_AUTO_DECODE_CN.md`。

当前工作树已用 accepted unified runtime 重新在源码树外构建 bridge，SHA-256 为
`821cdde7f048e699c7911a28e875a712f7538b6aef10b4f2aa7c8368f7471773`。RX 7900 XTX
上以 B8、cache off 完整读取静态 H.264 High 8-bit 与 HEVC Main 8-bit fixture，均为
120/120 帧，PTS 与独立 FFprobe 完全相同且严格递增。每个 reader 的 stream
create/destroy 为 `1/1`、event create/destroy 为 `6/6`、event record 为 `240`、
device wait 为 `120`、source acquire/release 为 `120/120`、final in-flight 为 `0`；
Vulkan export、HIP import/map/release/destroy 全部为 `120`，cache hit/miss 为 `0/0`。
host transfer、CPU Map、staging、D2H、non-D2D、failed bridge 均为 `0`。运行窗口没有
GPU reset、ring timeout、page fault 或 OOM；结束时 junction 64°C、memory 66°C。
