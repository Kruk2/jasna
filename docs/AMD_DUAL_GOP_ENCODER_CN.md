# Linux AMD HEVC 双 GOP 编码

`AmdDualGopFrameWriter` 把独立 closed GOP 交替交给两个持久 AMF 编码会话，
按原始顺序组装结果。它与 Windows 原生 split-frame 是不同的实现。

GUI 默认预设请求双 GOP；CLI 必须显式传入 `--amd-dual-gop-encode`。
自动准入按最终 HEVC 输出合同选择：Main10/P010 至少 3840×2160 像素，
Main/NV12 至少 5760×2880 像素，尺寸为正偶数且源码率有效。Full 输出可以由
兼容的 H.264 等输入转为 HEVC；Smart Render 会复制源包，因此还要求 HEVC
源流/profile 兼容。不准入的 GUI 任务沿用正常单会话编码。

队列、pinned staging、native surface、异步 D2H 与失败关闭均有明确生命周期。
长任务不得积累整片 GOP 或重复使用仍由编码器/复制事件持有的帧。其测试覆盖
PTS、顺序、错误传播及有界资源；真实编码收益按同源、同范围、同设置的既有
验收记录判断，不能拿旧的回收前 FPS 与新版本混比。

相关调用点在流水线和 GUI 集成 PR，源率/码流合同在编码器及 Smart Render PR。
Windows 不会自动启用这条 Linux 双 GOP 路线。性能探针不改变产品默认。
