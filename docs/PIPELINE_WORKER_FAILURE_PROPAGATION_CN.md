# Pipeline worker 失败传播

更新时间：2026-08-29

四段处理线程（decode/detect、primary、secondary、blend/encode）以及 async secondary
现在使用同一失败合同：第一个真实异常被记录，并立即设置共享 cancel event；后续由取消
产生的派生异常不会覆盖根因。

主线程不再对每个 worker 做无期限阻塞 `join()`。等待期间若已取消，会持续清空四条有界
队列，让可能卡在 `put()` 的生产者获得空间并退出；健康运行时完全不触碰队列。async
secondary 的内部 pusher 改为有超时的 `get()`，因此失败/取消时不会永久等 sentinel。

最终仍由主线程在完成资源回收后重新抛出第一个 worker 异常。用户主动 Stop 不会被记录成
worker failure，也不会生成伪根因。

验收覆盖首错优先、用户取消、健康 worker 不 drain、有界队列生产者释放、async secondary
取消路径与 pipeline 异常回抛；Linux 聚焦回归 77 passed。
