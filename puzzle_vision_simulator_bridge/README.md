# puzzle-vision-simulator 桥接层

这个目录把
[`lvreng/puzzle-vision-simulator`](https://github.com/lvreng/puzzle-vision-simulator)
固定到提交 `e9eb2e0fb945c348eedd0b0fa9258f5518d2892f`，并把它的
NumPy/OpenCV 拼接结果转换成当前项目的 `puzzle_geometry.PlanResult`。

桥接层复用当前框架的 A4 透视校正、碎片检测、毫米坐标、结果渲染和安全门限；
上游只负责根据已经检测出的多边形求拼接变换。因此同一组
`PieceObservation` 可以在当前固定尺寸规划器、当前 `outer_first` 规划器和上游
规划器之间直接比较。

## 目录说明

- `adapter.py`：毫米/像素、变换矩阵、碎片 ID、接缝和 `PlanResult` 适配。
- `upstream_loader.py`：校验提交后动态加载上游 `puzzle_sim.py`。
- `fetch_upstream.py`：下载锁定版本到被忽略的 `.upstream/` 依赖目录。
- `run_offline.py`：当前视觉检测 + 上游规划的完整离线入口。
- `benchmark.py`：同一真实照片、同一检测多边形的规划耗时对比。
- `test_adapter.py`：桥接、版本锁定、本地门禁和回退路径测试。
- `PERFORMANCE_ANALYSIS.md`：为什么默认路径慢、哪些比较口径有效。
- `benchmark_result.json`：本机本次可复核计时记录。

## 安装上游依赖

```bash
python3 -m puzzle_vision_simulator_bridge.fetch_upstream
```

也可以使用已有检出：

```bash
export PUZZLE_VISION_SIMULATOR_ROOT=/path/to/puzzle-vision-simulator
```

默认严格要求检出提交等于锁文件中的提交。上游检出的仓库没有根目录
`LICENSE` 文件，所以本目录不把其源码复制进当前仓库，而是在运行时下载并保持
独立 Git 来源。

## 运行

安全的混合入口会先校验上游提案；提案不满足本地几何门限时，回退到已经更快且
更准确的本地 `outer_first`：

```bash
python3 -m puzzle_vision_simulator_bridge.run_offline \
  sample_puzzle.jpg --fallback-outer
```

只观察上游原始判定（仅用于桌面仿真/分析，不应直接控制机械臂）：

```bash
python3 -m puzzle_vision_simulator_bridge.run_offline \
  sample_puzzle.jpg --validation upstream
```

保持当前安全门限并拒绝不合格的上游提案：

```bash
python3 -m puzzle_vision_simulator_bridge.run_offline \
  sample_puzzle.jpg --validation local
```

基准与测试：

```bash
python3 -m puzzle_vision_simulator_bridge.benchmark \
  --rounds 7 --fixed-rounds 3 \
  --output puzzle_vision_simulator_bridge/benchmark_result.json
python3 -m unittest -v puzzle_vision_simulator_bridge.test_adapter
```

## Python 接口

```python
from puzzle_vision_simulator_bridge import plan_with_upstream

plan = plan_with_upstream(
    stable_piece_observations,
    cut_mode="auto",
    validation="local",
)
if plan.valid:
    consume_existing_plan_result(plan)
```

返回值就是当前框架的 `PlanResult`，`operations` 还额外包含
`matrix_3x3_mm`。真实相机输入应使用 `cut_mode="auto"`；只有切割类别来自独立、
可信的现场输入时才应指定类别。

## 部署边界

上游依赖 CPython、NumPy 和 OpenCV，不能直接放进当前 CanMV MicroPython 固件。
这个桥接目录用于桌面验证、算法对比和混合规划实验。K230 板端应继续使用
`puzzle_geometry.plan_outer_first_rectangle()`；它在同输入基准中比上游更快。

