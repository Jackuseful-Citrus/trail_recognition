# 自由尺寸矩形 Planner 离线验证报告

## 审计基线

- 审计时工作树干净，分支为 `main`，HEAD 为
  `e3fc85ba738562952138872d0a68aa8d14f20889`。
- `ad768d8` 修改了固定 simulator Planner、配置、测试和当时的
  standalone；当前 HEAD 的 `puzzle_simulator_planner.py` 已不再包含该
  commit 中的 prefix gate/branch-limit 实现。
- 本任务未修改 `puzzle_simulator_planner.py`，也未修改固定
  `100x60 mm` 配置默认值、固定搜索预算或固定 simulator standalone。
- 审计时，现有固定 standalone 可由当时的模块源文件逐字节重建，
  SHA-256 为
  `04d4004a63d8b13a71ebb89096b9a5e22612912eb5aadabac4ac5b2e3cebe8a6`。

## 实现

新增 `puzzle_simulator_free_rect_planner.py`，入口为：

```python
plan_simulator_free_rectangle(pieces, validation="publish_best")
```

自由尺寸后端使用独立的 `FREE_RECT_*` 配置和独立的：

- full/partial 候选保留 shortlist；
- 1～4 片连通生成树搜索；
- 8000 ms 时间预算和 6000 个完整 matching set 上限；
- top-5 完整方案；
- publish-best 结果语义；
- 自由尺寸 metrics、cost、A4 下半区目标姿态和日志。

搜索 prefix 只硬拒绝候选/物理边复用、非连通扩展、无效刚体几何和
超过 170 mm 的灾难性跨度。Overlap、gap、outside 和矩形尺寸不参与
prefix hard reject。每个完整方案先生成刚体姿态并进行 pose graph
optimization，之后才计算 overlap、gap、hull gap 和最小面积矩形。

完整方案 cost 为：

```text
10 * overlap_ratio
+ 6 * fill_gap_ratio
+ 2 * hull_gap_ratio
+ 3 * area_prior_error
+ 3 * dimension_range_penalty
+ 2 * outer_piece_missing_ratio
+ 1 * seam_cost
+ 1 * closure_cost
```

最终只做旋转和平移，不缩放碎片。目标矩形长边水平放置，中心为现有
`TARGET_CENTER_MM`；在等价方向中用运动距离和旋转量选择机械代价较低
且能放入 A4 下半安全区的姿态。

## frame 33 回归 fixture

正式 fixture：

`legacy/fixtures/frame_33_free_rect_regression.json`

当前固定 simulator（`upstream`）基线：

| 项目 | 结果 |
|---|---:|
| valid | true（带 safety/local warning） |
| candidates | 80（full 7 / partial 73） |
| complete sets | 320 |
| selected partial | 1 |
| fixed score | 0.2629428801 |
| overlap | 183.5939 mm² |
| fixed-target fill gap | 402.5354 mm² |

自由尺寸默认预算结果：

| 项目 | 结果 |
|---|---:|
| valid | true |
| complete sets | 6000 |
| pose optimizations | 6000 |
| selected topology | 2 full + 1 partial |
| free cost | 1.3624939271 |
| inferred size | 106.6039 × 60.5837 mm |
| overlap | 183.5939 mm² |
| free MBR fill gap | 604.4566 mm² |
| hull gap | 315.9 mm² |

该 fixture 在自由路径上产生完整 matching sets 和 best proposal；
`plan_stats` 中没有 prefix-overlap prune。

## CPython A/B

同一 frame 33 fixture，各运行 3 次，取墙钟中位数：

| Planner | valid | 中位耗时 | complete sets | polygon intersections | AABB rejects |
|---|---:|---:|---:|---:|---:|
| 固定 `simulator` | true | 50.7 ms | 320 | 1,393 | 537 |
| `simulator_free_rect` | true | 5,909.1 ms | 6,000 | 24,881 | 11,119 |

自由模式按要求优化并完整评分每个完整方案，因此明显慢于只对最终
固定目标 best 做全局优化的现有路径。

## 测试

新增 8 个自由尺寸专项测试，覆盖：

- 90×50、100×60、110×70、120×90；
- 1、2、3、4 片；
- full seam 与 partial/T-junction seam；
- 任意初始旋转和平移；
- 不缩放和 S/T/R 几何一致性；
- ±1～2 mm 顶点噪声；
- frame 33 fixture 与固定输出快照；
- timeout best-so-far 与 timeout-before-complete；
- 同一输入连续 5 次的结果、统计和日志确定性；
- MicroPython 依赖检查、语法编译和 standalone 构建。

按各 legacy 测试所需工作目录分组运行的总结果为：

- 108 passed；
- 4 skipped；
- 1 个既有 legacy 归档测试失败。

唯一失败为
`legacy/test_k230_a4_recognition_test.py::test_standalone_contains_only_a4_runtime`，
原因是归档 builder 读取已经不存在的
`legacy/puzzle_config.py`。该失败在本任务修改前已存在，且不属于当前
runtime/standalone 构建路径。当前相关回归集为 106 passed、4 skipped。

## Standalone

构建命令：

```bash
python k230_realtime_a4/build_standalone.py \
  --planner-backend simulator_free_rect
```

生成文件：

`k230_realtime_a4/k230_realtime_a4_simulator_free_rect_standalone.py`

SHA-256：

`82561d05279dc56300d5bfe84be04cb810095b4d311aad9407bacd1adcd42004`

生成文件已通过 Python 语法编译和 import AST 检查，不包含 NumPy、
OpenCV 或 dataclasses 依赖。固定 standalone 未被覆盖，其 SHA-256
保持不变。

板端验证应在 CanMV IDE 中上传并运行新的 free standalone，不执行任何
机械动作前先检查以下日志序列：

```text
START_REALTIME_A4,...planner=simulator_free_rect,...
FREE_PLAN_START,pieces=...,source_area_mm2=...,candidates=...,full=...,partial=...
FREE_PLAN_PROGRESS,elapsed_ms=...,complete_sets=...,best_cost=...
FREE_PLAN_RESULT,valid=1,timed_out=...,complete_sets=...,cost=...,long_mm=...,short_mm=...
OPERATION,piece_id=...,source_x=...,source_y=...,target_x=...,target_y=...,rotation_deg=...
```

若 8000 ms 到期但已有完整方案，预期
`FREE_PLAN_RESULT,valid=1,timed_out=1`。若到期前没有完整方案，预期
`FREE_PLAN_INVALID,reason=no complete candidate before timeout,...`。

## 尚未解决

- 同一条长物理边目前不能被多个互不重叠的 partial 区间复用；严格的
  physical-edge reuse gate 会拒绝这种拓扑。
- 尚未加入扑克牌图案或 seam 视觉匹配。
- K230 上的真实搜索吞吐尚未测量。CPython 默认完整搜索约 5.9 秒，
  因而板端很可能依赖 8000 ms timeout 的 best-so-far；必须现场测量
  complete-set/pose-optimization 吞吐后再决定是否调整独立预算或搜索
  排序。
