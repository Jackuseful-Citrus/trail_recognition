# 自由尺寸矩形 Planner 离线验证报告

## 审计基线

- 当前部署只保留 CanMV 原生碎片识别、固定图 2 快速匹配和 free-rect
  planner。OpenCV/NumPy 碎片识别、逐片摆放识别以及旧 planning 1/2/3
  已从模块源文件和 free-rect standalone 中删除。
- `puzzle_simulator_planner.py` 现在只保存当前 free-rect planner 仍调用的
  edge-match、edge-align 和 pose-graph 底层原语，不再包含独立 planner
  入口。
- builder 只生成当前 free-rect standalone，不再提供 planner backend
  切换参数。

## 实现

新增 `puzzle_simulator_free_rect_planner.py`，入口为：

```python
plan_simulator_free_rectangle(pieces, validation="publish_best")
```

自由尺寸后端使用独立的 `FREE_RECT_*` 配置和独立的：

- full/partial 候选保留 shortlist；
- 1～4 片连通生成树搜索；
- 默认 8000 ms（实时配置 20000 ms）时间预算和 6000 个完整 matching set
  上限；
- top-5 完整方案；
- publish-best 结果语义；
- 基于现有 seam 分数区间的外露周长判据；
- 自由尺寸 metrics、cost、A4 下半区目标姿态和日志。

搜索 prefix 硬拒绝候选/物理边复用、非连通扩展、无效刚体几何和超过
170 mm 的灾难性跨度。完整 matching set 生成初始刚体姿态后，会先用
`Σ碎片周长 - Σ各物理边已覆盖区间` 估计拼合外露周长；外露周长若比
同面积、允许长宽比矩形的最大周长高 18% 以上，不进入昂贵的 pose graph
和面积计算。Overlap、gap、outside 和矩形尺寸仍不参与 prefix hard
reject。

周长计算不做 polygon union。选中 seam 已有的两侧边索引和分数区间直接
作为覆盖区间；另外只在最多 20 条变换后碎片边之间做一次 O(E²) 共线
接触扫描，补齐生成树未记录的闭环接缝。每条物理边上的区间先合并再扣
除，避免 partial/T-junction 或重复接触被重复计数。

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
+ 12 * perimeter_error_ratio
```

其中 `perimeter_error_ratio` 是外露周长超出同面积、允许长宽比矩形周长
区间的相对误差；区间内为 0。

最终只做旋转和平移，不缩放碎片。目标矩形长边水平放置，中心为现有
`TARGET_CENTER_MM`；在等价方向中用运动距离和旋转量选择机械代价较低
且能放入 A4 下半安全区的姿态。

### 固定图 2 四片直接路径

free-rect 在通用候选生成之前，先识别题目图 2 的四块固定碎片。识别
使用固定的面积占比（8%、40%、18%、34%）、一个三角形加三个四边形
的拓扑，以及各轮廓的刚体拟合误差。现场轮廓存在镜像手性，因此保存
两套常量模板：

- `NORMAL`：题图方向；
- `MIRROR_X`：水平镜像方向。

这里的面积和轮廓阈值只负责判断“是不是这套固定碎片”，不是最终方案
安全门限。匹配后直接使用预存的 100×60 mm 目标矩形和四个固定目标
中心，逐片计算当前中心到目标中心的刚体旋转和平移，不运行：

- edge candidate 生成；
- matching-set 枚举；
- pose graph optimization；
- overlap/gap/outside 最终 safety gate。

命中结果的 `search_nodes`、`candidate_count`、
`complete_matching_set_count` 和 `pose_optimization_count` 均为 0。
Overlap、gap 和 outside 仍计算并写入结果，但仅供诊断，不否决输出。

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

自由尺寸固定图 2 直接路径结果：

| 项目 | 结果 |
|---|---:|
| valid | true |
| template layout | `MIRROR_X` |
| max area-ratio error | 0.0099 |
| max template RMS | 2.44 mm |
| max template vertex error | 3.46 mm |
| candidates / complete sets | 0 / 0 |
| pose optimizations | 0 |
| target size | 100 × 60 mm |
| target center | (105, 225) mm |
| safety gates | skipped |

该 fixture 的四片目标中心由 `MIRROR_X` 常量模板直接给出：

| 固定角色 | fixture piece | target center (mm) |
|---|---|---:|
| TOP_LEFT | P4 | (141.33, 204.00) |
| RIGHT_TRIANGLE | P1 | (81.67, 215.00) |
| MIDDLE_LEFT | P3 | (121.89, 221.78) |
| BOTTOM_LEFT | P2 | (114.96, 243.41) |

## CPython A/B

同一 frame 33 fixture，固定图 2 直接路径运行 20 次的墙钟中位数为
0.42 ms（本机范围 0.39～1.09 ms）：

| Planner | valid | 中位耗时 | candidates | complete sets | pose optimizations |
|---|---:|---:|---:|---:|---:|
| 固定 `simulator` 历史基线 | true | 50.7 ms | 80 | 320 | 1 |
| `simulator_free_rect` 图 2 直接路径 | true | 0.42 ms | 0 | 0 | 0 |

因此，已知图 2 四片不再受 8000 ms 搜索预算影响，也不会因枚举数量过
多而在时间到期前漏解。非图 2 输入仍回退到原有自由尺寸枚举路径。

## 测试

自由尺寸专项测试覆盖：

- 90×50、100×60、110×70、120×90；
- 1、2、3、4 片；
- full seam 与 partial/T-junction seam；
- 选中 seam 与闭环补充 seam 的覆盖区间合并；
- 开链错误拼法的外露周长预筛；
- 任意初始旋转和平移；
- 不缩放和 S/T/R 几何一致性；
- ±1～2 mm 顶点噪声；
- frame 33 fixture 与固定输出快照；
- 图 2 `NORMAL`/`MIRROR_X` 模板识别；
- 用会主动抛错的 candidate generator 证明命中时没有进入枚举；
- 固定目标中心、模板角色、0 搜索节点和 safety gate 跳过标记；
- timeout best-so-far 与 timeout-before-complete；
- 同一输入连续 5 次的结果、统计和日志确定性；
- MicroPython 依赖检查、语法编译和 standalone 构建。

当前维护路径共 77 passed：planner/识别/source-clear/debug/protocol 49
项，A4 实时运行状态 28 项。free-rect planner 自身 14 项，其中新增真实
日志几何回归：6000 个 complete set 中周长预筛拒绝 5914 个，只运行 86
次 pose optimization，并找到带 1 条闭环补充接缝的方案。

## Standalone

构建命令：

```bash
python k230_realtime_a4/build_standalone.py
```

生成文件：

`k230_realtime_a4/k230_realtime_a4_simulator_free_rect_standalone.py`

SHA-256：

`974eb3ab1500543e7ce8a26d5320017cad167f1e8ce14ec8efb062663c2729b9`

生成文件已通过 Python 语法编译和 import AST 检查，不包含 NumPy、
OpenCV 或 dataclasses 依赖。文件大小为 466745 bytes、13828 行；旧识别
和 planning 入口的函数级审计计数均为 0。

板端验证应在 CanMV IDE 中上传并运行新的 free standalone，不执行任何
机械动作前先检查以下日志序列：

```text
START_REALTIME_A4,...planner=simulator_free_rect,...
PLANNING_INPUT_BYPASS,...reason=fixed_figure2_direct,border_blobs=...
FREE_FIXED_TEMPLATE_CHECK,matched=1,layout=MIRROR_X,...
FREE_FIXED_TEMPLATE_PIECE,role=...,piece_id=...,source_x=...,target_x=...,rotation_deg=...
FREE_FIXED_TEMPLATE_BYPASS,enumeration=SKIPPED,safety_gates=SKIPPED,target_mm=100.0x60.0,...
FREE_FIXED_TEMPLATE_RESULT,valid=1,mode=simulator_free_rect_figure2_direct,...nodes=0,...
OPERATION,piece_id=...,template_role=...,source_x=...,source_y=...,target_x=...,target_y=...,rotation_deg=...
```

若四片未匹配图 2，预期打印
`FREE_FIXED_TEMPLATE_CHECK,matched=0,...action=FALLBACK_TO_ENUMERATION`
并进入原来的 `FREE_PLAN_START` 路径。通用路径的 `FREE_PLAN_PROGRESS`
会额外显示 `perimeter_passed`/`perimeter_rejected`，最终结果显示
`exposed_perimeter_mm`、期望周长区间、误差与闭环补充接缝数。若该回退
路径在配置时间预算到期但
已有完整方案，预期
`FREE_PLAN_RESULT,valid=1,timed_out=1`。若到期前没有完整方案，预期
`FREE_PLAN_INVALID,reason=no complete candidate before timeout,...`。

## 尚未解决

- 同一条长物理边目前不能被多个互不重叠的 partial 区间复用；严格的
  physical-edge reuse gate 会拒绝这种拓扑。
- 尚未加入扑克牌图案或 seam 视觉匹配。
- K230 上尚需用真实相机复核 `NORMAL`/`MIRROR_X` 手性、旋转角正负号
  和四个固定目标中心；这不影响离线证明“命中后没有枚举”。
- 未匹配固定图 2 的通用 free-rect 输入仍可能依赖 8000 ms timeout 的
  best-so-far。
