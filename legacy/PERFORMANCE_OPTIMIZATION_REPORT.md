# K230 拼图识别与几何规划性能优化报告

日期：2026-07-29  
桌面样例：`sample_puzzle.jpg`  
板端固件目标：CanMV v1.6 / K230 V3P0

## 1. 结论

本轮按 Phase 0～6 完成了受控优化。原有 25 项测试全部通过，新增
40 项测试全部通过，共 65/65。实时使用的未知矩形规划器在同一桌面样例上从
152.494 ms 降至 44.543 ms；DFS 节点从 2358 降至 1120；精确多边形相交调用
从 14486 降至 2610。最终仍识别 4 块、顶点数保持为 3/4/4/4，并生成
100 mm × 60 mm 等价矩形解。

这里没有真实 K230 性能测量，因此没有声称板端 FPS、帧耗时或内存改善。单文件已经
重建并通过 CPython 语法及禁止 API 静态检查，仍需在已连接的 K230 上采集 `[PERF]`
日志。

## 2. 修改文件

| 文件 | 修改内容 | 原因 |
| --- | --- | --- |
| `puzzle_config.py` | 拆分轮廓、接缝、最终验收、几何、搜索、实时和图案配置 | 避免 20 mm 最终位置容差污染接缝筛选 |
| `puzzle_perf.py` | 增加 CPython/MicroPython 兼容计时、窗口平均值和计数器 | 提供可关闭的板端阶段耗时证据 |
| `puzzle_vision.py` | Moore 邻域有序边界跟踪、闭合轮廓简化、凹角保留、线拟合、扫描线覆盖率 | 把面积级洪泛热路径改为近似周长级，并消除复检逐点 PIP |
| `puzzle_geometry.py` | 边描述符和候选图、变换缓存、AABB、凹多边形三角剖分、MRV、搜索上限和统计 | 不再在每个 DFS 节点重建边候选，减少精确相交和搜索展开 |
| `puzzle_realtime_state.py` | 纯状态调度、倒计时分桶、UI 内容键、COMPLETE 视觉开关和可重置相位对象 | 使显示策略可独立测试 |
| `puzzle_image_strip.py` | 非背景 RGB 掩膜、`EdgeImageStrip`、灰度/梯度代价、可选候选重排 | 为扑克牌模式提供默认关闭的接口基础 |
| `k230_realtime_a4/k230_realtime_a4.py` | PLANNING 静态提示、PLACING 降频、状态驱动显示、扫描线缓存、COMPLETE 停止视觉 | 对齐实际逐片搬运流程并避免静态画布反复全屏绘制 |
| `k230_puzzle_planner.py` | 输出 `PLAN_PERF` 结构化规划统计 | 板端可直接看到候选、节点和相交调用量 |
| `offline_validate_puzzle.py` | JSON 增加 `plan_stats` | 保留机器可读性能证据 |
| `build_k230_standalone.py`、`k230_realtime_a4/build_standalone.py` | 合并新增共享模块 | 保持 CanMV IDE 单文件部署 |
| 两个 `*_standalone.py` | 由构建器重新生成 | 板端无需本地模块或 REPL |
| `performance_baseline.json`、`performance_optimized.json` | 保存优化前后环境、调用量和桌面计时 | 结果可复核 |
| 新增及更新的 `test_*.py` | 覆盖边界、候选图、AABB、凹多边形、扫描线、UI、边带和性能门禁 | 防止优化破坏正确性 |

`test_camera.py` 未修改。

## 3. 算法变化

### 3.1 轮廓

旧 CanMV 路径会在连通域内部做 Python 洪泛，再收集边界，工作量接近连通域面积。
新路径从第一个前景边界点开始执行 Moore 邻域有序跟踪，只访问边界邻域，正常工作量
接近周长。它记录 `boundary_steps`、`pixel_reads` 和最大步数保护；跟踪失败时才显式
回退旧洪泛，并增加 `boundary_fallback_count`，异常轮廓不会静默进入规划器。

有序轮廓先用闭合 Douglas-Peucker 简化，再用相邻边拟合和交点细化。代码不再无条件
取凸包：只有轮廓已经证明为凸形，或显式打开 `FORCE_CONVEX_CONTOURS` 时才做凸稳定化，
因此凹五边形的凹槽、面积和接缝不会被填掉。

### 3.2 一次性边候选图

每块 3～5 条边先转成 `EdgeDesc`。只枚举不同碎片间的边对，使用独立的
4 mm 绝对误差和 5% 相对误差筛选。每个合法边对只计算一次：

- 两个方向的刚体变换；
- 旋转和平移；
- 长度、端点角和综合几何代价；
- 两个方向的变换后顶点、AABB、面积和三角形。

DFS 通过按碎片对或开放边建立的索引复用这些对象，不再逐节点重复遍历全部新边、计算
三角函数和变换多边形。最终 20 mm 机械验收容差不会改变候选数量。

### 3.3 AABB 与凹多边形

每个观测多边形缓存 AABB、凸性和 ear-clipping 三角剖分。精确相交之前先检查 AABB；
明显分离时立即返回零面积。凸多边形仍走原裁剪路径；凹简单多边形分解成三角形后累加
相交面积。自交、退化或三角剖分面积不守恒时 fail-closed，不能构造
`PieceObservation`。共边接触保持零重叠。

### 3.4 搜索与矩形约束

近直角证据和碎片总面积产生矩形尺寸假设，数百个原始假设只展开评分最优的 12 个。
角点混合搜索和无角通用 DFS 都复用候选图。MRV 只调整“合法候选最少的碎片”展开顺序，
不会把其他碎片分支删除。每次添加后立即检查：

- 拼接边两侧关系；
- AABB 后的真实重叠；
- 每片是否仍保留水平/竖直外边候选；
- 部分包围尺寸；
- 已使用接缝和量化状态去重。

统一限制为 `MAX_DFS_NODES=1200` 和 `MAX_PLAN_TIME_MS=3000`，循环中保留
`os.exitpoint()`，避免板端搜索或 IDE Stop 无响应。

### 3.5 实时流程与 UI

- `ACQUIRE`：每 2 帧检测 A4，保留相机反馈和稳定识别。
- `PLANNING`：冻结输入，先显示一次静态 `PLANNING...`，DFS 内不绘图。
- `PLACING`：A4 改为每 8 帧检测；完整碎片识别与覆盖率只在 5 秒复检点运行。
- `COMPLETE`：最终画面显示后不再 `snapshot()`，也不执行 A4、碎片或覆盖率分析，只保留
  `os.exitpoint()` 和低频等待。

纯轮廓画布用 `last_rendered_state`、倒计时桶和结果键判断 `ui_dirty`。只有倒计时跨越
1 秒、完成集合、下一块、观测结果或错误改变时才清空并重绘；相机预览仍按配置频率更新。
计数器输出 `render_count`、`display_count`、`skipped_render_count`。

### 3.6 扫描线覆盖率

计划冻结后，所有目标多边形一次性转换成 `{y: [(x0,x1), ...]}`。复检只遍历缓存区间，
不再对每个采样点调用 `point_in_polygon()`。日志记录 `coverage_scan_ms`、
`sample_count`、`foreground_count` 和覆盖率。

### 3.7 扑克牌接口

`non_background_rgb` 按像素与深色背景的 RGB 距离生成掩膜，因此白底、红色和黑色图案
都可属于碎片。`EdgeImageStrip` 在几何边内侧按毫米采样灰度、梯度和有效掩膜；
`compute_edge_strip_cost()` 自动反向对齐另一条边。该代价只能重排已经通过几何筛选的
候选，不能放行非法边。当前默认 `ENABLE_IMAGE_STRIP_MATCHING=False`，实时规划结果
仍为纯几何结果。

## 4. 配置变化

| 配置 | 默认值 | 单位/含义 |
| --- | ---: | --- |
| `CONTOUR_DP_TOLERANCE_MM` | 2.2 | mm，闭合轮廓简化 |
| `LINE_FIT_MAX_ERROR_MM` | 2.5 | mm，直线拟合最大残差 |
| `MIN_EDGE_LENGTH_MM` | 18.0 | mm，赛题边描述下限 |
| `BOUNDARY_TRACE_MAX_STEP_FACTOR` | 8 | 周长估计倍数，循环保护 |
| `BOUNDARY_TRACE_MIN_POINTS` | 12 | 点，正常跟踪下限 |
| `ENABLE_BOUNDARY_FLOOD_FALLBACK` | `True` | 显式旧洪泛回退 |
| `FORCE_CONVEX_CONTOURS` | `False` | 是否强制凸轮廓 |
| `SEAM_LENGTH_ABS_TOLERANCE_MM` | 4.0 | mm，接缝绝对长度差 |
| `SEAM_LENGTH_REL_TOLERANCE` | 0.05 | 比例，接缝相对长度差 |
| `SEAM_ENDPOINT_ANGLE_TOLERANCE_DEG` | 35.0 | °，端点角软代价尺度 |
| `FINAL_VERTEX_TOLERANCE_MM` | 20.0 | mm，最终顶点验收 |
| `FINAL_CENTER_TOLERANCE_MM` | 15.0 | mm，最终中心验收 |
| `FINAL_ANGLE_TOLERANCE_DEG` | 12.0 | °，最终角度接口 |
| `GEOMETRY_EPSILON_MM` | 0.05 | mm，几何数值误差 |
| `OVERLAP_AREA_TOLERANCE_MM2` | 8.0 | mm²，严格重叠上限 |
| `MAX_RECTANGLE_HYPOTHESES` | 12 | 个，实际展开矩形假设 |
| `MAX_DFS_NODES` | 1200 | 个，统一搜索节点上限 |
| `MAX_PLAN_TIME_MS` | 3000 | ms，规划墙钟上限 |
| `STATE_POSITION_QUANTIZATION_MM` | 0.5 | mm，状态去重 |
| `STATE_ANGLE_QUANTIZATION_DEG` | 0.5 | °，状态去重 |
| `A4_DETECT_INTERVAL_ACQUIRE` | 2 | 帧 |
| `A4_DETECT_INTERVAL_PLACING` | 8 | 帧 |
| `A4_HOLD_MISSED_FRAMES` | 15 | 次 A4 检测，短时失检保持 |
| `PIECE_COUNT_WINDOW_DETECTIONS` | 12 | 次，未知数量观察窗口 |
| `PIECE_COUNT_SETTLE_DETECTIONS` | 8 | 次，规划前有效数量样本 |
| `PIECE_COUNT_MIN_CONFIRMATIONS` | 2 | 次，候选数量最低确认数 |
| `PIECE_LOW_GRAY_THRESHOLD` | 165 | 灰度，漏检探测重试 |
| `PIECE_THRESHOLD_PROBE_EVERY_N_DETECTIONS` | 4 | 次，建共识期间探测频率 |
| `PIECE_DIAGNOSTIC_PRINT_EVERY_N_DETECTIONS` | 5 | 次，稳定日志间隔 |
| `PLACING_VERIFICATION_INTERVAL_MS` | 5000 | ms |
| `UI_COUNTDOWN_REFRESH_INTERVAL_MS` | 1000 | ms |
| `ENABLE_STAGE_TIMING` | `False` | 板端详细计时默认关闭 |
| `TIMING_REPORT_INTERVAL_FRAMES` | 30 | 帧 |
| `ENABLE_IMAGE_STRIP_MATCHING` | `False` | 可选图案评分 |
| `IMAGE_STRIP_WIDTH_MM` | 3.0 | mm |
| `IMAGE_STRIP_SAMPLE_SPACING_MM` | 1.0 | mm |
| `IMAGE_STRIP_WEIGHT` | 0.15 | 候选排序权重 |
| `BACKGROUND_SEGMENTATION_MODE` | `"white"` | 兼容现有白片 |
| `BACKGROUND_COLOR_RGB` | `(30,70,100)` | RGB，非背景模式标定值 |
| `BACKGROUND_COLOR_DISTANCE_THRESHOLD` | 55.0 | RGB 欧氏距离 |

## 5. 测试结果

- 原有测试：25/25。
- 新增测试：40/40。
- 合计：65/65，失败 0。
- 离线样例：识别 4 块，顶点数 3/4/4/4；固定 100×60 mm 方案有效；未知尺寸
  `corner_outer_strict` 方案有效。
- 优化后的固定方案相对基线整体旋转 180°。矩形具有该对称性；按目标矩形中心做
  全局 180° 归一化后，各块目标中心误差为 0～0.013 mm，旋转误差为
  0～0.031°，因此几何解等价。
- 单文件构建成功；两个生成文件均通过 `py_compile`。
- 单文件静态检查未发现 `cv2`、`math.hypot`、`.draw_string(` 或本地
  `puzzle_*` 导入。

测试命令：

```text
python3 -m unittest test_puzzle_geometry.py test_puzzle_vision.py \
  test_puzzle_placement.py test_puzzle_perf.py \
  test_puzzle_boundary_trace.py test_puzzle_candidate_graph.py \
  test_puzzle_concave_geometry.py test_puzzle_scanline_coverage.py \
  test_puzzle_image_strip.py test_puzzle_performance_regression.py \
  k230_realtime_a4/test_realtime_a4.py
```

## 6. 性能对比

以下均为桌面 CPython 中位数或确定性调用量，不是 K230 实测。

| 指标 | 优化前 | 优化后 | 变化 |
| --- | ---: | ---: | ---: |
| CanMV 模拟组件轮廓提取 | 35.883 ms | 2.109 ms | -94.1% |
| 实时未知矩形 `plan_ms` | 152.494 ms | 44.543 ms | -70.8% |
| 原始跨碎片边对 | 84 | 84 | 不变，只构建一次 |
| 严格候选 | 5 | 4 | -20.0% |
| 宽松候选 | 47 | 8 | -83.0% |
| 实际展开矩形假设 | 478 | 12 | -97.5% |
| DFS 节点 | 2358 | 1120 | -52.5% |
| 精确多边形相交 | 14486 | 2610 | -82.0% |
| AABB 提前排除 | 0 | 636 | 新增 |
| 301 帧静态 PLACING 模拟 render | 151 | 5 | -96.7% |
| 扫描线目标覆盖率 | 1.725 ms | 0.121 ms | -93.0% |

补充数据：

- 固定尺寸离线规划：2200.151 → 2096.640 ms；节点 5308 → 2730；
  精确相交 255762 → 140566。
- 完整离线 CLI 墙钟：2.39 → 2.24 秒。
- 桌面完整检测中位数：1.632 → 3.316 ms。这里出现回退，原因是桌面路径现在保留并
  验证真实有序/凹轮廓，而旧路径直接取凸包；当前绝对耗时仍约 3.3 ms。板端关键的
  Python 洪泛替换则在同一模拟组件上显著下降。
- 平均板端帧时间、真实 render 驱动耗时和 FPS：未测，不能从桌面数据外推。

原始数据见 `performance_baseline.json` 和 `performance_optimized.json`。

## 7. 板端运行与日志

CanMV IDE 直接打开并运行：

```text
k230_realtime_a4/k230_realtime_a4_standalone.py
```

需要测量时临时把 `ENABLE_STAGE_TIMING=True` 后重建单文件。终端将输出：

```text
[PERF] frame=... capture=... a4_detect=... rectify=... blob=...
contour=... polygon_fit=... plan=... coverage_scan=... render=...
display=... total_frame=...
[PERF_COUNT] ... render_count=... display_count=...
skipped_render_count=...
PLAN_PERF,frame=...,time_ms=...,nodes=...,edge_pairs=...,filtered=...,
intersections=...,aabb_rejects=...,rect_hypotheses=...
```

## 8. 未完成项与风险

1. 尚未在真实 K230 上测量阶段耗时、FPS、停止延迟和峰值内存；285 KB 左右的实时
   单文件解析/内存开销仍需上板确认。
2. 扑克牌部分完成了 RGB 非背景掩膜、边带采样和评分接口，但当前实时入口仍使用灰度
   白色 `find_blobs`，没有接入现场蓝/绿背景 RGB 标定，也没有在真实扑克牌上验证。
3. 图案评分默认关闭，运行时还未从稳定帧为每条边生成条带并传给候选图；启用前需要
   现场曝光、背景色和条带宽度标定。
4. 当前放置覆盖率仍以白色为前景。扑克牌模式应改用非背景掩膜后再复用同一扫描线缓存。
5. ear clipping 支持简单凹多边形，不支持带孔多边形；检测到自交或剖分失败会
   fail-closed。近共线、极窄凹槽仍是需要真实铁片样本覆盖的边缘案例。
6. 详细计时默认关闭；开启后日志和计时本身会带来少量板端开销。

## 9. 现场日志后续修正

根据首次板端日志中 `4→2→0→3→4` 的碎片数量波动，实时入口增加了未知数量共识：
不会硬编码四片，但会优先等待观察窗口中重复出现的较高数量，避免偶发漏掉一片时拿
2/3 片子集提前规划。共识尚未建立、当前数量低于预期或检测少于两片时，会用新灰度图
以阈值 165 重试。`PIECE_DETECT` 现在输出原始 blob、接受数量、各类拒绝、边界失败、
顶点数和面积；`PLAN_PENDING` 输出明确 `reason`、预期数量、有效样本和稳定碎片数。

A4 已锁定状态的保持从 4 次检测失误延长到 15 次。保持期间继续复用最后可靠四角和
碎片跟踪历史；长期遮挡后仍会 fail-closed 返回 SEARCH。

## 10. 固定规划实机热区与实时路由

2026-07-30 的 K230 `PLAN_DEBUG` 日志确认，固定 100×60 mm beam 不是挂死，而是
在重复执行精确多边形相交：

- `fixed_rank depth=1` 从约 4 秒持续到 20 秒，相交计数
  `6888 → 34019`；
- `fixed_expand depth=2` 持续到约 50 秒，相交计数升至 `93715`；
- 第二次 `fixed_rank` 到 54 秒仍未完成，随后收到 IDE interrupt。

固定状态现在增量缓存 `outside/overlap/boundary/partial_rank`，并在一个等价量化
状态已经产生合法结果后，先按位姿键跳过重复候选，再执行精确重叠。这保持原评分公式、
beam 宽度、候选集合和最终门限不变。

更关键的是，实时 A4 配置现在设置
`PREFER_OUTER_FIRST_PLANNER=True`。固定目标尺寸仍保留给桌面验证和手工回退，但板端
主路径使用一次性候选图、外边约束、缓存变换和明确的节点/墙钟上限。

同一桌面照片、相同 `PieceObservation` 的最新规划中位数：

| 路径 | 中位数 | 精确相交 | 节点 | 结果 |
| --- | ---: | ---: | ---: | --- |
| 固定 beam（增量缓存后） | 2072.48 ms | 126500 | 2730 | 有效 |
| `outer_first` 实时主路径 | 34.76 ms | 1798 | 1120 | 有效 |

两者分数均为 `0.031757453540`，目标拓扑和误差指标等价；桌面速度比约
`59.6×`。这仍不是 K230 实测速度，但实机热区计数与桌面调用量一致，下一次板端日志
应从 `PLANNING_START,planner=outer_first` 开始，并在 3000 ms 墙钟限制内完成或明确
失败，不再进入分钟级固定 beam。

## 11. 宽松外框误放行修正

2026-07-30 的后续实机日志暴露了另一个独立问题：快速路径返回
`mode=outer_first_tolerant`，但结果为 `105.3×80.0 mm`，缺口
`2197.3 mm²`、`score=0.2609`。旧代码把竞赛的 20 mm 对应点容差同时用于最终外边
检查，并允许宽松结果最多留下矩形面积 30% 的空缺，因此开放扇形也能通过；这不是
绘图问题，而是最终验收过宽。

修正后，实时入口仍使用快速候选图，但显式把 `TARGET_RECT_SIZE_MM=(100,60)` 传给
规划器：

- 角点混合搜索只展开 100×60 和 60×100 两种方向，不再从噪声轮廓猜目标尺寸；
- 宽松模式只影响接缝候选，最终外边回到 5 mm 容差；
- 最终结果必须同时满足尺寸误差 ≤5 mm、score ≤0.06、gap ≤220 mm²、
  overlap ≤30 mm²、outside ≤250 mm²；
- 任一项不满足即返回无效方案，不生成搬运目标。

同一桌面样例走与实时入口相同的已知尺寸快速路径，得到
`corner_outer_strict`、100×60 mm、score `0.03175745`、gap `85.64 mm²`，
桌面单次约 53 ms；日志中的错误数值已加入回归测试。

后续复现还发现：正确样例的所有轮廓统一放大仅 1% 后，精确 100×60 搜索就会返回
`no outside-edge rectangle assembly`。实机冻结四块总面积约 6170～6230 mm²，
相对完整目标 6000 mm² 对应约 1.4～1.9% 的统一长度偏差，正处于这个失败区间。
因此已知目标路径现在先计算
`sqrt(6000 / input_area)`，仅当修正量不超过 4% 时围绕各块中心统一缩放规划轮廓；
旋转和源中心不变。4% 以外不校正，最终严格 Gate 也不变，所以不会重新放行上一轮
开放扇形。`PLAN_PERF` 新增 `input_area_mm2`、`area_scale`，无效方案另输出四条
`PLAN_INPUT` 毫米顶点，便于直接复现。

桌面回归覆盖 1～4% 统一尺度偏差；模拟实机四块分别按最新面积放大后，输入总面积
6246 mm²、自动尺度 0.9801，仍得到合法 100×60 `corner_outer_strict`，score
`0.03356`、gap `99.68 mm²`。

## 12. 白色分界线辅助 A4 标定

现场灰度缩略图显示：`dark_blob` 得到的黑色 A4 四角经过 `rotation_corr` 后，白色
物理分界线仍然倾斜；同时 CanMV 原生检测路径把分界位置固定为 148.5 mm，桌面路径
已有的自动行检测并未接入板端。这会同时影响毫米轮廓、上下区 ROI 和背景采样。

板端 A4 候选现在把黑纸上的白色分界线作为第二个标定特征：

- 在名义位置 ±8 mm 内跨 25 个横向样本搜索显著亮于黑纸的像素；
- 要求白线覆盖至少 70% 宽度，并通过 2.5 px 直线残差 Gate，避免大碎片长边误报；
- 暗色回退改用连通域轮廓真实角点，不再使用会丢失透视梯形的旋转外接矩形；
- 在候选四角对应的投影 A4 坐标中估计 `divider_y_mm` 和全宽高度差
  `divider_slope_mm`，绝对值超过 3 mm 时拒绝候选；
- 不再移动外角去迎合白线；无可靠白线时拒绝初始 A4 锁定，不会用碎片短边或
  名义中线强行校准；
- 动态分界位置已接入上/下碎片 ROI、下半区背景标定、状态示意图以及运动忽略带。

`A4_LOCK` 新增位置、投影后斜率、置信度和冻结状态；`PIECE_DETECT` 也记录实际
使用的分界位置。新增投影白线、错误斜率拒绝和 CanMV 动态 ROI 回归。
