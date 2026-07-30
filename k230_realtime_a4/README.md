# K230 固定四点手动 A4 拼图识别

当前实时入口已回退为固定四点手动标定。板端只需运行：

```text
k230_realtime_a4_standalone.py
```

新增的 `puzzle-vision-simulator` 兼容规划入口是：

```text
k230_realtime_a4_simulator_standalone.py
```

它和原实时入口共享 A4 标定、视觉识别、稳定冻结、搬运提示与最终验收，只替换冻结
后的拼接规划后端。该单文件不依赖 NumPy/OpenCV；生成时已经把
`puzzle_simulator_planner.py` 展开到脚本中，适用于 CanMV MicroPython。原单文件
仍保留 `outer_first`，便于同一现场输入直接 A/B。

不要在 `>>>` 后输入命令。CanMV IDE 不自动同步本地依赖，因此首轮实机测试应使用上述
单文件版。

两个单文件的生成命令分别是：

```bash
python3 k230_realtime_a4/build_standalone.py
python3 k230_realtime_a4/build_standalone.py --planner simulator
```

模拟器后端默认 `SIMULATOR_PLANNER_VALIDATION="local"`：先按上游语义产生拼法，再
通过本项目的 gap、overlap、outside 和尺寸 Gate，失败时不会进入机械放置阶段。
`"upstream"` 只应用于仿真对照；它会保留 `local_gate_failures` 警告，但仍把上游
提案标成有效，不应直接驱动硬件。可用
`--simulator-validation upstream --output /tmp/simulator_proposal.py` 生成临时对照
脚本，不要把该模式作为板端正式入口。

## 手动标定

实时程序使用 `realtime_a4_config.py` 中的 `A4_CORNERS_PX`，顺序固定为物理
A4 的 `TL, TR, BR, BL`。当前 A4 在相机里横放、碎片区位于画面左侧，所以四点在
画面中的对应顺序是 `左下、左上、右上、右下`：

```python
AUTO_CALIBRATE_A4 = False
A4_CORNERS_PX = [
    (133.0, 441.0),
    (140.0, 78.0),
    (619.0, 83.0),
    (614.0, 449.0),
]
```

这组值来自此前稳定锁定日志。启动后会短暂显示固定绿框；如纸张或相机位置发生
变化，只修改这四个坐标并重新生成单文件版。手动模式不会搜索 A4、不会检测白色
分界线、不会平滑四角，也不会在运行中更新透视矩阵；上下区域使用固定的
`DIVIDER_Y_MM = 148.5`。

## 实时处理顺序

1. 启动时直接载入固定的 `A4_CORNERS_PX`，不执行自动 A4 或白线检测。
2. 始终使用这四点调用原生 `rotation_corr()` 展开为标准 A4。
3. 透视校正后从当前基本为空的下半区稀疏采样，以中位灰度建立纸面背景模型；
   白片阈值取 `背景 + 30`，低对比重试取 `背景 + 20`，并根据背景亮度起伏自动增加
   噪声余量。整体曝光变化时阈值随之移动。上下区边界固定为 148.5 mm，在线两侧
   保留 2 mm 排除带，因此白线不会被当成碎片。背景样本不足时回退固定 180/165。
4. ACQUIRE 只在上半区识别白色碎片，并在 A4 毫米坐标内跟踪、稳定和规划。
   规划前使用 12 次检测的未知数量共识，不硬编码四片；偶发少识别一片时不会拿
   2/3 片子集提前规划。跨帧关联允许同一块的拟合轮廓相差最多 2 个顶点，但同时
   要求中心距离不超过 15 mm、面积变化不超过 35%，避免串块。达到稳定后按窗口
   内多数顶点数选择面积居中的代表轮廓交给规划器，而不是使用最后一帧的偶然
   拟合。共识建立期间每 4 次检测执行一次低对比重试。多边形
   收尾会把相距不足 7 mm 的相邻假顶点拟合成一个角，并删除夹角接近 180°、偏离
   邻点连线不超过 3 mm 的假顶点；每次修改同时受面积和简单多边形校验保护。
5. 2～4 块稳定后，常规单文件优先调用候选图 `outer_first` 规划器，但把已知
    100×60 mm 原型尺寸作为硬约束：角点路径只搜索 100×60/60×100 两种方向，
    最终结果还必须通过固定尺寸、score、gap、overlap、outside 五项 Gate。宽松模式
    只放宽接缝测量，不能放宽最终矩形。这样不运行板端耗时很高的固定尺寸 beam，
    也不会把开放扇形当成目标。若需桌面对照，可把
    `PREFER_OUTER_FIRST_PLANNER=False` 恢复固定规划；相同冻结输入不会反复搜索。
    由于完整拼图面积已知为 6000 mm²，搜索前会用碎片总面积校正不超过 4% 的统一
    轮廓尺度偏差；超过范围则拒绝校正，防止缺块或额外 blob 被掩盖。
    模拟器单文件则完整经过兼容后端的全边候选、T 形部分边候选、连通匹配集、
    刚体传播、闭环 pose-graph 优化和全局矩形归一化，最后复用同一安全 Gate。
6. 有效方案生成后冻结参考轮廓、ID、目标、操作顺序与总面积；A4 四角始终使用
    手动配置。程序进入
    `WAIT_FOR_MOTION`
    阶段，并在同一张纯轮廓 A4 简图上显示实际碎片位置与下半区目标轮廓。
    轮廓画面右下角同时显示透视校正后的 `240×336` 灰度工作图，标题中的 `T` 是
    本次分割阈值、`F` 是工作图来源帧，方便直接判断漏检来自曝光还是轮廓拟合。
7. 等待阶段只在 `80×112` 透视灰度图上做稀疏运动差分。手或电磁铁触发
    `MOVING` 后，A4、碎片识别、tracker、coverage 和完成判定全部暂停；连续稳定后
    才快速采集 3 个确认样本，随后回到等待运动。
8. 单块可分离时，以 32 点双向轮廓边界距离的 RMS/P90/P95 为主判据，不要求顶点数
    相同；连通域合并时，使用运动前后新增目标 coverage、面积比与 spill。两条路径
    任一可靠通过即可，第一块同样支持增量 coverage。
9. 全部逐块完成后，以同一 3 帧 burst 的 fill、冻结总面积比、bbox 和 20 mm 包络
    spill 做 2/3 最终 Gate；只有 `FINAL_ACCEPTED` 后才进入 `COMPLETE`。

本模式假定 A4 和水平安装的 K230 均保持静止；启动后若移动纸张或相机，需要重启
程序重新获取 A4 四角。

## 实时性能与停止响应

锁定后各阶段采用不同频率：

```python
A4_DETECT_INTERVAL_ACQUIRE = 2
PIECE_DETECT_EVERY_N_FRAMES = 3
REALTIME_PIECE_WORK_WIDTH = 240
REALTIME_PIECE_WORK_HEIGHT = 336
MOTION_SAMPLE_WIDTH = 80
MOTION_SAMPLE_HEIGHT = 112
MOTION_START_CONFIRM_FRAMES = 2
MOTION_END_CONFIRM_FRAMES = 4
POST_MOTION_STABLE_FRAMES = 4
POST_MOTION_VERIFY_SAMPLES = 3
PLACING_VERIFICATION_INTERVAL_MS = 30000
ENABLE_PLACEMENT_WATCHDOG = False
PIECE_COUNT_WINDOW_DETECTIONS = 12
PIECE_COUNT_SETTLE_DETECTIONS = 8
PIECE_SEGMENTATION_MODE = "background_delta"
PIECE_BACKGROUND_DELTA_GRAY = 30
PIECE_BACKGROUND_RELAXED_DELTA_GRAY = 20
PIECE_LOW_GRAY_THRESHOLD = 165
REQUIRED_STABLE_FRAMES = 4
CENTER_STABLE_TOLERANCE_MM = 4.0
ANGLE_STABLE_TOLERANCE_DEG = 8.0
TRACK_MAX_VERTEX_COUNT_DELTA = 2
TRACK_VERTEX_MISMATCH_MAX_DISTANCE_MM = 15.0
TRACK_VERTEX_MISMATCH_MAX_AREA_RATIO = 0.35
CONTOUR_DP_TOLERANCE_MM = 3.0
VERTEX_MERGE_DISTANCE_MM = 7.0
VERTEX_COLLINEAR_ANGLE_TOLERANCE_DEG = 18.0
VERTEX_COLLINEAR_MAX_OFFSET_MM = 4.0
SHOW_GRAY_WORK_THUMBNAIL = True
GRAY_THUMBNAIL_MAX_WIDTH = 128
GRAY_THUMBNAIL_MAX_HEIGHT = 180
IDE_STREAM_ENABLED = True
IDE_STREAM_EVERY_N_OUTPUTS = 2
IDE_STREAM_QUALITY = 50
AUTO_CALIBRATE_A4 = False
ENABLE_DYNAMIC_DIVIDER = False
```

A4 检测在手动模式下完全不运行；碎片多边形每3帧更新，其余帧复用最近一次有效识别，
但画面、FPS 和停止检查继续运行。相比每帧处理320×448碎片图，像素遍历面积降低约
44%，同时避免在已有有效矩形时重复执行黑色连通域后备检测。规划冻结后不再持续
识别碎片；运动期间只有稀疏差分，稳定后才执行 3 次有限确认。纯轮廓界面仅在状态、
完成集合或观测结果变化时重画；
COMPLETE 最终画面提交后不再抓帧、检测 A4 或计算覆盖率。

显示仍使用 `Display.init(..., to_ide=True)`，并新增对最终合成画面的显式
`compress_for_ide(quality=50)` 传输。显式传输每两次屏幕提交执行一次，规划中和
完成画面强制发送；若固件不支持或压缩失败，终端输出
`IDE_STREAM_ERROR,frame=...,reason=...`，但 LCD 和识别流程继续运行。VS Code 中需从
CanMV Toolbox 打开 `Preview`，或运行 `CanMV: 启用预览`。

轮廓洪泛遍历和几何搜索内部均定期调用 `os.exitpoint()`；内部检测捕获异常时会把
`IDE interrupt` 重新抛给外层，最终输出正常的：

```text
STOP,reason=ide_interrupt,frame=...
```

本版本在稳定识别到2～4块后会先后输出：

```text
PLANNING_START,frame=...,planner=outer_first,count=...
PLAN_DEBUG,planner=outer_first,stage=corner_expand_strict,elapsed_ms=...,
pieces=4,depth=2,states=...,expanded=...,nodes=...,work=...,
best_score=...,intersections=...,aabb_rejects=...
PLAN_PERF,...,input_area_mm2=...,area_scale=...
PLANNING_DONE,frame=...,elapsed_ms=...,valid=1,mode=...,nodes=...
PLAN_FAIL_DETAIL,frame=...,class=target_geometry,complete=...,
max_depth=...,seam_pairs=...,input_area_mm2=...,target_area_mm2=...,
area_error_pct=...,area_scale=...,closest_size=...x...,
size_error_mm=...,closest_gap_mm2=...,corner_reason=...
PLAN,frame=...,stable=1,count=...,mode=...,target_w_mm=...,target_h_mm=...
PIECE,id=...,sx_mm=...,sy_mm=...,tx_mm=...,ty_mm=...,rot_deg=...
PLAN_END
PLACEMENT_START,frame=...,count=...,trigger=motion,next=P1,verify_samples=3
MOTION_START,...
MOTION_END,...
POST_MOTION_STABLE,...
VERIFY_START,...
VERIFY_SAMPLE,...
VERIFY_RESULT,...
PIECE_ACCEPTED,...
PLACEMENT_CHECK,frame=...,check=1,observed=...,matched=...,completed=.../...
FINAL_SCENE_METRICS,...
FINAL_ACCEPTED,...
PLACEMENT_COMPLETE,frame=...,elapsed_ms=...,count=...
```

模拟器单文件还输出：

```text
PLANNING_START,frame=...,planner=simulator,count=...
SIMULATOR_PLAN_PERF,frame=...,cut_mode=auto,validation=local,
candidates=...,full=...,partial=...,sets=...,selected=...,
selected_partial=...,actual_size=...x...,local_gate_failures=...
```

其 `PLAN_FAIL_DETAIL.class` 分为 `search_limit`、`no_edge_candidates`、
`no_connected_topology` 和 `local_geometry_gate`。最后一种表示上游兼容算法已经
得到完整拼法，但该拼法的实际矩形、重叠、外溢或缺口未通过本地执行门限。

实时配置默认每 2000 ms 最多输出一条 `PLAN_DEBUG`。时钟只在已有的搜索批次/
`exitpoint` 边界读取，不在每次多边形相交时读取。常见 `stage`：

- `fixed_boundary_anchors`：生成100×60 mm边界锚点；
- `fixed_expand`：按深度扩展碎片放置状态；
- `fixed_rank`：计算候选状态精确相交代价并做 beam 排序；
- `fixed_evaluate`：对完整四片状态做最终评分；
- `strict_dfs` / `tolerant_dfs`：固定路径失败后的通用接缝搜索；
- `candidate_graph_*` / `corner_*` / `outer_*`：候选图外边优先路径；实时已知
  100×60 mm 时，`corner_*` 只展开两个固定方向。

若在 IDE 中止前最后一条始终是同一个阶段，对比相邻两条的 `states`、`expanded`、
`nodes` 和 `intersections` 即可判断是状态爆炸、精确相交过慢，还是搜索没有产生新
候选。开关和间隔为 `ENABLE_PLAN_DEBUG`、`PLAN_DEBUG_INTERVAL_MS`。

只有最终 Gate 通过才会出现 `PLANNING_DONE,...valid=1`，随后状态画布右侧显示
目标长方形和四块的正确位置。已知 100×60 mm 时，日志中的 `target_w_mm` /
`target_h_mm` 必须是 100/60（顺序可互换）；例如
`score=0.2609,gap_mm2=2197.3,target=105.3x80.0` 会 fail-closed，不再进入放置。
若无解，`PLAN_INVALID` 后会一次性输出四条 `PLAN_INPUT`，包含每块冻结多边形的毫米
顶点；这几行可以直接用于桌面复现，不会在搜索循环中增加持续日志开销。
同时输出一条 `PLAN_FAIL_DETAIL`，其中 `class` 用于区分主因：

- `search_limit`：尚未完成判定就触及节点或时间上限；
- `seam_connectivity`：候选拼缝图无法把全部碎片连成一个完整候选；
- `target_geometry`：已经产生完整候选，但其长宽偏离已知目标；`closest_size`、
  `size_error_mm` 和 `closest_gap_mm2` 给出最接近的一次；
- `rectangle_gate`：完整候选尺寸可接受，但矩形外边、缺口、重叠或 score Gate
  仍未通过。

`corner_reason`、`corner_depth` 和 `corner_complete` 记录角点优先分支退出的位置，
避免角点分支先触及搜索上限、随后被外边分支的笼统失败原因覆盖。
实时主路径模式为 `corner_outer_*`、`outer_first_*`；手工关闭优先开关后才进入
`fixed_*`。
外边—拼缝和角块—外边路径受1200节点和3000 ms墙钟上限保护。当前固定尺寸 beam
路径尚未使用该墙钟上限；板端明显慢于桌面时会持续输出上述心跳，最终失败仍输出
`valid=0` 和 `PLAN_INVALID`。

近似90°顶点只用于提高搜索优先级，不是硬条件。矩形角可能恰好由两块碎片分别占据，
此时任何单块都没有90°顶点；算法仍会从未配对的候选外边启动。

若仍需更高帧率，可依次把 `PIECE_DETECT_EVERY_N_FRAMES` 调到4，再把
`A4_DETECT_INTERVAL_ACQUIRE` 调到3。手持晃动较明显时优先保留 ACQUIRE 的
A4 更新频率。

## 自动标定画面

默认不需要切换任何调试开关。程序启动、搜索和初次锁定 A4 时显示摄像头：

- 黄色四边形：当前帧检测到的最佳 A4 候选；
- 绿色四边形：已锁定并平滑后的 A4；
- 灰线：A4 中部分界位置；
- 彩色多边形和十字：展开后识别到的碎片轮廓及几何中心。

绿色框锁定后保留 20 帧供确认。方案生成后无条件切换到纯轮廓画布，不再显示彩色
摄像头画面；冻结四角不会因搬运遮挡或碎片移到下半区而更新或翻转。

纯轮廓画面中：

- 彩色实线：最近一次复检得到的未完成碎片实际位置；
- 黄色目标线：当前建议搬运的下一块；
- 灰色目标线：后续待搬碎片；
- 绿色目标线及 `OK`：已确认到位且已取消跟踪的碎片；
- 右侧显示完成数量、下一块、运动状态和各块目标中心/旋转角。

终端状态顺序：

```text
START_REALTIME_A4,...
A4_SEARCH,frame=...,rects=...,dark_blobs=...,candidates=...
A4_LOCK,frame=...,source=...,confidence=...,orientation=...,corners=...
PLAN_PENDING,frame=...,count=...,stable_pieces=...,reason=...,
expected=...,count_samples=.../...,a4=1,...
PIECE_DETECT,frame=...,region=upper,segmentation=background_delta,
threshold=...,bg=...,bg_high=...,bg_spread=...,delta=...,bg_samples=...,
raw_blobs=...,
accepted=...,rejected=...,raw_vertices=...,vertices=...,areas_mm2=...
PLANNING_START,...
PLANNING_DONE,...
PLAN,...mode=corner_outer_*|outer_first_*,target_w_mm=...,target_h_mm=...
PIECE,...
PLAN_END
PLACEMENT_START,...
MOTION_START,...
MOTION_END,...
POST_MOTION_STABLE,...
VERIFY_START,...
VERIFY_SAMPLE,...
VERIFY_RESULT,...
PIECE_ACCEPTED|PIECE_REJECTED,...
PLACEMENT_CHECK,...
FINAL_SCENE_METRICS,...
FINAL_ACCEPTED,...
PLACEMENT_COMPLETE,...
```

手动版 `A4_LOCK` 输出：

```text
source=manual,confidence=1.00,motion_px=0.0,orientation=manual,
divider_y_mm=148.5,divider_slope_mm=0.0,divider_confidence=0.00,
divider_detected=0,frozen=1,corners=...
```

其中 `corners` 必须始终等于配置的四点；任何后续帧都不会改变这些数值。

## 现场调整

必须让 A4 四个角都留在画面内。若一直 `A4_SEARCH`：

- `rects=0`：将 `A4_RECT_EDGE_THRESHOLD` 从 7000 降到 5000 或 3500；
- `rects>0,candidates=0`：查看新版日志中的 `rejected=` 统计；
- `rejected=touches_edge:...`：A4 有边或角被画面裁掉，必须拉远相机；
- `rejected=aspect:...`：原始矩形的宽高比不像 A4；
- `rejected=brightness:...`：候选内部不像黑色工作面；
- 黑底偏亮：将 `A4_DARK_THRESHOLD` 从 135 提高到 150；
- 白桌面被误选：降低 `A4_MAX_INSIDE_GRAY`。

若已有 `A4_LOCK` 但 `count=0`：

- 确认碎片在灰色分界线以上；
- 先看 `bg` 和 `delta`：正常情况下主检测 `delta` 约为 30，宽松重试约为 20；
- 若白片与纸面实际灰度差不足 30，将 `PIECE_BACKGROUND_DELTA_GRAY` 逐步降至
  25、20；若纸面纹理或反光被误检，则提高该值或
  `PIECE_BACKGROUND_NOISE_MARGIN_GRAY`；
- `segmentation=background_delta_fallback` 表示有效背景样本不足，此时才使用固定
  `WHITE_GRAY_THRESHOLD=180`；
- 推荐深色、哑光、无纹理绿色纸。下半区可以逐步放入目标碎片，但建立方案前应基本
  为空；少于一半的白片覆盖不会改变中位背景估计；
- 查看彩色轮廓是否完整，不要先改拼图规划参数。

`raw_vertices` 是当前帧直接拟合的顶点数，`vertices` 是跟踪稳定后交给规划器的
时间窗口代表结果。若前者波动而后者稳定，说明时间滤波正在正常吸收边缘毛刺；
若持续 `rejected=polygon`，再检查 `trace_failures=fit_invalid` 或
`piece_invalid`。非法单块现在只会被拒绝，不会再让整帧抛出
`piece polygon is not simple`。

## 验证范围

桌面端共 83 项测试通过，其中包括 A4 角点排序、横向/纵向暗色候选筛选、触边假框
否决、连续帧锁定、稀疏候选禁止累积锁定和轻微抖动跟踪、
原有轮廓回归、通用几何规划、未知尺寸的2/3/4片矩形、四片均无90°顶点的矩形，
带小幅顶点测量误差的容差规划、整张A4上下区域检测、乱序形状关联、逐片完成退休
和连通域合并后的扫描线覆盖率后备确认，以及候选图、AABB、凹多边形、UI 状态和
扑克牌边带接口；新增部分覆盖倾斜白色分界线辅助标定、实时分区位置、本次开放扇形
误放行日志的已知目标 Gate、已知面积尺度校正、近点倒角拟合、
浅共线点删除、顺逆点序一致性、
真实短边/凹角保护和反光咬口。桌面性能数据见
`../PERFORMANCE_OPTIMIZATION_REPORT.md`；真实
K230 上的外框阈值、规划耗时、FPS、内存和手持抖动余量仍需根据板端日志调整。
