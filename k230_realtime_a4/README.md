# K230 轻微晃动实时 A4 拼图识别

这是与原固定四点版本隔离的试验目录。板端只需运行：

```text
k230_realtime_a4_standalone.py
```

不要在 `>>>` 后输入命令。CanMV IDE 不自动同步本地依赖，因此首轮实机测试应使用上述
单文件版。

## 实时处理顺序

1. 将 800×480 画面缩小为保持比例的 320×192 灰度图。
2. 用原生 `find_rects()` 搜索高对比矩形。
3. 同时以最大黑色连通域作为边缘模糊时的后备候选。
4. 同时支持画面中的纵向/横向 A4，并按 A4 宽高比、画面占比、中心位置和内部黑色
   亮度筛选候选；任何触碰画面边缘的四边形都不会锁定。
5. 根据白色碎片分布自动判断哪一半是物理 A4 上半区，不要求赛前手填
   `top/bottom/left/right`；画面旋转 90° 或 180° 仍可自动定向。
6. 连续 3 帧有效后锁定，使用自适应平滑跟踪四角：
   大幅运动快速跟随，小幅边缘抖动加强平滑。
7. 按阶段配置的频率用最新四角调用原生 `rotation_corr()` 展开为标准 A4。
8. 透视校正后从当前基本为空的下半区稀疏采样，以中位灰度建立纸面背景模型；
   白片阈值取 `背景 + 30`，低对比重试取 `背景 + 20`，并根据背景亮度起伏自动增加
   噪声余量。整体曝光变化时阈值随之移动；细黑分界线低于绿色背景，不进入白片
   前景。背景样本不足时才回退到固定 180/165。
9. ACQUIRE 只在上半区识别白色碎片，并在 A4 毫米坐标内跟踪、稳定和规划。
   规划前使用 12 次检测的未知数量共识，不硬编码四片；偶发少识别一片时不会拿
   2/3 片子集提前规划。跨帧关联允许同一块的拟合轮廓相差最多 2 个顶点，但同时
   要求中心距离不超过 15 mm、面积变化不超过 35%，避免串块。达到稳定后按窗口
   内多数顶点数选择面积居中的代表轮廓交给规划器，而不是使用最后一帧的偶然
   拟合。共识建立期间每 4 次检测执行一次低对比重试。多边形
   收尾会把相距不足 7 mm 的相邻假顶点拟合成一个角，并删除夹角接近 180°、偏离
   邻点连线不超过 3 mm 的假顶点；每次修改同时受面积和简单多边形校验保护。
10. 2～4块稳定后，按赛题“每片至少有一条目标外边”的条件执行外框优先规划：
   优先尝试没有等长配对的边和邻近直角的边，将候选外边对齐矩形坐标轴，再递归
   匹配内部拼缝。
11. 若检测到明确的近似90°角块，先由角点相邻边组合和全部碎片总面积推导未知
    矩形尺寸，再把无角块通过拼缝加入；角块不明确时直接走外边—拼缝路径。
12. 搜索过程中要求每个已放碎片始终保留一条未用于拼缝的水平或竖直候选外边，
    同时按 90～120 mm 长边、50～90 mm 短边、重叠和面积缝隙逐层剪枝。目标尺寸
    和碎片形状、数量均不硬编码。
13. 有效方案生成后立即冻结，不因机械结构拿起碎片而失效；程序进入 `PLACING`
    阶段，并在同一张纯轮廓 A4 简图上显示实际碎片位置与下半区目标轮廓。
    轮廓画面右下角同时显示透视校正后的 `240×336` 灰度工作图，标题中的 `T` 是
    本次分割阈值、`F` 是工作图来源帧，方便直接判断漏检来自曝光还是轮廓拟合。
14. 每5秒重新识别整张 A4，通过形状而不是当前位置重新关联剩余碎片。对应顶点、
    中心和目标轮廓均满足阈值后输出 `PIECE_PLACED`，并永久停止跟踪该碎片。
15. 已完成碎片与下一块接触而合并为一个白色连通域时，使用预计算扫描线统计目标
    多边形内部白色覆盖率；全部完成后进入 `COMPLETE` 并停止摄像头视觉分析。

相机的轻微平移、转动和透视变化先由 A4 四角吸收，因此碎片稳定判断不再直接承受
原始画面晃动。

## 实时性能与停止响应

锁定后各阶段采用不同频率：

```python
A4_DETECT_INTERVAL_ACQUIRE = 2
A4_DETECT_INTERVAL_PLACING = 8
PIECE_DETECT_EVERY_N_FRAMES = 3
REALTIME_PIECE_WORK_WIDTH = 240
REALTIME_PIECE_WORK_HEIGHT = 336
PLACING_VERIFICATION_INTERVAL_MS = 5000
UI_COUNTDOWN_REFRESH_INTERVAL_MS = 1000
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
```

A4 在 ACQUIRE 每2帧、PLACING 每8帧更新一次，碎片多边形在 ACQUIRE 每3帧更新；
其余帧复用最近一次有效识别，
但画面、FPS 和停止检查继续运行。相比每帧处理320×448碎片图，像素遍历面积降低约
44%，同时避免在已有有效矩形时重复执行黑色连通域后备检测。规划冻结后不再每帧
识别碎片，只在5秒检查点扫描上、下半区，因此机械搬运期间显示刷新不受完整轮廓
提取持续拖慢。纯轮廓界面仅在倒计时、完成集合或观测结果变化时重画；
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
PLANNING_DONE,frame=...,elapsed_ms=...,valid=1,mode=...,nodes=...
PLAN,frame=...,stable=1,count=...,mode=...,target_w_mm=...,target_h_mm=...
PIECE,id=...,sx_mm=...,sy_mm=...,tx_mm=...,ty_mm=...,rot_deg=...
PLAN_END
PLACEMENT_START,frame=...,count=...,check_interval_ms=5000,next=P1
PLACEMENT_CHECK,frame=...,check=1,observed=...,matched=...,completed=.../...
PIECE_PLACED,frame=...,id=P1,method=vertices,...
PLACEMENT_COMPLETE,frame=...,elapsed_ms=...,count=...
```

`PLANNING_DONE` 必须很快出现；随后状态画布右侧显示目标长方形和四块的正确位置。
可能的模式为 `corner_outer_strict`、`outer_first_strict` 及其 `tolerant` 容差版。
外边—拼缝和角块—外边路径统一受1200节点和3000 ms墙钟上限保护；
超过上限或没有合法矩形时输出 `valid=0` 和 `PLAN_INVALID`，不会无限搜索。

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
摄像头画面；即使 A4 短暂丢失也保留冻结方案和最近轮廓，重新锁定后继续5秒复检。
搬运阶段还会保持 A4 四角的物理标签，避免碎片移到下半区后自动方向判断翻转。

纯轮廓画面中：

- 彩色实线：最近一次复检得到的未完成碎片实际位置；
- 黄色目标线：当前建议搬运的下一块；
- 灰色目标线：后续待搬碎片；
- 绿色目标线及 `OK`：已确认到位且已取消跟踪的碎片；
- 右侧显示完成数量、下一块、下次检查倒计时和各块目标中心/旋转角。

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
PLACEMENT_CHECK,...
PIECE_PLACED,...
PLACEMENT_COMPLETE,...
```

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

桌面端共 76 项测试通过，其中包括 A4 角点排序、横向/纵向暗色候选筛选、触边假框
否决、连续帧锁定、稀疏候选禁止累积锁定和轻微抖动跟踪、
原有轮廓回归、通用几何规划、未知尺寸的2/3/4片矩形、四片均无90°顶点的矩形，
带小幅顶点测量误差的容差规划、整张A4上下区域检测、乱序形状关联、逐片完成退休
和连通域合并后的扫描线覆盖率后备确认，以及候选图、AABB、凹多边形、UI 状态和
扑克牌边带接口；新增部分覆盖近点倒角拟合、浅共线点删除、顺逆点序一致性、
真实短边/凹角保护和反光咬口。桌面性能数据见
`../PERFORMANCE_OPTIMIZATION_REPORT.md`；真实
K230 上的外框阈值、规划耗时、FPS、内存和手持抖动余量仍需根据板端日志调整。
