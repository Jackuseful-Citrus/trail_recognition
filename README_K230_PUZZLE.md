# K230 A4 拼图识别与几何规划 V1

## 交付结构

- `puzzle_config.py`：唯一参数入口，包含自动 A4、毫米比例、阈值、搜索门限和稳定窗口。
- `puzzle_a4_boundary.py`：原生 A4 四边形搜索、候选评分、自动定向和连续帧锁定。
- `puzzle_geometry.py`：不依赖摄像头、OpenCV、NumPy 的纯 Python 几何、跟踪和矩形规划核心。
- `puzzle_perf.py`：兼容 CPython/CanMV 的可关闭阶段计时和调用计数器。
- `puzzle_realtime_state.py`：状态驱动显示与阶段调度的纯 Python 逻辑。
- `puzzle_image_strip.py`：默认关闭的非背景分割和接缝图像带接口。
- `puzzle_vision.py`：桌面 OpenCV 离线适配层，以及不依赖 `cv2` 的 CanMV 原生
  `image.Image` 适配层。
- `k230_puzzle_planner.py`：CanMV K230 v1.6 摄像头循环、稳定判定、状态画布和结构化输出。
- `k230_puzzle_planner_standalone.py`：供不自动同步依赖文件的 CanMV IDE 直接运行的
  单文件板端入口。
- `build_k230_standalone.py`：从共享源模块重新生成上述单文件入口。
- `k230_a4_auto_calibration_standalone.py`：只运行当前自动 A4 搜索、分隔线检测和
  多帧锁定的独立上板测试；使用方法见 `README_K230_A4_RECOGNITION_TEST.md`。
- `offline_validate_puzzle.py`：桌面照片验证入口。
- `test_puzzle_geometry.py`、`test_puzzle_vision.py`：纯几何与当前照片回归测试。
- `sample_puzzle.jpg`：本次提供的原始示例照片。

原有 `test_camera.py` 和 `tools/vision/fisheye_black_tape_tracker.py` 未修改。

## 板端部署

当前 v1.6 固件会提示 `REPL input is not supported`，CanMV IDE 运行编辑器缓冲区时也
不会自动上传本地导入模块。因此最简单的部署方式是直接在 IDE 打开并运行：

```text
k230_puzzle_planner_standalone.py
```

该文件内已经包含配置、几何、视觉和入口，不需要 REPL，也不需要
`puzzle_config.py` 等同目录依赖。

如果使用 IDE 的板端文件管理功能，也可以将入口及其共享模块真正写入板端文件系统的同一
目录：

```text
k230_puzzle_planner.py
puzzle_config.py
puzzle_a4_boundary.py
puzzle_geometry.py
puzzle_vision.py
```

再运行 `k230_puzzle_planner.py`。修改源模块后，可在电脑端执行
`python3 build_k230_standalone.py` 刷新单文件版。程序使用与已验证相机脚本相同的
传感器设置：

```python
sensor.set_hmirror(True)
sensor.set_vflip(True)
sensor.set_framesize(width=800, height=480)
sensor.set_pixformat(Sensor.RGB565)
Display.init(Display.ST7701, width=800, height=480, to_ide=True)
```

## A4 自动标定

板端默认 `AUTO_CALIBRATE_A4 = True`，不读取手填四点：

1. 在 320×192 灰度图中用原生 `find_rects()` 搜索 A4 外框；
2. 边缘矩形失败时，以最大黑色连通域的 `min_corners()` 作为后备；
3. 按 A4 横/竖比例、画面占比、中心位置、黑色内部亮度和触边情况评分；
4. 根据白色碎片分布自动判断物理上半区，兼容横放、竖放和 180° 旋转；
5. 连续三帧候选有效才锁定，然后使用锁定四点识别碎片。

启动或未锁定时画面自动显示黄色候选框；绿色锁定框保留 20 帧后自动进入结果画布，
不需要现场切换 `DEBUG_SHOW_CAMERA`。终端依次输出 `A4_SEARCH`、`A4_LOCK` 和
`PLAN_PENDING/PLAN`。A4 四角必须完整位于画面内；被相机裁掉的真实角点无法可靠
自动恢复，因此此时程序会保持搜索而不会输出错误方案。

## 白色阈值调整

桌面离线回归仍固定使用灰度阈值 180，以保证历史照片结果可重复。实时
`k230_realtime_a4` 板端则在 A4 透视校正后，从基本为空的下半区估计纸面中位灰度，
主阈值使用 `背景 + 30`，低对比重试使用 `背景 + 20`；全局光线变亮或变暗时阈值
同步移动。细黑分界线比绿色纸更暗，不会进入白片前景。

- 白片缺边且日志 `delta=30`：逐步降低 `PIECE_BACKGROUND_DELTA_GRAY`；
- 纸面反光被识别：提高该差值或 `PIECE_BACKGROUND_NOISE_MARGIN_GRAY`；
- 日志出现 `background_delta_fallback`：背景样本不足，当前帧回退到固定 180；
- 小亮点较多：提高 `MIN_PIECE_AREA_MM2`，或增加一次开运算；
- 白片内部有孔：增加 `MORPH_CLOSE_ITERATIONS`；
- 边角被过度磨平：降低 `POLYGON_APPROX_EPSILON`。

`THRESHOLD_MODE = "otsu"` 只影响桌面 OpenCV 离线路径；CanMV 板端为兼容当前
v1.6 固件，使用原生稀疏直方图和 `find_blobs()`，不依赖 Otsu 或 OpenCV。

## 轮廓、中心与姿态

桌面处理链为：A4 单应透视 → 灰度阈值 → 自动微调分界行 → 上半区屏蔽 →
开/闭运算 → 有序外轮廓 → Douglas-Peucker → 直线拟合 → 共线伪边合并。
凹角默认保留，仅对已经验证为凸形的轮廓做条件式凸稳定化。

实时 CanMV 板端先把图像缩放为 240×336，再用固件原生
`rotation_corr(corners=...)`
将 A4 四角展开至整个工作图；随后用 `find_blobs()` 得到白色连通域，通过
`to_numpy_ref()` 在每个连通域内做近似周长级 Moore 邻域边界跟踪，最后用有序轮廓
简化得到
3～5 个顶点。该路径不导入 `cv2`，适配实际
`CanMV_K230_V3P0_micropython_v1.6-...` 固件。

最终只接受 3～5 个主要顶点。对手工片，有明显转角的真实短边会保留；短边不再被
单独当作噪声删除，近共线毛刺才会合并。中心严格使用鞋带公式得到的面积加权多边形质心；
退化多边形会报错，不会改用外接矩形中心。当前姿态用于跟踪稳定性，取多边形最长边
轴；真正输出的旋转角来自“当前多边形到目标拼图片”的完整刚体变换。

## 矩形搜索

规划前一次性建立跨碎片边候选图并缓存双向刚体变换。规划器固定一块为根
（不会限制解，因为任意连通拼图都可从该块展开），再递归执行：

1. 从候选图索引读取已筛选的相容边；
2. 反向对齐边中点并检查端点误差；
3. 检查两片位于拼接边两侧；
4. 先做 AABB 排除，再以凸裁剪或凹多边形三角剖分计算真实重叠面积；
5. 标记已使用拼接边并递归；
6. 完整候选用最小面积外接矩形、面积缺口、重叠、尺寸和外边界评价。

当前手工片的已知目标由 `TARGET_RECT_SIZE_MM = (100.0, 60.0)` 给出。规划器先做
固定矩形边界引导的 beam search：枚举任意碎片/顶点作为四角锚点，并允许其他片通过
等长边附着。这样支持“一条长内边对应多块较短内边”的 T 型分割，不再错误地强迫
每片只做一对一整边拼接。固定矩形搜索把越界、重叠和未覆盖面积作为主损失。

若固定尺寸搜索失败，再使用约 3 mm 的严格整边回溯；严格解仍失败时，
`ENABLE_TOLERANT_FALLBACK` 才按赛题允许的 20 mm 对应顶点误差再次搜索。所有模式
仍把明显重叠作为硬约束，超过独立的分数、空隙或顶点误差门限就返回
`NO VALID PLAN`。现场未知尺寸碎片可把 `TARGET_RECT_SIZE_MM` 设为 `None`，
恢复 90～120 mm × 50～90 mm 的通用矩形搜索。矩形只展开评分最优的 12 个假设，
DFS 统一受 1200 节点和 3000 ms 墙钟上限保护。

## 多帧状态

跟踪使用顶点数、归一化边长/内角序列、紧致度、面积、位置和姿态，不按当前 x 坐标
编号。数量变化、中心移动超过 2 mm 或姿态变化超过 2° 会立即清空稳定状态和旧方案。
只有最近窗口满足稳定帧数后才运行规划并打印正式方案。

## 离线验证

```bash
python3 -m unittest
python3 offline_validate_puzzle.py sample_puzzle.jpg \
  --output offline_puzzle_result.png \
  --json offline_puzzle_result.json
```

当前照片使用 `OFFLINE_A4_CORNERS_PX`，它和板端 `A4_CORNERS_PX` 分开。最新离线结果：

- 识别 4 块，顶点数为 3、4、4、4；P3、P4 左上角的真实短边已保留；
- 中心约为 `(144.2,49.1)`、`(95.2,105.6)`、`(84.8,73.7)`、
  `(63.0,43.6)` mm；
- 分界线约为 `y=142.0 mm`，未作为碎片；
- 固定阈值 180；
- 得到 `FIXED_TOLERANT` 的 100×60 mm 最优方案，保持 P4 左上、P3 中部、
  P2 下部、P1 右上的正确拓扑；
- 方案得分约 `0.0318`，最大对应顶点/边界间隙约 `1.6 mm`，
  小于赛题的 20 mm 要求；
- 估计未覆盖面积约 `85.6 mm²`、边界外面积约 `84.2 mm²`、重叠约
  `4.1 mm²`；这些轻微误差来自手工剪裁和照片测量；
  输出图用灰框明确画出 100×60 mm 目标矩形。

当前完整回归为 76 项；其中性能优化阶段的 65 项测试、性能基线、优化后数据和仍待
上板验证的项目见 `PERFORMANCE_OPTIMIZATION_REPORT.md`。新增 5 项覆盖反光倒角
近点合并、浅共线点删除、点序一致性以及真实短边和凹角保护。

## CanMV v1.6 API 边界

当前实机固件已确认没有可导入的 `cv2`。板端入口不再导入它，启动时只检查
`image.Image` 原生接口：

```text
to_grayscale, rotation_corr, find_blobs, to_numpy_ref,
draw_line, draw_cross, draw_string_advanced
```

状态画布使用 `image.Image(800, 480, image.RGB565)` 和
`Display.show_image`。实时入口还会在轮廓画面右下角绘制实际参与分割的透视校正
灰度工作图，并限频调用 `compress_for_ide()` 将同一张合成画面显式发送给 IDE。
相机、显示与媒体资源在 `finally` 中按
Sensor → Display → exitpoint sleep → MediaManager 顺序释放。桌面离线验证仍使用
OpenCV，不影响已确认的示例照片结果。本地回归包含一个模拟 CanMV 原生图像接口的
测试，但实际速度和内存仍须由板端输出确认。

IDE Stop 在该固件上会抛出文本为 `IDE interrupt` 的异常；入口将其记录为正常
`STOP,reason=ide_interrupt` 后清理资源。脚本结束后 IDE 显示
`REPL input is not supported by this firmware` 是固件不提供交互 REPL 的提示，不是
拼图程序故障。

## 尚未完成的实机指标

以下内容必须在固定好的 K230、镜头和实物上测量，目前不能声称已通过：

- 连续运行 20 分钟；
- 实际 FPS；
- 静止 20 帧的中心/角度标准差；
- IDE Stop 后立即重启；
- 实机 ST7701 文字可读性和内存余量。

建议板端首次测试先保留默认 1.5 px/mm；若 FPS 和内存余量充分，再提高透视图比例。

## 下一阶段控制接口

机械臂或 XYZR 层只需消费每条稳定 `PIECE` 记录：

```text
PIECE,id=P1,sx_mm=72.4,sy_mm=51.8,tx_mm=84.2,ty_mm=216.5,
rot_deg=37.6,ambiguous=0,confidence=0.96
```

坐标原点为 A4 左上，X 向右、Y 向下，单位 mm；旋转逆时针为正，范围
`[-180°, 180°)`。控制层还应携带 `frame`/方案序号，并在执行前拒绝已经被后续
`PLAN_PENDING` 或 `DETECTION_ERROR` 失效的旧方案。
