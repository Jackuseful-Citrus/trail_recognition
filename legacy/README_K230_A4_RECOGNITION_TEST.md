# K230 A4 自动标定独立测试

这个测试入口只包含自动 A4 边界识别、方向判断、分隔线检测和多帧锁定，不会读取
预设 A4 四角，也不会运行碎片识别、拼图规划或放置验证。单文件内已经直接展开全部
参数、检测函数和跟踪器，不需要项目中的任何 Python 模块。

## 上板运行

在 CanMV IDE 中直接打开并运行：

```text
k230_a4_auto_calibration_standalone.py
```

默认相机配置与现有主程序一致：`800×480`、水平镜像、垂直翻转；A4 检测使用
`320×192` 灰度图。程序每帧自动寻找 A4；连续 3 帧识别成功后进入锁定状态，但仍会
持续检测和平滑更新。移动或更换 A4 后会自动重新标定。

画面含义：

- 黄色框：当前帧最佳 A4 候选；
- 绿色框：已锁定的 A4；
- `TL/TR/BR/BL`：物理 A4 坐标方向，不是单纯的画面方向；
- 青色线：识别到的 A4 中部分隔线；
- `R/B/V/D`：原始矩形数、暗色连通域数、有效候选数、分隔线是否有效。

终端会输出 `A4_TEST_START`、`A4_TEST_STATUS` 和 `A4_TEST_LOCK`。锁定结果中的
`corners` 顺序固定为物理 `TL|TR|BR|BL`，可直接复制用于后续手动标定。

## 调试开关

单文件 `k230_a4_auto_calibration_standalone.py` 中的自动标定开关为：

```python
AUTO_CALIBRATE_A4 = True
FREEZE_AFTER_LOCK = False
DETECT_EVERY_N_FRAMES = 1
```

保持上述设置时不使用固定点位，并会持续自动跟踪。若只想在启动时自动标定一次，
可以把 `FREEZE_AFTER_LOCK` 改为 `True`。修改工程中的 A4 核心参数后，也可以用以下
命令重新同步生成：

```bash
python3 build_k230_a4_recognition_test.py
```

生成时会把当前 `puzzle_config.py` 中与 A4 有关的参数复制成单文件顶部的普通常量；
运行单文件时不会再读取 `puzzle_config.py`。
