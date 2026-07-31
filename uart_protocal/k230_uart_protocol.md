# K230 UART 通信协议说明

本文档用于 K230 侧开发者实现与 STM32 主控的 UART 通信。拼图任务当前使用
第 10 节的“单步执行协议”，STM32 会回传 `BUSY/DONE/ERROR`。
第 4 节 `0x01` 视觉目标帧只为兼容旧固件保留，新拼图流程不要使用。

## 1. 物理连接

使用普通 3.3 V TTL UART，不使用 RS485，不使用 5 V UART，不使用 UART5/SBUS。

```text
K230 TX  -> STM32 UART7_RX / PE7
K230 RX  -> STM32 UART7_TX / PE8
K230 GND -> STM32 GND
```

注意事项：

- TX/RX 必须交叉连接。
- 两端必须共地。
- 电平均为 3.3 V TTL。
- 不要把 TTL UART 直接接到 RS485 A/B 线上。

## 2. 串口参数

| 项目 | 参数 |
| --- | --- |
| UART | STM32 UART7 |
| 波特率 | 115200 |
| 数据位 | 8 |
| 校验 | None |
| 停止位 | 1 |
| 流控 | None |
| 字节序 | 小端 |

如果后续目标框、轨迹或控制量发送频率较高，可以双方同步改为 `460800 / 8N1` 或 `921600 / 8N1`。

## 3. 帧格式

每帧格式如下：

```text
header_lsb header_msb command length payload crc_lsb crc_msb
```

| 字段 | 长度 | 值/说明 |
| --- | --- | --- |
| `header_lsb` | 1 byte | `0x55` |
| `header_msb` | 1 byte | `0xAA` |
| `command` | 1 byte | 命令字 |
| `length` | 1 byte | payload 字节数 |
| `payload` | N bytes | 命令数据 |
| `crc_lsb` | 1 byte | CRC16 低字节 |
| `crc_msb` | 1 byte | CRC16 高字节 |

帧头等价于 C 里的 `uint16_t header = 0xAA55`，因为小端传输，所以线上字节顺序是：

```text
55 AA
```

CRC 只覆盖：

```text
command length payload
```

不包含帧头，也不包含 CRC 自身。

## 4. 旧版视觉目标结果帧（兼容保留）

当前 STM32 已实现的命令为 `0x01`，用于发送视觉识别目标结果。

| 字段 | 值 |
| --- | --- |
| `command` | `0x01` |
| `length` | `12` |
| `payload` | 3 个 little-endian IEEE754 float |

payload 排列：

| 偏移 | 类型 | 名称 | 说明 |
| --- | --- | --- | --- |
| 0 | `float32` | `target_x` | 目标 x 坐标 |
| 4 | `float32` | `target_y` | 目标 y 坐标 |
| 8 | `float32` | `confidence` | 置信度，建议范围 `0.0` 到 `1.0` |

完整 C 结构体表达如下，仅用于说明内存布局：

```c
typedef struct __attribute__((packed))
{
    uint16_t header;      // 0xAA55, wire bytes: 55 AA
    uint8_t  command;     // 0x01
    uint8_t  length;      // 12
    float    target_x;    // little-endian float32
    float    target_y;    // little-endian float32
    float    confidence;  // little-endian float32
    uint16_t crc16;       // little-endian, CRC over command+length+payload
} K230TargetPacket_t;
```

## 5. CRC16 算法

CRC 使用 `CRC-16/CCITT-FALSE`：

| 参数 | 值 |
| --- | --- |
| Polynomial | `0x1021` |
| Init | `0xFFFF` |
| RefIn | false |
| RefOut | false |
| XorOut | `0x0000` |
| 输出字节序 | little-endian，低字节在前 |

C 参考实现：

```c
static uint16_t crc16_ccitt_false(const uint8_t *data, uint16_t len)
{
    uint16_t crc = 0xFFFF;

    for (uint16_t i = 0; i < len; ++i) {
        crc ^= (uint16_t)data[i] << 8;
        for (uint8_t bit = 0; bit < 8; ++bit) {
            if ((crc & 0x8000) != 0) {
                crc = (uint16_t)((crc << 1) ^ 0x1021);
            } else {
                crc <<= 1;
            }
        }
    }

    return crc;
}
```

## 6. 发送示例

假设 K230 要发送：

```text
target_x = 1.0
target_y = 2.0
confidence = 0.5
```

float32 little-endian payload 为：

```text
1.0f -> 00 00 80 3F
2.0f -> 00 00 00 40
0.5f -> 00 00 00 3F
```

参与 CRC 的数据为：

```text
01 0C 00 00 80 3F 00 00 00 40 00 00 00 3F
```

CRC16/CCITT-FALSE 结果为：

```text
0x824A
```

最终整帧线上字节为：

```text
55 AA 01 0C 00 00 80 3F 00 00 00 40 00 00 00 3F 4A 82
```

## 7. K230 侧发送建议

- 建议固定周期发送，例如 20 Hz 到 100 Hz，根据视觉算法输出频率决定。
- 没检测到目标时，可以发送 `confidence = 0.0`，`target_x/target_y` 可置 0。
- 不要在 payload 中发送 ASCII 字符串；当前 STM32 解析的是二进制 float。
- 每次发送必须包含完整帧，不能只发 payload。
- 如果 STM32 侧 `rx_bytes` 增加但 `valid_frames` 不增加，优先检查帧头、长度、CRC 和字节序。

## 8. STM32 侧调试观测量

STM32 固件提供以下 watch 变量，可通过 GDB/SWD 观察：

```c
volatile K230_Target_t g_k230_target;
volatile K230_UartStats_t g_k230_uart_stats;
```

关键计数含义：

| 变量 | 说明 |
| --- | --- |
| `g_k230_uart_stats.rx_events` | UART IDLE/DMA 接收事件次数 |
| `g_k230_uart_stats.rx_bytes` | STM32 收到的原始字节数 |
| `g_k230_uart_stats.valid_frames` | CRC 正确的完整帧数 |
| `g_k230_uart_stats.crc_errors` | CRC 错误帧数 |
| `g_k230_uart_stats.format_errors` | 长度或格式错误次数 |
| `g_k230_uart_stats.ring_overflows` | STM32 环形缓冲区溢出次数 |
| `g_k230_target.target_x` | 最近一次有效目标 x |
| `g_k230_target.target_y` | 最近一次有效目标 y |
| `g_k230_target.confidence` | 最近一次有效目标置信度 |
| `g_k230_target.last_update_ms` | 最近一次有效帧更新时间，单位 ms |

排查顺序：

1. 先确认 `rx_bytes` 是否增加，判断物理收发是否存在。
2. 再看 `valid_frames` 是否增加，判断帧头、长度和 CRC 是否正确。
3. 如果 `crc_errors` 增加，重点检查 CRC 覆盖范围、CRC 字节序和 float 字节序。
4. 如果 `format_errors` 增加，重点检查 `length` 是否为 12，以及是否混入了额外字节。

## 9. 拼图任务协议确认项与待确认问题

本节只记录和拼图任务业务相关的问题。帧头、`command`、`length`、payload、CRC16 等基础帧格式保持前文约定，不在这里重复确认。

### 9.1 当前已确认结论

- 拼图任务采用“执行完一块后再请求下一块”的交互方式，不一次下发整套拼图列表。
- 一块拼图的高层数据至少包含：起始 `x/y`、终止 `x/y`、旋转角度。
- 坐标单位使用 `mm`。
- 坐标原点为机械上电后的零点。
- 坐标统一使用机械坐标，也就是下位机实际执行所用坐标；视觉侧不要发送图像像素坐标。
- 视觉侧发送坐标时，需要考虑目标位置和实际位置之间的距离偏差，把最终可执行的机械坐标发给下位机。
- 角度单位使用 `degree`，字段名建议为 `angle_deg`。
- 角度范围使用 `-180~180`。
- 角度正方向为顺时针。
- 角度零位为机械夹爪零位。
- 不区分 `pick_angle_deg` 和 `place_angle_deg`，先只传一个角度。
- 需要考虑逐步执行数据：`x/y/z` 和 `isMagnet`。每执行一步，下位机返回完成状态，上位机再发下一步。

### 9.2 当前最大的协议层级选择

现在有两种可行层级，需要最终选择一种：

1. 单块拼图任务：上位机发起始点、终止点和角度，STM32 自己拆解 Z 轴、吸附、移动和释放流程。
2. 单步运动指令：上位机每一步直接发 `x/y/z/angle/isMagnet`，STM32 执行完回完成状态，上位机再发下一步。

如果希望 K230/上位机完全掌控流程，建议选第 2 种。如果希望 STM32 自己管理吸盘和 Z 轴动作，建议选第 1 种。

### 9.3 坐标定义

已确认：

- `x/y` 单位使用 `mm`。
- `x/y` 坐标原点使用机械上电后的零点。
- 坐标统一使用机械坐标，即下位机实际执行坐标。
- 高层拼图任务需要区分起始点 `start_x_mm/start_y_mm` 和终止点 `end_x_mm/end_y_mm`。
- 只发送一个 `x/y` 的方案不适合当前拼图任务，容易造成错误信息。

仍需确认：

1. `x` 正方向是什么：向右、向左，还是机械坐标系某一轴正方向？
2. `y` 正方向是什么：向前、向后，还是机械坐标系某一轴正方向？
3. `z` 使用连续坐标 `z_mm`，还是只用 `up/down` 枚举？
4. `z` 的零点、正方向和安全高度是多少？
5. 坐标有效范围是多少，例如 `x/y/z` 的最小值和最大值，用于 STM32 做非法值保护。
6. 视觉侧如何补偿目标位置和实际位置之间的距离偏差，需要在 K230 算法侧明确。

### 9.4 角度定义

已确认：

- 角度单位使用度 `degree`。
- 字段名建议为 `angle_deg`。
- 角度范围使用 `-180~180`。
- 角度正方向为顺时针。
- 角度零位为机械夹爪零位。
- 不区分拾取角和放置角。

仍需确认：

1. 单个 `angle_deg` 的业务含义是什么：从起始姿态到终止姿态的旋转量，还是最终放置姿态相对夹爪零位的目标角？
2. 角度允许误差是多少，例如 `±2 deg`、`±5 deg`，用于判断是否对准？

### 9.5 动作和状态语义

已确认方向：

- 上位机/K230 发任务或单步指令。
- 下位机执行完成后返回完成状态。
- 上位机收到完成状态后再发下一步或下一块。
- 如果采用单步指令，payload 应包含 `x/y/z/angle/isMagnet`。

仍需确认：

1. 最终选择“单块拼图任务”还是“单步运动指令”。
2. `isMagnet` 是否定义为 `0 = release/off`，`1 = suck/on`。
3. 是否需要 `valid` 字段表示本帧目标有效，避免“没识别到目标”和“沿用上一帧”混淆。
4. 没识别到拼图时，K230 是不发帧，还是发 `valid = 0` 的帧？
5. 下位机完成状态是否至少包含 `busy`、`done`、`error`。
6. 错误码 `error_code` 需要定义哪些值，例如坐标超限、执行超时、吸附失败、CRC 错误等。

### 9.6 时序和安全

仍需确认：

1. K230/上位机发送频率是多少：固定周期发送，还是只在下一步任务需要执行时发送？
2. STM32 多久收不到新帧算通信超时，例如 `200 ms`、`500 ms` 或 `1000 ms`？
3. 超时后 STM32 应该保持上一目标、停止动作，还是进入安全状态？
4. 如果 CRC 正确但坐标超范围，STM32 应该忽略该帧、停止任务，还是回传错误？
5. 如果上位机连续发送同一个任务，STM32 是否允许重复执行，还是必须用 `sequence` 去重？
6. 执行中收到新任务时，STM32 应该打断当前任务、排队等待，还是忽略直到当前任务完成？

### 9.7 候选 payload 草案

候选方案 A：一帧一块拼图任务。上位机只发起始点、终止点和角度，STM32 自己拆解 Z 轴和吸盘动作。

```c
typedef struct __attribute__((packed))
{
    uint8_t  version;          // 先填 1
    uint8_t  sequence;         // 每发一块递增，用于去重
    uint8_t  piece_id;         // 拼图片编号，没有编号时填 0
    uint8_t  flags;            // bit0 valid，其余预留

    float    start_x_mm;       // 起始点 x，机械坐标，单位 mm
    float    start_y_mm;       // 起始点 y，机械坐标，单位 mm
    float    end_x_mm;         // 终止点 x，机械坐标，单位 mm
    float    end_y_mm;         // 终止点 y，机械坐标，单位 mm
    float    angle_deg;        // 旋转角度，-180~180，顺时针为正
} K230PuzzlePiecePayloadDraft_t;
```

候选方案 B：一帧一步运动指令。上位机每次直接发一个目标点、Z 轴位置和吸附状态；STM32 执行完成后返回状态，上位机再发下一步。

```c
typedef struct __attribute__((packed))
{
    uint8_t  version;          // 先填 1
    uint8_t  sequence;         // 每发一步递增，用于去重
    uint8_t  step_type;        // move / suck / release / wait 等，待定义
    uint8_t  flags;            // bit0 valid，其余预留

    float    x_mm;             // 目标 x，机械坐标，单位 mm
    float    y_mm;             // 目标 y，机械坐标，单位 mm
    float    z_mm;             // 目标 z，机械坐标，单位 mm；也可后续改成 up/down 枚举
    float    angle_deg;        // 目标角度，-180~180，顺时针为正

    uint8_t  is_magnet;        // 0 = release/off, 1 = suck/on
    uint8_t  reserved[3];      // 对齐和预留，先填 0
} K230PuzzleStepPayloadDraft_t;
```

候选方案 B 更符合“每执行一步下位机给完成状态，上位机发下一步”的流程，但它要求上位机/K230 明确规划 Z 轴和吸盘时序。

下位机返回状态的候选 payload：

```c
typedef struct __attribute__((packed))
{
    uint8_t version;           // 先填 1
    uint8_t sequence;          // 对应已执行的上位机 sequence
    uint8_t status;            // 0 busy, 1 done, 2 error
    uint8_t error_code;        // 0 表示无错误
} STM32PuzzleStatusPayloadDraft_t;
```

## 10. 拼图单步执行协议（当前正式版本）

第 9 节是设计确认过程，本节是 K230 和 STM32 当前实际实现，若两节冲突以本节为准。

### 10.1 通信时序

```text
K230                 STM32
  |---- 单步 0x10 ---->|
  |<--- BUSY 0x90 -----|  已接收并开始执行
  |                     |  执行机械动作
  |<--- DONE 0x90 -----|  单步完成
  |---- 下一单步 ------>|
```

K230 必须等到当前 `sequence` 收到 `DONE` 或 `ERROR`，才能发送下一步。

### 10.2 K230 下发单步命令

| 项目 | 值 |
| --- | --- |
| `command` | `0x10` |
| `length` | `8` |

payload 固定为 8 字节：

| 偏移 | 类型 | 名称 | 值/说明 |
| --- | --- | --- | --- |
| 0 | `uint8` | `version` | 固定 `1` |
| 1 | `uint8` | `sequence` | 每个新动作递增，`255` 后回绕到 `0` |
| 2 | `uint8` | `action` | 动作类型，见下表 |
| 3 | `uint8` | `flags` | 固定 `0x01`，bit0 表示 valid |
| 4 | `float32` | `value` | 小端 IEEE754，含义由 action 决定 |

动作定义：

| action | 名称 | value | 完成条件 |
| --- | --- | --- | --- |
| `1` | `MOVE_X_ABS` | X 绝对坐标，mm | XY 规划结束且实际位置进入容差 |
| `2` | `MOVE_Y_ABS` | Y 绝对坐标，mm | XY 规划结束且实际位置进入容差 |
| `3` | `ROTATE_REL` | 相对旋转角度，degree | C610 旋转动作启动并结束 |
| `4` | `GRIP` | `0.0=放置`，`1.0=夹取` | 舵机下放、切换电磁铁、舵机抬起 |
| `5` | `Z` | `0.0=上`，`1.0=下` | 调试保留；正式流程不建议 K230 单独使用 |

X/Y 是机械绝对坐标。当前原点是上电软件零点（左上角），X 向右为正，
Y 向下为正；X 有效范围 `0~181 mm`，Y 有效范围 `0~305 mm`。
移动 X 时保持当前 Y 目标不变，移动 Y 时保持当前 X 目标不变。

旋转为相对角度，范围 `-180~180 degree`，顺时针为正。
这是协议坐标定义；首次上板必须低速验证 C610 当前 `direction=+1` 的机械正向确实为顺时针，
若相反应在 STM32 侧统一校准方向，K230 不要临时把角度符号取反。

```c
typedef struct __attribute__((packed))
{
    uint8_t version;       // 1
    uint8_t sequence;      // 每个新动作递增
    uint8_t action;        // 1~5
    uint8_t flags;         // 0x01
    float   value;         // little-endian float32
} K230StepPayload_t;
```

### 10.3 STM32 状态返回

| 项目 | 值 |
| --- | --- |
| `command` | `0x90` |
| `length` | `8` |

| 偏移 | 类型 | 名称 | 说明 |
| --- | --- | --- | --- |
| 0 | `uint8` | `version` | 固定 `1` |
| 1 | `uint8` | `sequence` | 原样返回对应命令序号 |
| 2 | `uint8` | `action` | 原样返回动作类型 |
| 3 | `uint8` | `status` | `0=BUSY`，`1=DONE`，`2=ERROR` |
| 4 | `uint8` | `error_code` | `DONE/BUSY` 时为 `0` |
| 5 | `uint8[3]` | `reserved` | 固定填 `0` |

错误码：`1` 版本错误，`2` flags 错误，`3` 不支持的 action，`4` value 非法，
`5` 超范围，`6` 执行器正忙，`7` 硬件未就绪，`8` 执行超时，
`9` 相同 sequence 对应不同命令内容。

### 10.4 重发和防重复执行

- K230 超时未收到状态时，可以原样重发同一 `sequence`、`action` 和 `value`。
- STM32 收到完全相同的重发帧，只重发当前 `BUSY` 或最近终态，不重复执行。
- 同一 `sequence` 修改 action/value 会返回错误码 `9`。
- STM32 忙时收到其他序号会返回错误码 `6`，不会打断当前机械动作。

### 10.5 完成状态的硬件含义

- X/Y 的 `DONE` 使用电机反馈位置判定，当前到位容差是 `0.1 mm`。
- 旋转的 `DONE` 需要观察到 C610 `move_active` 从运行变为结束，当前容差为 `1 degree`。
- `GRIP=1` 的动作顺序是：舵机下放、电磁铁吸合、舵机抬起。
- `GRIP=0` 的动作顺序是：舵机下放、电磁铁释放、舵机抬起。
- 电磁铁没有吸力传感器，`DONE` 只代表 GPIO 已切换并等待了稳定时间。
- Z 舵机没有位置反馈，`DONE` 只代表 PWM 已下发并等待了预设动作时间。
- 因此吸附成功、拼图是否掉落仍需 K230 视觉或后续传感器确认。

### 10.6 STM32 调试计数

重点观察：`rx_bytes`、`valid_frames`、`step_frames`、`tx_frames`、`tx_bytes`、
`crc_errors`、`tx_errors`，以及 `g_k230_step_status` 的当前序号、动作、完成数和错误。
先确认 RX/TX 原始计数变化，再判断帧、CRC 和业务执行结果。

### 10.7 十六进制示例

`sequence=1`，动作 `MOVE_X_ABS`，目标 `100.0 mm`：

```text
K230 -> STM32:
55 AA 10 08 01 01 01 01 00 00 C8 42 68 BB

STM32 -> K230, BUSY:
55 AA 90 08 01 01 01 00 00 00 00 00 76 02

STM32 -> K230, DONE:
55 AA 90 08 01 01 01 01 00 00 00 00 27 A8
```

以上 CRC 均按第 5 节的 CRC-16/CCITT-FALSE 计算，CRC 低字节先发送。
