# K230 自动交互验证报告

日期：2026-07-29  
结论：Codex 已从当前 Ubuntu 环境建立无需 GUI 点击的 K230 自动控制链路。

## 结果

| 项目 | 结果 | 证据 |
|---|---|---|
| 系统识别 K230 | 成功 | `/dev/ttyACM2`，VID:PID `1209:abd1`，序列号 `001000000` |
| 连接板端 | 成功 | `CanMV K230 V3.0 - 1G`，握手固件 `v0.4.0` |
| 传送探针 | 成功 | 通过 CanMV IDE `ScriptExec` 将源码传入 RAM；未写 SD 卡 |
| 启动探针 | 成功 | 两次启动均返回 `{"status":"ok"}` |
| 读取板端输出 | 成功 | 两次均读到 BEGIN、0–9 heartbeat、END |
| 停止程序 | 成功 | 两次均看到 `Exception: IDE interrupt` 和软重启 |
| 再次运行 | 成功 | 第二次完整重复 heartbeat |
| framebuffer | 成功 | 板端合成图像通过 IDE framebuffer 返回 17,175 字节 JPEG |
| 持久文件上传 | 当前固件不支持 | protocol 0 的 `writeFile`、`fileExec`、`listDir` 均为 false |

板端运行时输出：

```text
CanMV v1.6(based on Micropython e00a144) on 2026-04-03
MicroPython implementation version=(1, 21, 0)
machine=k230_canmv_v3p0 with K230
```

## 设备与权限

- `lsusb`：`1209:abd1 Generic OpenMV Cam`
- udev：`ID_VENDOR=Kendryte`、`ID_MODEL=CanMV`
- 唯一确认设备：`/dev/ttyACM2`
- `/dev/ttyACM0`、`/dev/ttyACM1` 属于 `1a86:55d2 QinHeng USB Dual_Serial`，未被误选
- 当前用户 `jjjack` 属于 `dialout`，串口节点可读写
- `dmesg` 被 Ubuntu 普通用户策略拒绝：`Operation not permitted`；没有使用 sudo

## CanMV 扩展审计

- 扩展：`kendryte747.canmv-vscode@0.9.6`
- 目录：`/home/jjjack/.vscode/extensions/kendryte747.canmv-vscode-0.9.6`
- 扩展注册了连接、断开、运行、停止、预览、远程文件、示例和工具面板等 24 个命令。
- 运行类命令包括 `canmv.runCurrentScript`、`canmv.stopScript`、
  `canmv.runRemoteFile`、`canmv.runExampleFile` 和 `canmv.runOnK230`。
- 扩展没有注册 URI handler，也没有贡献 VS Code task definition；`code` CLI
  没有通用的扩展命令调用参数。
- 扩展提供 `canmv.mcp` stdio MCP server definition，并打包独立
  `out/mcp/server.js` 与 `bin/linux-x64/canmv-backend`。
- MCP 服务公开设备发现、连接、脚本执行、停止、终端输出、文件操作和
  framebuffer 工具，因此正式控制路径不需要 GUI。

## 实际控制路径

```text
tools/k230_bridge.py
  -> CanMV 扩展的独立 stdio MCP server
  -> 扩展自带 canmv-backend
  -> USBDBG / CDC ACM
  -> K230 CanMV MicroPython
```

当前固件只支持 legacy protocol 0。探针和后续候选程序使用官方 IDE
`ScriptExec` 直接传入 RAM 并运行，不创建或覆盖 `/sdcard/main.py`。
`deploy` 命令已实现；在能力位没有 `writeFile` 的当前固件上会返回非零
`unsupported_firmware`，而不是伪报上传成功。

## 关键执行命令

```text
lsusb
udevadm info --query=property --name=/dev/ttyACM{0,1,2}
id
groups
dmesg --color=never
code --list-extensions --show-versions
rg / sed <CanMV package.json and installed extension sources>
lsof /dev/ttyACM2
kill -STOP <CanMV backend PID>   # 只读占用诊断后立即恢复
kill -TERM <CanMV backend PID>   # 释放扩展的独占串口会话
python3 tools/k230_bridge.py probe
python3 tools/k230_bridge.py run tools/k230_probe_payload.py --wait 6 --until @@K230_PROBE_END
python3 tools/k230_bridge.py logs
python3 tools/k230_bridge.py deploy tools/k230_probe_payload.py
python3 tools/k230_bridge.py stop
```

所有硬件会话都有超时；`probe` 硬限制最多 30 秒。桥接只会终止经
`/proc/<pid>/exe` 确认属于当前 CanMV 扩展目录的串口占用进程，发现其他
占用者时会拒绝操作。

## 新增文件

- `tools/k230_bridge.py`：机器可读 JSON CLI；支持 `probe`、`deploy`、
  `run`、`stop`、`logs`、`capture`
- `tools/k230_probe_payload.py`：无外设 heartbeat 探针
- `tools/k230_framebuffer_probe.py`：无摄像头、无 GPIO 的内存合成帧探针
- `artifacts/k230_bridge/20260729T154813.053220Z_probe.json`：最终权威会话记录
- `artifacts/k230_bridge/20260729T154813.053220Z_frame.jpg`：板端 framebuffer
- `artifacts/k230_bridge/last_terminal.log`：完整板端终端输出

## 夜间闭环判断

可以建立“自动修改 → RAM ScriptExec 运行 → 解析 stdout/framebuffer →
评分 → 继续优化”的闭环。桥接默认返回 JSON、保存原始日志、失败返回非零
状态，并支持显式超时。

当前限制是不能通过这版板端固件持久上传候选文件或远程执行 SD 卡文件。
只要候选源码和评分结果可通过内存执行与终端/framebuffer 完成，夜间闭环
可直接使用；若闭环必须上传模型或其他资产，需要以后在单独授权下升级到
暴露 `writeFile`/`fileExec` 能力的固件，或使用已验证的网络传输路径。
