#!/usr/bin/env python3
"""Headless CanMV K230 bridge using the installed VS Code extension backend.

The bridge never writes /sdcard/main.py or /sdcard/boot.py.  On legacy
firmware, where remote-file capabilities are unavailable, scripts are sent
through the CanMV IDE ScriptExec path and run from RAM.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import glob
import json
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import sys
import threading
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROBE_SOURCE = Path(__file__).with_name("k230_probe_payload.py")
FRAMEBUFFER_PROBE_SOURCE = Path(__file__).with_name("k230_framebuffer_probe.py")
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "k230_bridge"
LAST_SESSION = ARTIFACT_DIR / "last_session.json"
LAST_TERMINAL = ARTIFACT_DIR / "last_terminal.log"
EXPECTED_VID_PID = ("1209", "abd1")
DEFAULT_BAUD = 12_000_000
DEFAULT_TIMEOUT = 30.0
FORBIDDEN_REMOTE_PATHS = {
    "/sdcard/main.py",
    "/sdcard/boot.py",
    "/flash/main.py",
    "/flash/boot.py",
}


class BridgeError(RuntimeError):
    """An expected bridge failure with a machine-readable category."""

    def __init__(self, message: str, category: str = "bridge_error") -> None:
        super().__init__(message)
        self.category = category


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def extension_path() -> Path:
    candidates = [
        Path(item)
        for item in glob.glob(
            str(Path.home() / ".vscode" / "extensions" / "kendryte747.canmv-vscode-*")
        )
        if Path(item, "package.json").is_file()
        and Path(item, "out", "mcp", "server.js").is_file()
    ]
    if not candidates:
        raise BridgeError(
            "Installed kendryte747.canmv-vscode extension was not found",
            "extension_not_found",
        )

    def version_key(path: Path) -> tuple[int, ...]:
        match = re.search(r"canmv-vscode-(\d+(?:\.\d+)*)$", path.name)
        return tuple(int(part) for part in match.group(1).split(".")) if match else ()

    return max(candidates, key=version_key)


def run_readonly(command: list[str], timeout: float = 5.0) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }


def command_audit(ext_path: Path) -> dict[str, Any]:
    package_path = ext_path / "package.json"
    package = json.loads(package_path.read_text(encoding="utf-8"))
    source = (ext_path / "out" / "extension.js").read_text(
        encoding="utf-8", errors="replace"
    )
    mcp_source = (ext_path / "out" / "mcp" / "server.js").read_text(
        encoding="utf-8", errors="replace"
    )
    commands = [
        item["command"]
        for item in package.get("contributes", {}).get("commands", [])
        if isinstance(item, dict) and isinstance(item.get("command"), str)
    ]
    return {
        "name": package.get("displayName"),
        "id": f"{package.get('publisher')}.{package.get('name')}",
        "version": package.get("version"),
        "directory": str(ext_path),
        "commands": commands,
        "mcp_server_provider": package.get("contributes", {}).get(
            "mcpServerDefinitionProviders", []
        ),
        "standalone_mcp_server": str(ext_path / "out" / "mcp" / "server.js"),
        "bundled_backend": str(
            ext_path / "bin" / "linux-x64" / "canmv-backend"
        ),
        "registered_uri_handler": "registerUriHandler" in source,
        "contributed_tasks": bool(
            package.get("contributes", {}).get("taskDefinitions")
        ),
        "standalone_stdio_json_rpc": (
            "process.stdin.on('data'" in mcp_source
            and "case 'tools/call':" in mcp_source
        ),
    }


class McpClient:
    def __init__(self, ext_path: Path, deadline: float) -> None:
        self.ext_path = ext_path
        self.deadline = deadline
        self.request_id = 0
        self.stderr_lines: list[str] = []
        package = json.loads(
            (ext_path / "package.json").read_text(encoding="utf-8")
        )
        env = os.environ.copy()
        env.update(
            {
                "CANMV_EXTENSION_PATH": str(ext_path),
                "CANMV_EXTENSION_VERSION": str(package.get("version", "unknown")),
                "CANMV_BAUD_RATE": str(DEFAULT_BAUD),
                "CANMV_MCP_IDLE_DISCONNECT_MS": "30000",
                "CANMV_MCP_OUTPUT_DIR": str(ARTIFACT_DIR),
            }
        )
        self.process = subprocess.Popen(
            ["node", str(ext_path / "out" / "mcp" / "server.js")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            cwd=ext_path,
        )
        self.stderr_thread = threading.Thread(
            target=self._drain_stderr, name="canmv-mcp-stderr", daemon=True
        )
        self.stderr_thread.start()

    def _drain_stderr(self) -> None:
        if not self.process.stderr:
            return
        for line in self.process.stderr:
            self.stderr_lines.append(line)

    def remaining(self, maximum: float | None = None) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise BridgeError("Command exceeded its hard timeout", "timeout")
        return min(remaining, maximum) if maximum is not None else remaining

    def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None, timeout: float = 16
    ) -> Any:
        self.request_id += 1
        request_id = self.request_id
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        }
        if not self.process.stdin or not self.process.stdout:
            raise BridgeError("MCP stdio pipes are unavailable", "mcp_io")
        try:
            self.process.stdin.write(
                json.dumps(request, separators=(",", ":")) + "\n"
            )
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise BridgeError(f"MCP request write failed: {exc}", "mcp_io") from exc

        request_deadline = time.monotonic() + self.remaining(timeout)
        while time.monotonic() < request_deadline:
            ready, _, _ = select.select(
                [self.process.stdout],
                [],
                [],
                min(0.2, max(0.0, request_deadline - time.monotonic())),
            )
            if not ready:
                continue
            line = self.process.stdout.readline()
            if not line:
                break
            response = json.loads(line)
            if response.get("id") != request_id:
                continue
            if "error" in response:
                error = response["error"]
                raise BridgeError(
                    f"{name}: {error.get('message', error)}", "mcp_rpc"
                )
            result = response.get("result", {})
            content = result.get("content", [])
            text = content[0].get("text", "") if content else ""
            if result.get("isError"):
                raise BridgeError(f"{name}: {text}", "board_protocol")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"text": text}
        raise BridgeError(f"{name} timed out", "timeout")

    def close(self) -> None:
        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        try:
            self.process.wait(timeout=min(2.0, max(0.1, self.remaining(2.0))))
        except (subprocess.TimeoutExpired, BridgeError):
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1)
        self.stderr_thread.join(timeout=0.5)


def serial_holders(port: str) -> list[int]:
    result = run_readonly(["lsof", "-t", port], timeout=2)
    holders: list[int] = []
    for line in result["stdout"].splitlines():
        try:
            holders.append(int(line.strip()))
        except ValueError:
            continue
    return sorted(set(holders))


def validated_backend(pid: int, ext_path: Path) -> dict[str, Any]:
    try:
        executable = Path(f"/proc/{pid}/exe").resolve()
        command_line = (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode("utf-8", "replace")
            .strip()
        )
    except OSError as exc:
        raise BridgeError(
            f"Cannot identify process {pid} holding the K230 serial port: {exc}",
            "serial_busy",
        ) from exc
    expected_root = (ext_path / "bin").resolve()
    if executable.name != "canmv-backend" or expected_root not in executable.parents:
        raise BridgeError(
            f"Refusing to terminate unrelated serial holder PID {pid}: {executable}",
            "serial_busy",
        )
    return {"pid": pid, "executable": str(executable), "command_line": command_line}


def release_extension_serial(port: str, ext_path: Path) -> list[dict[str, Any]]:
    released: list[dict[str, Any]] = []
    for pid in serial_holders(port):
        holder = validated_backend(pid, ext_path)
        os.kill(pid, signal.SIGTERM)
        holder["action"] = "SIGTERM"
        released.append(holder)
    deadline = time.monotonic() + 0.8
    while time.monotonic() < deadline:
        remaining = [
            pid for pid in serial_holders(port) if any(x["pid"] == pid for x in released)
        ]
        if not remaining:
            return released
        time.sleep(0.03)
    if released:
        raise BridgeError(
            "CanMV extension backend did not release the serial port",
            "serial_busy",
        )
    return released


class BoardSession:
    def __init__(self, timeout: float) -> None:
        if not 1 <= timeout <= 600:
            raise BridgeError("Timeout must be between 1 and 600 seconds", "arguments")
        self.started = time.monotonic()
        self.deadline = self.started + timeout
        self.ext_path = extension_path()
        self.client = McpClient(self.ext_path, self.deadline)
        self.board: dict[str, Any] | None = None
        self.released_holders: list[dict[str, Any]] = []
        self.connected = False

    def connect(self) -> dict[str, Any]:
        detected = self.client.call_tool("canmv_detect_boards", timeout=5)
        candidates = [
            board
            for board in detected.get("boards", [])
            if (
                str(board.get("vid", "")).lower(),
                str(board.get("pid", "")).lower(),
            )
            == EXPECTED_VID_PID
            and "canmv" in str(board.get("name", "")).lower()
            and "k230" in str(board.get("name", "")).lower()
        ]
        if len(candidates) != 1:
            raise BridgeError(
                f"Expected exactly one confirmed CanMV K230, found {len(candidates)}",
                "board_detection",
            )
        self.board = candidates[0]
        port = str(self.board["port"])
        self.released_holders = release_extension_serial(port, self.ext_path)
        last_error: Exception | None = None
        for _ in range(3):
            try:
                connected = self.client.call_tool(
                    "canmv_connect_board",
                    {"port": port, "baudRate": DEFAULT_BAUD},
                    timeout=12,
                )
                self.connected = True
                return connected
            except BridgeError as exc:
                last_error = exc
                if "busy" not in str(exc).lower():
                    raise
                self.released_holders.extend(
                    release_extension_serial(port, self.ext_path)
                )
                time.sleep(0.05)
        raise BridgeError(
            f"Could not acquire K230 serial port: {last_error}", "serial_busy"
        )

    def close(self) -> None:
        if self.connected:
            try:
                self.client.call_tool("canmv_disconnect_board", timeout=3)
            except BridgeError:
                pass
            self.connected = False
        self.client.close()

    @property
    def backend_log(self) -> str:
        return "".join(self.client.stderr_lines)


def host_audit(ext_path: Path, port: str) -> dict[str, Any]:
    audit = {
        "lsusb": run_readonly(["lsusb"]),
        "serial_node": run_readonly(["udevadm", "info", "--query=property", f"--name={port}"]),
        "identity": run_readonly(["id"]),
        "groups": run_readonly(["groups"]),
        "dmesg": run_readonly(
            ["dmesg", "--color=never", "--since", "10 minutes ago"]
        ),
        "extension": command_audit(ext_path),
    }
    try:
        stat = Path(port).stat()
        audit["serial_access"] = {
            "readable": os.access(port, os.R_OK),
            "writable": os.access(port, os.W_OK),
            "mode": oct(stat.st_mode & 0o777),
            "uid": stat.st_uid,
            "gid": stat.st_gid,
        }
    except OSError as exc:
        audit["serial_access"] = {"error": str(exc)}
    return audit


def terminal_output(client: McpClient, clear: bool = False) -> str:
    result = client.call_tool("canmv_terminal_output", {"clear": clear}, timeout=3)
    return str(result.get("text", ""))


def wait_for_marker(client: McpClient, marker: str, timeout: float) -> str:
    deadline = time.monotonic() + min(timeout, client.remaining())
    output = ""
    while time.monotonic() < deadline:
        output = terminal_output(client)
        if marker in output:
            return output
        time.sleep(0.15)
    raise BridgeError(
        f"Timed out waiting for board marker {marker!r}; tail={output[-500:]!r}",
        "probe_output",
    )


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def save_record(record: dict[str, Any], terminal: str = "") -> dict[str, str]:
    stamp = record.get("stamp") or utc_stamp()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    record_path = ARTIFACT_DIR / f"{stamp}_{record.get('command', 'session')}.json"
    encoded = (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode()
    atomic_write(record_path, encoded)
    atomic_write(LAST_SESSION, encoded)
    atomic_write(LAST_TERMINAL, terminal.encode("utf-8", "surrogateescape"))
    return {
        "record": str(record_path),
        "last_session": str(LAST_SESSION),
        "terminal_log": str(LAST_TERMINAL),
    }


def load_script(path_text: str) -> tuple[Path, str]:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise BridgeError(f"Script not found: {path}", "arguments")
    data = path.read_bytes()
    if len(data) > 2 * 1024 * 1024:
        raise BridgeError("Script exceeds the 2 MiB safety limit", "arguments")
    try:
        return path, data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BridgeError("Script must be UTF-8 text", "arguments") from exc


def base_record(command: str, session: BoardSession) -> dict[str, Any]:
    return {
        "ok": False,
        "command": command,
        "stamp": utc_stamp(),
        "transport": "CanMV VS Code extension standalone MCP -> bundled canmv-backend -> USBDBG",
        "timeout_seconds": session.deadline - session.started,
    }


def perform_probe(timeout: float) -> dict[str, Any]:
    if timeout > 30:
        raise BridgeError("Probe timeout may not exceed 30 seconds", "arguments")
    session = BoardSession(timeout)
    record = base_record("probe", session)
    terminal_runs: list[str] = []
    try:
        connected = session.connect()
        record.update(
            {
                "device": session.board,
                "connection": connected,
                "released_serial_holders": session.released_holders,
                "host_audit": host_audit(
                    session.ext_path, str(session.board["port"])
                ),
            }
        )
        record["initial_stop"] = session.client.call_tool(
            "canmv_stop_script", timeout=4
        )
        record["firmware"] = session.client.call_tool(
            "canmv_firmware_info", timeout=4
        )
        record["capabilities"] = session.client.call_tool(
            "canmv_board_capabilities", timeout=4
        )
        source = PROBE_SOURCE.read_text(encoding="utf-8")
        record["probe_source"] = str(PROBE_SOURCE)
        record["runs"] = []
        for run_number in (1, 2):
            terminal_output(session.client, clear=True)
            start = session.client.call_tool(
                "canmv_run_script", {"script": source}, timeout=8
            )
            wait_for_marker(session.client, "@@K230_PROBE_END", timeout=6)
            stop = session.client.call_tool("canmv_stop_script", timeout=4)
            time.sleep(0.35)
            output = terminal_output(session.client)
            terminal_runs.append(output)
            heartbeat_indices = [
                int(match.group(1))
                for match in re.finditer(r"@@K230_HEARTBEAT[ \t]+(\d+)", output)
            ]
            run = {
                "run": run_number,
                "start": start,
                "stop": stop,
                "output": output,
                "begin_seen": "@@K230_PROBE_BEGIN" in output,
                "end_seen": "@@K230_PROBE_END" in output,
                "heartbeat_indices": heartbeat_indices,
                "stop_interrupt_seen": "IDE interrupt" in output,
                "soft_reboot_seen": "MPY: soft reboot" in output,
            }
            run["valid"] = (
                run["begin_seen"]
                and run["end_seen"]
                and heartbeat_indices == list(range(10))
                and run["stop_interrupt_seen"]
            )
            record["runs"].append(run)

        record["framebuffer"] = {"available": False}
        try:
            terminal_output(session.client, clear=True)
            frame_source = FRAMEBUFFER_PROBE_SOURCE.read_text(encoding="utf-8")
            record["framebuffer_probe_source"] = str(FRAMEBUFFER_PROBE_SOURCE)
            record["framebuffer_run"] = session.client.call_tool(
                "canmv_run_script", {"script": frame_source}, timeout=8
            )
            wait_for_marker(
                session.client, "@@K230_FRAMEBUFFER_READY", timeout=4
            )
            record["preview_start"] = session.client.call_tool(
                "canmv_start_preview", timeout=3
            )
            # Let the backend receive a frame before querying it. Calling the
            # MCP wait path with no cached frame reconnects legacy firmware.
            time.sleep(2)
            frame = session.client.call_tool(
                "canmv_get_latest_frame",
                {"waitMs": 0, "fresh": False},
                timeout=min(3, session.client.remaining()),
            )
            record["framebuffer"] = {
                key: value for key, value in frame.items() if key != "dataBase64"
            }
            if frame.get("available") and frame.get("dataBase64"):
                frame_path = ARTIFACT_DIR / f"{record['stamp']}_frame.jpg"
                atomic_write(frame_path, base64.b64decode(frame["dataBase64"]))
                record["framebuffer"]["path"] = str(frame_path)
        except BridgeError as exc:
            record["framebuffer"]["error"] = str(exc)
        finally:
            try:
                record["framebuffer_stop"] = session.client.call_tool(
                    "canmv_stop_script", timeout=4
                )
            except BridgeError as exc:
                record["framebuffer_stop_error"] = str(exc)
            try:
                record["preview_stop"] = session.client.call_tool(
                    "canmv_stop_preview", timeout=3
                )
            except BridgeError as exc:
                record["preview_stop_error"] = str(exc)

        record["ok"] = all(item["valid"] for item in record["runs"])
        if not record["ok"]:
            raise BridgeError("Probe output validation failed", "probe_output")
        return record
    finally:
        if session.connected:
            try:
                record["final_safety_stop"] = session.client.call_tool(
                    "canmv_stop_script", timeout=2
                )
            except BridgeError as exc:
                record["final_safety_stop_error"] = str(exc)
        session.close()
        record["backend_log"] = session.backend_log
        record["elapsed_seconds"] = round(time.monotonic() - session.started, 3)
        record["evidence"] = save_record(record, "\n".join(terminal_runs))


def perform_run(path_text: str, timeout: float, wait: float, until: str | None) -> dict[str, Any]:
    path, source = load_script(path_text)
    session = BoardSession(timeout)
    record = base_record("run", session)
    output = ""
    try:
        connected = session.connect()
        record.update(
            {
                "device": session.board,
                "connection": connected,
                "released_serial_holders": session.released_holders,
                "script": str(path),
                "delivery": "in_memory_script_exec",
                "persistent_write": False,
            }
        )
        session.client.call_tool("canmv_stop_script", timeout=4)
        terminal_output(session.client, clear=True)
        record["start"] = session.client.call_tool(
            "canmv_run_script", {"script": source}, timeout=8
        )
        wait_budget = max(0.0, session.client.remaining() - 4.0)
        wait_deadline = time.monotonic() + min(wait, wait_budget)
        while time.monotonic() < wait_deadline:
            output = terminal_output(session.client)
            if until and until in output:
                break
            time.sleep(0.2)
        record["stop"] = session.client.call_tool("canmv_stop_script", timeout=4)
        time.sleep(0.3)
        output = terminal_output(session.client)
        record["output"] = output
        record["until"] = until
        record["until_seen"] = bool(until and until in output)
        record["ok"] = True
        return record
    finally:
        if session.connected:
            try:
                record["final_safety_stop"] = session.client.call_tool(
                    "canmv_stop_script", timeout=2
                )
            except BridgeError as exc:
                record["final_safety_stop_error"] = str(exc)
        session.close()
        record["backend_log"] = session.backend_log
        record["elapsed_seconds"] = round(time.monotonic() - session.started, 3)
        record["evidence"] = save_record(record, output)


def perform_stop(timeout: float) -> dict[str, Any]:
    session = BoardSession(timeout)
    record = base_record("stop", session)
    output = ""
    try:
        record["connection"] = session.connect()
        record["device"] = session.board
        record["released_serial_holders"] = session.released_holders
        terminal_output(session.client, clear=True)
        record["stop"] = session.client.call_tool("canmv_stop_script", timeout=5)
        time.sleep(0.3)
        output = terminal_output(session.client)
        record["output"] = output
        record["ok"] = True
        return record
    finally:
        session.close()
        record["backend_log"] = session.backend_log
        record["elapsed_seconds"] = round(time.monotonic() - session.started, 3)
        record["evidence"] = save_record(record, output)


def perform_deploy(path_text: str, timeout: float) -> dict[str, Any]:
    path, source = load_script(path_text)
    session = BoardSession(timeout)
    record = base_record("deploy", session)
    record["script"] = str(path)
    try:
        record["connection"] = session.connect()
        record["device"] = session.board
        record["released_serial_holders"] = session.released_holders
        capabilities = session.client.call_tool(
            "canmv_board_capabilities", timeout=4
        )
        record["capabilities"] = capabilities
        if not capabilities.get("capabilities", {}).get("writeFile"):
            record.update(
                {
                    "ok": False,
                    "uploaded": False,
                    "reason": (
                        "Connected firmware exposes no remote writeFile capability; "
                        "use run for non-persistent RAM ScriptExec delivery"
                    ),
                }
            )
            raise BridgeError(record["reason"], "unsupported_firmware")
        import hashlib

        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        remote_path = f"/sdcard/codex_bridge_{path.stem}_{digest}.py"
        if remote_path.lower() in FORBIDDEN_REMOTE_PATHS:
            raise BridgeError("Refusing to overwrite a startup script", "safety")
        record["write"] = session.client.call_tool(
            "canmv_write_file",
            {"path": remote_path, "content": source, "encoding": "utf8"},
            timeout=15,
        )
        record["remote_path"] = remote_path
        record["uploaded"] = True
        record["ok"] = True
        return record
    finally:
        session.close()
        record["backend_log"] = session.backend_log
        record["elapsed_seconds"] = round(time.monotonic() - session.started, 3)
        record["evidence"] = save_record(record)


def perform_capture(path_text: str, timeout: float, overwrite: bool) -> dict[str, Any]:
    output_path = Path(path_text).expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise BridgeError(
            f"Output already exists: {output_path}; pass --overwrite to replace it",
            "arguments",
        )
    session = BoardSession(timeout)
    record = base_record("capture", session)
    try:
        record["connection"] = session.connect()
        record["device"] = session.board
        record["released_serial_holders"] = session.released_holders
        record["preview_start"] = session.client.call_tool(
            "canmv_start_preview", timeout=4
        )
        frame = session.client.call_tool(
            "canmv_get_latest_frame",
            {"waitMs": 5000, "fresh": True},
            timeout=min(14, session.client.remaining()),
        )
        if not frame.get("available") or not frame.get("dataBase64"):
            raise BridgeError(
                "No framebuffer was published by the running board script",
                "frame_unavailable",
            )
        atomic_write(output_path, base64.b64decode(frame["dataBase64"]))
        record["frame"] = {
            key: value for key, value in frame.items() if key != "dataBase64"
        }
        record["output"] = str(output_path)
        record["preview_stop"] = session.client.call_tool(
            "canmv_stop_preview", timeout=3
        )
        record["ok"] = True
        return record
    finally:
        session.close()
        record["backend_log"] = session.backend_log
        record["elapsed_seconds"] = round(time.monotonic() - session.started, 3)
        record["evidence"] = save_record(record)


def perform_logs() -> dict[str, Any]:
    if not LAST_SESSION.is_file():
        raise BridgeError("No bridge session log exists yet", "logs_not_found")
    return {
        "ok": True,
        "command": "logs",
        "last_session_path": str(LAST_SESSION),
        "terminal_log_path": str(LAST_TERMINAL),
        "session": json.loads(LAST_SESSION.read_text(encoding="utf-8")),
        "terminal": (
            LAST_TERMINAL.read_text(encoding="utf-8", errors="surrogateescape")
            if LAST_TERMINAL.is_file()
            else ""
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe", help="run the bounded two-pass hardware probe")
    probe.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)

    deploy = subparsers.add_parser(
        "deploy", help="persist a script under a non-startup hashed path when supported"
    )
    deploy.add_argument("script")
    deploy.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)

    run = subparsers.add_parser("run", help="send and run a local script from RAM")
    run.add_argument("script")
    run.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    run.add_argument("--wait", type=float, default=10.0)
    run.add_argument("--until")

    stop = subparsers.add_parser("stop", help="interrupt the current board script")
    stop.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)

    subparsers.add_parser("logs", help="return the last complete saved session log")

    capture = subparsers.add_parser(
        "capture", help="save a framebuffer JPEG published by a running script"
    )
    capture.add_argument("output")
    capture.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    capture.add_argument("--overwrite", action="store_true")
    return parser


def _handle_termination(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGTERM, _handle_termination)
    args = build_parser().parse_args(argv)
    record: dict[str, Any] | None = None
    try:
        if args.command == "probe":
            record = perform_probe(args.timeout)
        elif args.command == "deploy":
            record = perform_deploy(args.script, args.timeout)
        elif args.command == "run":
            if args.wait < 0:
                raise BridgeError("--wait must be non-negative", "arguments")
            record = perform_run(
                args.script, args.timeout, min(args.wait, args.timeout), args.until
            )
        elif args.command == "stop":
            record = perform_stop(args.timeout)
        elif args.command == "logs":
            record = perform_logs()
        elif args.command == "capture":
            record = perform_capture(args.output, args.timeout, args.overwrite)
        else:
            raise BridgeError(f"Unknown command: {args.command}", "arguments")
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0 if record.get("ok") else 1
    except BridgeError as exc:
        error = {
            "ok": False,
            "command": getattr(args, "command", None),
            "error": {"category": exc.category, "message": str(exc)},
        }
        print(json.dumps(error, ensure_ascii=False, indent=2))
        return 2
    except KeyboardInterrupt:
        print(
            json.dumps(
                {
                    "ok": False,
                    "command": getattr(args, "command", None),
                    "error": {"category": "interrupted", "message": "Interrupted"},
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
