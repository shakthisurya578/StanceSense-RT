"""StanceSense-RT as a Windows desktop app.

Double-click the StanceSense-RT icon (created by scripts/install_desktop_app.ps1),
or run it yourself:

    .venv/Scripts/pythonw.exe scripts/desktop_app.py

What it does:
  1. starts the dashboard server (Streamlit) in the background, with no console
     window, listening on this computer only (127.0.0.1);
  2. shows a small splash while the server loads;
  3. opens the dashboard in its own app window (Microsoft Edge, or Chrome, in
     --app mode: no tabs, no address bar) using a private browser profile, so it
     never touches your normal browser windows;
  4. stops the server - and a camera window it started - once that app window is closed.

Pressing Start in the dashboard still opens the native camera window, exactly as
before. Launching the app while it is already open just opens another window.
Logs: %LOCALAPPDATA%/StanceSense-RT/dashboard.log and launcher.log.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITLE = "StanceSense-RT"
DEFAULT_PORT = 8531          # not 8501 (Docker dashboard) or 8517 (dev preview)
APP_DIR = Path(os.environ.get("LOCALAPPDATA") or ROOT) / "StanceSense-RT"
PROFILE = APP_DIR / "browser-profile"
ICON_PNG = ROOT / "assets" / "stancesense.png"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_SYS32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
# Full paths: PATH is not guaranteed to include them when started from a shortcut.
POWERSHELL = str(_SYS32 / "WindowsPowerShell" / "v1.0" / "powershell.exe")
TASKKILL = str(_SYS32 / "taskkill.exe")


# ---------------------------------------------------------------------- helpers

def log(msg: str):
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        with open(APP_DIR / "launcher.log", "a", encoding="utf-8") as fh:
            fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    except OSError:
        pass


def message_box(text: str, error: bool = True):
    log(("ERROR: " if error else "") + text)
    try:
        ctypes.windll.user32.MessageBoxW(0, text, TITLE, 0x10 if error else 0x40)
    except Exception:
        print(text, file=sys.stderr)


def healthy(port: int, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health",
                                    timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def powershell(script: str) -> str:
    try:
        out = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script],
                             capture_output=True, text=True, timeout=30, creationflags=NO_WINDOW)
        return out.stdout
    except Exception as exc:
        log(f"powershell failed: {exc}")
        return ""


def _pids(text: str) -> set:
    return {int(t) for t in text.split() if t.strip().isdigit()}


def profile_browser_pids() -> set:
    """Browser processes running our private profile (their command line names it)."""
    prof = str(PROFILE).replace("'", "''")
    return _pids(powershell(
        "Get-CimInstance Win32_Process -Filter \"Name='msedge.exe' or Name='chrome.exe'\" | "
        f"Where-Object {{ $_.CommandLine -and $_.CommandLine.Contains('{prof}') }} | "
        "ForEach-Object { $_.ProcessId }"))


def pid_listening_on(port: int) -> set:
    return _pids(powershell(
        f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | "
        "ForEach-Object { $_.OwningProcess }"))


def kill_tree(pid: int):
    subprocess.run([TASKKILL, "/PID", str(pid), "/T", "/F"], capture_output=True,
                   creationflags=NO_WINDOW)


def visible_windows(pids: set) -> list:
    """Titles of visible top-level windows owned by these processes."""
    if not pids:
        return []
    user32 = ctypes.windll.user32
    titles = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            pid = wt.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value in pids:
                n = user32.GetWindowTextLengthW(hwnd)
                if n:
                    buf = ctypes.create_unicode_buffer(n + 1)
                    user32.GetWindowTextW(hwnd, buf, n + 1)
                    titles.append(buf.value)
        return True

    user32.EnumWindows(cb, 0)
    return titles


def find_browser():
    env = os.environ
    roots = [env.get("ProgramFiles(x86)"), env.get("ProgramFiles"), env.get("LOCALAPPDATA")]
    for rel in (r"Microsoft\Edge\Application\msedge.exe", r"Google\Chrome\Application\chrome.exe"):
        for root in roots:
            if root and (Path(root) / rel).exists():
                return str(Path(root) / rel)
    return None


# ------------------------------------------------------------------- the server

def server_python() -> str:
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv.exists():
        return str(venv)
    exe = Path(sys.executable)
    return str(exe.with_name("python.exe")) if exe.name.lower() == "pythonw.exe" else str(exe)


def server_command(port: int) -> list:
    """Streamlit on this PC only (127.0.0.1), no auto-opened browser, no telemetry."""
    return [server_python(), "-m", "streamlit", "run", "scripts/app.py",
            "--server.port", str(port), "--server.address", "127.0.0.1",
            "--server.headless", "true", "--browser.gatherUsageStats", "false",
            "--client.toolbarMode", "minimal"]


def start_server(port: int) -> subprocess.Popen:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
           "TF_CPP_MIN_LOG_LEVEL": "3", "STANCESENSE_DESKTOP": "1"}
    cmd = server_command(port)
    logfile = open(APP_DIR / "dashboard.log", "w", encoding="utf-8")
    log(f"starting server: {' '.join(cmd)}")
    # The camera window the dashboard launches inherits this hidden console and
    # this log file, so its printout lands in dashboard.log too.
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=logfile,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            creationflags=NO_WINDOW)


def wait_for_server(port: int, server, timeout: float = 90.0) -> bool:
    """Poll the health endpoint behind a small splash window."""
    deadline = time.monotonic() + timeout

    def ready():
        return healthy(port, timeout=0.5)

    def gave_up():
        return time.monotonic() > deadline or (server is not None and server.poll() is not None)

    try:
        import tkinter as tk
    except ImportError:
        while not ready():
            if gave_up():
                return False
            time.sleep(0.5)
        return True

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    result = {"ok": False}
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    w, h = 420, 170
    x = (root.winfo_screenwidth() - w) // 2
    y = (root.winfo_screenheight() - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")
    root.configure(bg="#0F5257")
    try:
        img = tk.PhotoImage(file=str(ICON_PNG)).subsample(3, 3)
        tk.Label(root, image=img, bg="#0F5257").place(x=24, y=42)
        root._img = img
    except Exception:
        pass
    tk.Label(root, text=TITLE, fg="white", bg="#0F5257",
             font=("Segoe UI Semibold", 18)).place(x=130, y=46)
    status = tk.Label(root, text="Starting the dashboard...", fg="#BFE8C0", bg="#0F5257",
                      font=("Segoe UI", 10))
    status.place(x=132, y=88)
    started = time.monotonic()

    def poll():
        if ready():
            result["ok"] = True
            root.destroy()
        elif gave_up():
            root.destroy()
        else:
            status.config(text=f"Starting the dashboard...  {time.monotonic() - started:.0f} s")
            root.after(400, poll)

    root.after(300, poll)
    root.mainloop()
    return result["ok"]


# ------------------------------------------------------------------- the window

def open_window(browser: str, url: str) -> subprocess.Popen:
    PROFILE.mkdir(parents=True, exist_ok=True)
    args = [browser, f"--app={url}", f"--user-data-dir={PROFILE}",
            "--no-first-run", "--no-default-browser-check", "--disable-sync",
            "--start-maximized", "--disable-features=Translate"]
    log(f"opening window: {' '.join(args)}")
    return subprocess.Popen(args)


def wait_until_closed(window: subprocess.Popen, server, poll: float = 1.0):
    """Return once every app window of our profile has closed (or the server died)."""
    time.sleep(2.0)
    pids = {window.pid} if window.poll() is None else profile_browser_pids()
    seen, missing, started = False, 0, time.monotonic()
    while True:
        if server is not None and server.poll() is not None:
            log("server exited on its own")
            return "server"
        if window.poll() is not None and window.pid in pids:
            pids.discard(window.pid)
            pids |= profile_browser_pids()
        titles = visible_windows(pids)
        if titles:
            seen, missing = True, 0
        else:
            missing += 1
        if seen and missing >= 3:
            return "closed"
        if not seen and time.monotonic() - started > 60:
            log("no app window appeared within 60 s")
            return "no-window"
        time.sleep(poll)


# ------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--server-only", action="store_true",
                    help="start the server and wait; do not open a window")
    args = ap.parse_args()
    os.chdir(ROOT)

    browser = find_browser()
    if browser is None and not args.server_only:
        message_box("Microsoft Edge or Google Chrome is needed to show the dashboard window, "
                    "and neither was found.")
        return 2

    port, server, adopted = args.port, None, set()
    if healthy(port):
        if profile_browser_pids():
            # Already open: another window shares the running server and the
            # instance that owns it shuts it down when the last window closes.
            log("already running - opening another window")
            open_window(browser, f"http://127.0.0.1:{port}")
            return 0
        adopted = pid_listening_on(port)              # left over from a crashed launcher
        log(f"adopting the running server (pid {adopted})")
    else:
        while not port_free(port):
            port += 1
            if port > args.port + 20:
                message_box("No free local port for the dashboard.")
                return 2
        server = start_server(port)
        if not wait_for_server(port, server):
            if server.poll() is None:
                kill_tree(server.pid)
            message_box("The dashboard did not start. Details are in\n"
                        f"{APP_DIR / 'dashboard.log'}")
            return 1

    url = f"http://127.0.0.1:{port}"
    try:
        if args.server_only:
            log(f"server-only mode at {url}")
            while server is None or server.poll() is None:
                time.sleep(1.0)
            return 0
        why = wait_until_closed(open_window(browser, url), server)
        log(f"window loop ended: {why}")
        if why == "server":
            message_box("The dashboard stopped unexpectedly. Details are in\n"
                        f"{APP_DIR / 'dashboard.log'}")
    finally:
        for pid in profile_browser_pids():           # background leftovers of our profile
            kill_tree(pid)
        if server is not None and server.poll() is None:
            kill_tree(server.pid)
        for pid in adopted:
            kill_tree(pid)
        log("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
