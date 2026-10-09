"""The Windows desktop launcher (scripts/desktop_app.py).

Only the pieces that can be checked without opening windows or starting a server:
the server must stay local, the port probe must notice a taken port, and the
window/process helpers must not crash on empty input.
"""
import socket
import sys
from pathlib import Path

import pytest

if sys.platform != "win32":
    pytest.skip("Windows-only launcher", allow_module_level=True)

sys.path.insert(0, "scripts")
import desktop_app as da  # noqa: E402


def test_server_listens_on_this_pc_only():
    cmd = da.server_command(8531)
    assert cmd[cmd.index("--server.address") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--server.port") + 1] == "8531"
    assert "scripts/app.py" in cmd
    assert cmd[cmd.index("--browser.gatherUsageStats") + 1] == "false"


def test_server_uses_the_project_venv_python():
    py = Path(da.server_python())
    assert py.name.lower() == "python.exe"            # never pythonw: Streamlit needs stdio
    if (da.ROOT / ".venv").exists():
        assert py == da.ROOT / ".venv" / "Scripts" / "python.exe"


def test_port_probe_sees_a_taken_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        assert da.port_free(s.getsockname()[1]) is False


def test_pid_parsing_ignores_noise():
    assert da._pids("123\r\n 456 \nabc\n\n") == {123, 456}
    assert da._pids("") == set()


def test_window_and_browser_helpers():
    assert da.visible_windows(set()) == []
    b = da.find_browser()
    assert b is None or Path(b).exists()
    assert Path(da.POWERSHELL).name.lower() == "powershell.exe"


def test_icon_exists_for_the_shortcut():
    assert (da.ROOT / "assets" / "stancesense.ico").exists()
    assert da.ICON_PNG.exists()
