"""Media for the TARP report's live-assessment section, from a real session.

    .venv/Scripts/python.exe scripts/capture_report_media.py
    .venv/Scripts/python.exe scripts/capture_report_media.py --only stills

Three parts (all by default; ``--only`` picks one):

  stills     frames of a recorded assessment (recordings/*.avi) at the moments the
             app locked a reading or confirmed a squat -> fig_live_seated.png,
             fig_live_squat.png. Panel labels take their angles from the database.
  dashboard  screenshots of the dashboard for the same session (headless Microsoft
             Edge driven over the DevTools protocol) -> dash_result_profile.png,
             dash_result_squat.png, dashboard_history.png, dashboard_verification.png
  speed      how long the pretrained pose model takes per frame on this CPU, timed
             on the recording's 640x480 frames -> models/pose_speed.json

Images go to docs/tarp_report/figures/. Nothing in the project is modified apart
from those outputs: the dashboard used for the screenshots runs from a temporary
folder so the app's own status file is left alone.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from stancesense.storage import Store  # noqa: E402

FIG = os.path.join(ROOT, "docs", "tarp_report", "figures")
DB = os.path.join(ROOT, "data", "stancesense.db")
RECORDING = os.path.join(ROOT, "recordings", "assessment_20261007_195940.avi")
SESSION = 10            # the stored session that recording belongs to

# Frames of RECORDING where the app showed each moment (found from its "LOCKED" banners).
SEATED = [(700, "calibration"), (750, "countdown"), (957, "ir_left"),
          (1087, "er_left"), (1345, "ir_right"), (1829, "er_right")]
SQUAT = [(2188, "standing"), (2562, "descent"), (2584, "rep4"), (2658, "rep5")]
SPEED_CLIP = "subject_002_squat_good_front"   # raw Multi-View video used to time the pose model
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
MINUS = chr(0x2212)


# ------------------------------------------------------------------ stills

def _font(size):
    for name in ("times.ttf", "Times New Roman.ttf", "DejaVuSerif.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _read_frames(path, wanted):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    got, i = {}, 0
    last = max(wanted)
    while i <= last:
        ok, f = cap.read()
        if not ok:
            break
        if i in wanted:
            got[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    missing = set(wanted) - set(got)
    if missing:
        raise SystemExit(f"frames {sorted(missing)} not in {path} ({i} frames)")
    return got


def _grid(panels, cols, out, scale=1.0):
    """panels: [(rgb frame, label)] -> one image with a caption under each panel."""
    h, w = panels[0][0].shape[:2]
    w, h = int(w * scale), int(h * scale)
    pad, label_h = 14, 62
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * (h + label_h) + (rows + 1) * pad), "white")
    draw, font = ImageDraw.Draw(canvas), _font(34)
    for k, (img, label) in enumerate(panels):
        r, c = divmod(k, cols)
        x, y = pad + c * (w + pad), pad + r * (h + label_h + pad)
        canvas.paste(Image.fromarray(img).resize((w, h), Image.LANCZOS), (x, y))
        tw = draw.textlength(label, font=font)
        draw.text((x + (w - tw) / 2, y + h + 10), label, fill="black", font=font)
    canvas.save(out)
    print("->", os.path.relpath(out, ROOT))


def stills():
    with Store(DB) as s:
        p, summ = s.profile_for(SESSION), s.rep_summary(SESSION)
    if p is None:
        raise SystemExit(f"session #{SESSION} has no hip profile in {DB}")
    frames = _read_frames(RECORDING, {i for i, _ in SEATED + SQUAT})
    deg = "\u00b0"
    def signed(v):                                   # typographic minus sign
        return f"{v:+.0f}".replace("-", MINUS)

    seated_labels = {
        "calibration": "(a) Calibration: body scale captured",
        "countdown": "(b) 4 s GET READY countdown before a step",
        "ir_left": f"(c) Left leg, internal rotation locked ({signed(p['ir_left'])}{deg})",
        "er_left": f"(d) Left leg, external rotation locked ({signed(-p['er_left'])}{deg})",
        "ir_right": f"(e) Right leg, internal rotation locked ({signed(p['ir_right'])}{deg})",
        "er_right": f"(f) Right leg, external rotation locked ({signed(-p['er_right'])}{deg})",
    }
    _grid([(frames[i], seated_labels[k]) for i, k in SEATED], 3, os.path.join(FIG, "fig_live_seated.png"))
    n = summ["confirmed"]
    squat_labels = {
        "standing": "(a) Upright detected: counting starts",
        "descent": "(b) Descending into the fourth squat",
        "rep4": "(c) Rep 4 confirmed by the rules",
        "rep5": f"(d) Rep {n} confirmed: set complete ({n}/{summ['counted']})",
    }
    _grid([(frames[i], squat_labels[k]) for i, k in SQUAT], 2, os.path.join(FIG, "fig_live_squat.png"))


# --------------------------------------------------------------- dashboard

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http(url, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                return r.read()
        except Exception:
            time.sleep(0.5)
    raise SystemExit(f"timed out waiting for {url}")


def _streamlit(cwd, port, env_extra):
    env = {**os.environ, "PYTHONUTF8": "1", "TF_CPP_MIN_LOG_LEVEL": "3", **env_extra}
    cmd = [sys.executable, "-m", "streamlit", "run", os.path.join(ROOT, "scripts", "app.py"),
           "--server.port", str(port), "--server.address", "127.0.0.1", "--server.headless", "true",
           "--browser.gatherUsageStats", "false", "--theme.base", "light",
           "--client.toolbarMode", "minimal"]
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=NO_WINDOW)
    _wait_http(f"http://127.0.0.1:{port}/_stcore/health")
    return proc


def _browser():
    env = os.environ
    for root in (env.get("ProgramFiles(x86)"), env.get("ProgramFiles"), env.get("LOCALAPPDATA")):
        for rel in (r"Microsoft\Edge\Application\msedge.exe", r"Google\Chrome\Application\chrome.exe"):
            if root and os.path.exists(os.path.join(root, rel)):
                return os.path.join(root, rel)
    raise SystemExit("Microsoft Edge or Google Chrome is needed for the screenshots")


class CDP:
    """Minimal Chrome DevTools Protocol client over one page's websocket."""

    def __init__(self, ws):
        self.ws, self.n = ws, 0

    @classmethod
    async def connect(cls, port):
        from tornado.websocket import websocket_connect
        targets = json.loads(_wait_http(f"http://127.0.0.1:{port}/json"))
        page = next(t for t in targets if t.get("type") == "page")
        return cls(await websocket_connect(page["webSocketDebuggerUrl"], max_message_size=200 * 2 ** 20))

    async def call(self, method, **params):
        self.n += 1
        mid = self.n
        await self.ws.write_message(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.read_message())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    async def wait_text(self, text, timeout=60):
        end = time.time() + timeout
        while time.time() < end:
            if await self.js(f"document.body && document.body.innerText.includes({json.dumps(text)})"):
                return
            await asyncio.sleep(0.5)
        raise SystemExit(f"page never showed {text!r}")

    async def settle(self, seconds=2.5):
        """Let Streamlit finish its reruns and charts their animation."""
        await asyncio.sleep(seconds)

    async def shot(self, out, x, y, w, h, scale):
        r = await self.call("Page.captureScreenshot", format="png", captureBeyondViewport=True,
                            clip=dict(x=x, y=y, width=w, height=h, scale=1))
        import base64
        with open(out, "wb") as fh:
            fh.write(base64.b64decode(r["data"]))
        _trim_bottom(out)
        print("->", os.path.relpath(out, ROOT))


# JS: document-space box spanning from one heading's block to just before another text, inside a column
_BOX_JS = r"""
(() => {
  const all = [...document.querySelectorAll('h1,h2,h3,h4,h5,p,div[data-testid="stMarkdownContainer"],[data-testid="stMetric"],[data-testid="stCaptionContainer"],[data-testid="stAlert"],[data-testid="stDataFrame"],[data-testid="stTable"],[data-testid="stVegaLiteChart"],[data-testid="stArrowVegaLiteChart"]')];
  const vis = el => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  const find = t => all.find(e => vis(e) && e.innerText && e.innerText.trim().startsWith(t));
  const a = find(START), b = END ? find(END) : null;
  if (!a) return null;
  let col = a.closest('[data-testid="column"],[data-testid="stColumn"],[data-testid="stVerticalBlock"]');
  const cr = (col || a).getBoundingClientRect(), ar = a.getBoundingClientRect();
  let bottom;
  if (b) { bottom = b.getBoundingClientRect().top; }
  else {
    bottom = ar.bottom;
    for (const e of all) {
      if (!vis(e) || !(col ? col.contains(e) : true)) continue;
      const r = e.getBoundingClientRect();
      if (r.top >= ar.top && r.bottom > bottom) bottom = r.bottom;
    }
  }
  return {x: cr.left + scrollX, y: ar.top + scrollY, w: cr.width, h: bottom - ar.top};
})()
"""


async def _box(cdp, start, end=None, pad=10):
    js = _BOX_JS.replace("START", json.dumps(start)).replace("END", json.dumps(end) if end else "null")
    b = await cdp.js(js)
    if not b:
        raise SystemExit(f"could not find {start!r} on the page")
    # stop short of the END element; trailing white space is trimmed after the capture
    return b["x"] - pad, max(0, b["y"] - pad), b["w"] + 2 * pad, b["h"] + pad - (6 if end else -pad)


def _trim_bottom(path, margin=24):
    """Cut trailing white rows (Streamlit pads the end of a tab)."""
    img = Image.open(path).convert("RGB")
    a = np.asarray(img)
    rows = np.where((a < 245).any(axis=(1, 2)))[0]
    if len(rows):
        img.crop((0, 0, img.width, min(img.height, rows[-1] + margin))).save(path)


async def _dashboard(edge_port, assess_url, main_url):
    cdp = await CDP.connect(edge_port)
    # The Assess result sits in the left 3/5 of the page; 1600 px keeps its four metric
    # tiles wide enough that "IR-dominant" is not cut. Full-width tabs use 1100 px so their
    # text stays readable at the report's 14.5 cm text width.
    await cdp.call("Emulation.setDeviceMetricsOverride", width=1600, height=4200,
                   deviceScaleFactor=2, mobile=False)
    await cdp.call("Page.enable")

    # Assess tab, after a finished run: the result of SESSION
    await cdp.call("Page.navigate", url=assess_url)
    await cdp.wait_text("Your hip profile")
    await cdp.settle()
    x, y, w, h = await _box(cdp, "Assessment complete", "Squat check")
    await cdp.shot(os.path.join(FIG, "dash_result_profile.png"), x, y, w, h, 2)
    x, y, w, h = await _box(cdp, "Squat check", "Saved as session")
    await cdp.shot(os.path.join(FIG, "dash_result_squat.png"), x, y, w, h, 2)

    # History and Verification tabs
    await cdp.call("Emulation.setDeviceMetricsOverride", width=1100, height=4200,
                   deviceScaleFactor=2, mobile=False)
    await cdp.call("Page.navigate", url=main_url)
    await cdp.wait_text("Live assessment")
    for tab, ready, start, end, name in (
            ("History", "Stored assessments", "Stored assessments", "Squat run", "dashboard_history.png"),
            ("Verification", "How well does it work?", "How well does it work?", None,
             "dashboard_verification.png")):
        await cdp.js("[...document.querySelectorAll('button[role=\"tab\"]')]"
                     f".find(b => b.innerText.trim() === {json.dumps(tab)}).click()")
        await cdp.wait_text(ready)
        await cdp.settle()
        x, y, w, h = await _box(cdp, start, end)
        await cdp.shot(os.path.join(FIG, name), x, y, w, min(h, 1500), 2)
    cdp.ws.close()


def dashboard():
    with Store(DB) as s:
        if s.profile_for(SESSION) is None:
            raise SystemExit(f"session #{SESSION} not in {DB}")
    tmp = tempfile.mkdtemp(prefix="stancesense_shots_")
    procs = []
    try:
        # Assess tab reads its last result from .streamlit/live_status.json in the working
        # folder; a temporary folder gets a "finished" status for SESSION.
        os.makedirs(os.path.join(tmp, ".streamlit"))
        with open(os.path.join(tmp, ".streamlit", "live_status.json"), "w") as fh:
            json.dump(dict(stage="DONE", cue="Assessment complete", progress=1.0, done=True,
                           session_id=SESSION), fh)
        p1, p2 = _free_port(), _free_port()
        procs.append(_streamlit(tmp, p1, {"STANCESENSE_DB": DB}))
        procs.append(_streamlit(ROOT, p2, {}))
        edge_port = _free_port()
        procs.append(subprocess.Popen(
            [_browser(), "--headless=new", f"--remote-debugging-port={edge_port}",
             f"--user-data-dir={os.path.join(tmp, 'browser')}", "--no-first-run",
             "--hide-scrollbars", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW))
        asyncio.run(_dashboard(edge_port, f"http://127.0.0.1:{p1}", f"http://127.0.0.1:{p2}"))
    finally:
        for p in procs:
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True,
                           creationflags=NO_WINDOW)
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------- speed

def _cpu_name():
    try:
        out = subprocess.run(
            [os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "WindowsPowerShell",
                          "v1.0", "powershell.exe"), "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"],
            capture_output=True, text=True, timeout=30, creationflags=NO_WINDOW)
        return out.stdout.strip() or platform.processor()
    except Exception:
        return platform.processor()


def speed(n_frames=300, warmup=30):
    import mediapipe as mp
    import yaml
    from stancesense.kinematics.pose_estimator import PoseEstimator

    import glob
    from stancesense.datasets.multiview_fitness import discover_clips, extract_clip

    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "default.yaml")))["pose"]
    os.chdir(ROOT)
    pose = PoseEstimator(os.path.join("config", "default.yaml"))
    # Raw camera footage is needed: the app's own recordings have the HUD drawn over the
    # head, which the BlazePose detector relies on, so they would time a failing detector.
    zips = glob.glob(os.path.join(ROOT, "data", "A Multi-View Raw Video Dataset*", "*.zip"))
    if not zips:
        raise SystemExit("the Multi-View dataset zip is needed for the timing")
    clip = next(c for c in discover_clips(zips[0]) if c.clip_id == SPEED_CLIP)
    tmp = tempfile.mkdtemp(prefix="stancesense_speed_")
    try:
        cap = cv2.VideoCapture(extract_clip(clip, tmp))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            h, w = f.shape[:2]
            k = 640 / max(h, w)                       # webcam-sized: 640 px on the long side
            frames.append(cv2.cvtColor(cv2.resize(f, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA),
                                       cv2.COLOR_BGR2RGB))
        cap.release()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    frames = frames[:n_frames + warmup]
    times, found = [], 0
    for k, f in enumerate(frames):
        t0 = time.perf_counter()
        _, world = pose.infer(f)
        dt = (time.perf_counter() - t0) * 1000
        if k >= warmup:
            times.append(dt)
            found += world is not None
    files = {}
    for sub, name in (("pose_detection", "pose_detection.tflite"), ("pose_landmark", "pose_landmark_full.tflite")):
        path = os.path.join(os.path.dirname(mp.__file__), "modules", sub, name)
        files[name] = round(os.path.getsize(path) / 1e6, 2) if os.path.exists(path) else None
    out = dict(frames_timed=len(times), frame_size=list(frames[0].shape[1::-1]),
               median_ms=round(statistics.median(times), 1), mean_ms=round(statistics.mean(times), 1),
               p90_ms=round(float(np.percentile(times, 90)), 1),
               fps_at_median=round(1000 / statistics.median(times), 1),
               pose_found_frac=round(found / len(times), 3),
               model_complexity=cfg["model_complexity"], mediapipe=mp.__version__,
               model_files_mb=files, cpu=_cpu_name(), source=f"Multi-View clip {SPEED_CLIP}")
    with open(os.path.join(ROOT, "models", "pose_speed.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("-> models/pose_speed.json", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["stills", "dashboard", "speed"])
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    for name, fn in (("stills", stills), ("speed", speed), ("dashboard", dashboard)):
        if args.only in (None, name):
            fn()


if __name__ == "__main__":
    main()
