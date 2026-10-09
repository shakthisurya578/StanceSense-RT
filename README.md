# StanceSense-RT: how to run it, and what every file does

StanceSense-RT uses one webcam to do two things:

1. **Hip-rotation profile.** You sit with your lower legs hanging and turn each foot
   as far in and as far out as it goes. The app measures how far each hip rotates
   inwards (IR) and outwards (ER), and turns that into a starting squat stance
   (how wide, how much toe-out).
2. **Squat check.** You stand up and squat facing the camera. A rule-based
   classifier counts each rep and checks it is a real squat (9 biomechanical checks,
   no machine learning). From the confirmed squats it says whether your stance is
   NARROW, MODERATE or WIDE and whether to change it.

Everything runs on the CPU. Pose comes from Google's MediaPipe BlazePose (33 body
landmarks per frame). Results are saved in a small SQLite database and shown in a
dashboard. Rotation is reported as a **functional, qualitative** measure. It is not
an X-ray measurement of bone version, and no angle-accuracy claim is made.

---

## 1. One-time setup (Windows)

You need **Python 3.11** and a webcam. From the project folder:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Optional, but it's the easiest way to use the app: add the desktop icon.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_desktop_app.ps1
```

This puts a **StanceSense-RT** icon on the Desktop and in the Start menu. Run it
again if you move the project folder. `-Uninstall` removes the icons.

---

## 2. How to run it

### 2.1 The desktop app (normal use)

Double-click **StanceSense-RT**. A splash shows for about 3 seconds, then the
dashboard opens in its own window. Close the window when you are done, and
everything it started stops with it.

In the **Assess** tab, set a subject id and how many squat reps to collect, then
press **Start**. A camera window opens. Follow the text at the top of it:

1. **Calibrate.** Sit with both lower legs hanging, fully in frame.
2. **Each leg, three steps:** foot straight ahead (sets the zero), then turn it as far
   as it goes one way, then the other way. A 4-second **GET READY** countdown comes
   before every step and nothing is measured during it. At each extreme, hold still
   at your limit for 2 seconds. A reading only locks at the furthest point you reached.
3. **Stand up** facing the camera. Rep counting waits until you are upright.
4. **Squat.** The set ends after the target number of confirmed reps.

Keys in the camera window: **ESC** quit, **S** skip a leg, **SPACE** end the squat set
early. The result appears in the dashboard and is saved to `data/stancesense.db`. Every
saved run is listed in the **History** tab. A video of the run is saved in `recordings/`.

### 2.2 The dashboard without the icon

```powershell
.venv\Scripts\python.exe -m streamlit run scripts/app.py
```

It opens in your browser at http://localhost:8501. It works the same as the desktop app.

### 2.3 Only the camera window

```powershell
.venv\Scripts\python.exe scripts/live_assessment.py
.venv\Scripts\python.exe scripts/live_assessment.py --reps 8 --subject alice
.venv\Scripts\python.exe scripts/live_assessment.py --no-squat --countdown 5 --hold 3
```

| Option | Meaning |
|---|---|
| `--subject` | name stored with the session (default `local`) |
| `--reps N` | stop after N confirmed squats, `0` = until SPACE (default 5) |
| `--no-squat` | hip-rotation profile only |
| `--countdown S` | seconds of GET READY before each rotation step (default 4) |
| `--hold S` | seconds to hold still at your limit (default 2) |
| `--no-store` | don't save to the database |
| `--no-record` | don't save a video |
| `--codec mp4v` | smaller .mp4 instead of .avi (may not open in Windows Photos) |

### 2.4 No webcam: the offline demo

Runs the whole chain on a recorded MM-Fit workout (data in `data/mm-fit/`):

```powershell
.venv\Scripts\python.exe scripts/demo_end_to_end_mmfit.py --workout w19 --no-store
```

It prints the profile, the stance and the squat results, and writes
`demo_output/end_to_end_w19.png` and `.txt`. The dashboard's **Offline demo** tab runs
the same thing.

### 2.5 Docker (dashboard only)

```powershell
docker compose up --build -d          # dashboard at http://localhost:8501
docker compose run --rm tools         # run the tests inside the container
docker compose down
```

Docker Desktop on Windows cannot pass a webcam into a Linux container, so **Start** is
turned off in the Docker dashboard. Run the camera on Windows (2.1 or 2.3). It saves to
`data/stancesense.db`, which the container reads, so those runs show up in History.

### 2.6 Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

138 tests, about 10 seconds.

### 2.7 Re-running the evaluation (Verification tab numbers)

The squat classifier was evaluated on the *Multi-View Raw Video Dataset of Seven
Fitness Exercises* (the zip under `data/A Multi-View Raw Video Dataset.../`). The
numbers are already saved in `models/rule_classifier/`. To reproduce them from
scratch, run the steps in order. Step 2 runs MediaPipe on all 1042 videos and takes hours.

```powershell
.venv\Scripts\python.exe scripts/dedupe_multiview.py                     # 1. find duplicate videos
.venv\Scripts\python.exe scripts/preprocess_multiview.py --data "data/A Multi-View Raw Video Dataset of Seven Fitness Ex/Dataset Exercise Quality-wise.zip"   # 2. landmarks
.venv\Scripts\python.exe scripts/rebuild_multiview_scale.py              # 3. one body scale per person
.venv\Scripts\python.exe scripts/eval_rule_classifier.py --split dev     # 4a. the 6 development people
.venv\Scripts\python.exe scripts/eval_rule_classifier.py --split eval    # 4b. the 20 held-out people
.venv\Scripts\python.exe scripts/e2e_rule_classifier.py --clip subject_002_squat_good_front   # 5. raw video -> live chain
```

People are split once with a fixed seed: 6 development people (rules were set while
looking at them) and 20 evaluation people (scored once with the frozen rules). The
saved evaluation result covers 767 videos (109 squat, 658 not squat) from the 20
evaluation people: accuracy 98.0%, precision 94.3%, recall 91.7%, F1 93.0%,
specificity 99.1%, balanced accuracy 95.4% (from `models/rule_classifier/metrics_eval.json`).

### 2.8 Rebuilding the TARP report

```powershell
.venv\Scripts\python.exe scripts/make_tarp_figures.py          # diagrams and result figures
.venv\Scripts\python.exe scripts/capture_report_media.py       # recording stills, dashboard screenshots, pose-model timing
.venv\Scripts\python.exe scripts/build_tarp_report.py          # -> docs/tarp_report/TARP_Report_StanceSense-RT.docx
powershell -ExecutionPolicy Bypass -File scripts\tarp_finalize.ps1   # Word updates contents/lists and saves a PDF
```

The report's numbers are read from the saved results at build time (evaluation files, the live
session in `data/stancesense.db`, `models/pose_speed.json`, the test count from pytest); nothing
is typed in by hand. Close the report in Word before rebuilding it, because Word locks the file.
`build_tarp_report.py some\path.docx` builds a copy elsewhere instead.

---

## 3. How the pieces fit together

```
webcam frame
  -> MediaPipe BlazePose ............ kinematics/pose_estimator.py
  -> AssessmentController ........... app/controller.py   (what step are we on? what should the user do?)
       calibrate: hip width, femur ... kinematics/calibrator.py
       rotation angle of each shank .. kinematics/rotation.py
       4 extremes -> hip profile ..... profiling/profiler.py
       profile -> starting stance .... recommendation/stance.py + config/stance_rules.yaml
       squat reps: count + check ..... squat/rule_classifier.py (signals from squat/signals.py)
       stance advice from the squats . squat/rule_classifier.py (stance_recommendation)
  -> camera window text + bars ...... app/hud.py, driven by app/live_session.py
  -> SQLite database ................ storage/store.py -> data/stancesense.db
  -> dashboard ...................... scripts/app.py (text from ui/report.py)
```

The dashboard does not touch the camera. **Start** launches `scripts/live_assessment.py`
as a separate program that owns the webcam and draws its own window at full frame rate.
The two talk through a small status file (`.streamlit/live_status.json`) and the database.
Streaming every frame into a web page would flicker and drop to a few frames a second.

---

## 4. What each file does

### 4.1 Programs you run (`scripts/`)

**`desktop_app.py`** — the Windows desktop app.
Starts the dashboard server hidden (no console) on `127.0.0.1:8531`, shows a small
Tkinter splash while it loads, then opens the dashboard in Microsoft Edge (or Chrome)
`--app` mode with a private profile in `%LOCALAPPDATA%\StanceSense-RT`. It watches for
that window to close (Windows API through `ctypes`) and then stops the server and the
camera window it started. Uses `subprocess`, `tkinter`, `ctypes`, `urllib`. It exists so
the project opens like a normal app with one double-click.

**`install_desktop_app.ps1`** — creates (or with `-Uninstall`, removes) the Desktop and
Start-menu shortcuts. Uses the Windows Script Host COM object. Each shortcut runs
`desktop_app.py` with `.venv\Scripts\pythonw.exe`, so no console window appears.

**`make_app_icon.py`** — draws the app icon (a squatting figure) with Pillow and saves
`assets/stancesense.ico` (all Windows sizes) and `stancesense.png`. The icon is drawn by
code so it can be regenerated.

**`app.py`** — the dashboard (Streamlit). Four tabs:
- *Assess*: settings, Start/Stop, live progress (read from the status file), then the result.
- *History*: every saved session from the database, plus the IR/ER trend over time.
- *Verification*: the classifier's evaluation, read from `models/rule_classifier/`.
- *Offline demo*: runs `demo_end_to_end_mmfit.py` on a chosen MM-Fit workout.

It imports only the database and formatting code, so the page loads fast. MediaPipe and
OpenCV stay in the camera program.

**`live_assessment.py`** — the camera window. Opens the webcam (`ingestion/video_stream.py`),
feeds every frame with its timestamp to a `LiveSession`, shows the annotated frame with
OpenCV, handles the keys, writes the status file for the dashboard, records the video and
saves the result. Uses OpenCV and MediaPipe. It is a separate program because a native
window keeps the full camera frame rate.

**`demo_end_to_end_mmfit.py`** — the no-camera demo. Takes one MM-Fit workout (3-D pose
already recorded), measures rotation from the quiet-standing frames, builds the profile
and stance, runs every labelled squat set through the rule classifier, and writes a figure
and a text summary (`matplotlib`). It shows the whole chain working without a webcam.
MM-Fit has no seated rotation test, so its rotation numbers are only a demonstration of
the code path.

**`dedupe_multiview.py`** — scans the Multi-View zip for byte-identical videos filed
under two names (some under two different people, or as squat and non-squat). Writes
`data/processed_multiview/duplicates.json`; 47 files are excluded. Needed so the same
recording can't appear in both the development and evaluation people.

**`preprocess_multiview.py`** — runs MediaPipe on every Multi-View video once and caches
each clip's 3-D landmarks, leg visibility, frame rate and a first body-scale estimate as a
`.npz` file under `data/processed_multiview/`, plus a `manifest.csv`. Interrupted runs
resume. It exists so the evaluation never has to run MediaPipe again.

**`rebuild_multiview_scale.py`** — replaces each clip's own body-scale estimate (femur
length, hip width) with one value per person, pooled from that person's well-tracked
frames. Close-up clips cut the legs off, so their own estimate is wrong; a person's femur
doesn't change between clips. Writes `models/multiview_scale_correction.json`.

**`eval_rule_classifier.py`** — the evaluation. Splits the 26 people once with a fixed
seed (6 development, 20 evaluation), runs the classifier on every cached video, and
writes per-video results (`results_*.csv`), metrics (`metrics_*.json`: confusion matrix,
accuracy, precision, recall, F1, specificity, balanced accuracy) and the reason for each
wrong answer. The Verification tab and the report read these files.

**`e2e_rule_classifier.py`** — checks the real-time path on raw video: MediaPipe frame by
frame, `RuleSquatRunner` with timestamps, the way the live app runs. It also counts
knee-angle valleys (`scipy` peak finding) as an independent reference for the rep count.
Writes `models/rule_classifier/e2e_realtime.json`.

**`make_tarp_figures.py`** — draws the report figures (architecture, shank geometry,
state machine, a signal trace, confusion matrix, per-exercise results, stance advice,
pose overlay) from the saved results with matplotlib. Output: `docs/tarp_report/figures/`.

**`capture_report_media.py`** — makes the report's live-assessment media from a real session
(session #10 and its recording in `recordings/`). It cuts stills from the recording at the moments
the app locked a reading or confirmed a squat (OpenCV, Pillow). It screenshots the dashboard
for that session with headless Microsoft Edge driven over the DevTools protocol (a temporary
copy of the dashboard, so the app's own status file is untouched). It also times the pretrained
pose model on a raw Multi-View video and writes `models/pose_speed.json`.

**`build_tarp_report.py`** — writes the TARP report in the VIT template with `python-docx`.
Tables and numbers come from `models/rule_classifier/`, `models/experiments/multiview_A.json`,
`models/pose_speed.json`, the live session in the database and `config/stance_rules.yaml`.
Your name, register number, guide and signing date are set at the top of the script.

**`tarp_finalize.ps1`** — opens the report in Microsoft Word through COM, updates the
contents, figure and table lists, saves, and exports a PDF. If your own Word is already open it
leaves that window alone and doesn't quit it.

### 4.2 The library (`src/stancesense/`)

**`app/controller.py`** — the brain of the guided assessment, with no UI code in it.
Stages: CALIBRATE → for each leg NEUTRAL → EXTREME_A → EXTREME_B → SQUAT → DONE. For
every frame it returns a `UiState` (title, instruction, detail line, progress bar,
countdown, live angle). It refuses to measure a leg whose hip, knee or ankle is not
clearly visible. It runs the 4-second countdown before each rotation step, and locks an
extreme only within 3° of the furthest point reached, held still for 2 seconds. Holds are
timed in seconds, not frames. For the squat part it waits until you stand upright (knee
over 145° for 1 s), then hands frames to the rule classifier and finishes after the
target number of confirmed reps. Kept separate from the window so it can be unit-tested.

**`app/live_session.py`** — `LiveSession` joins the parts for one person: frame → pose →
controller → squat classifier → database. `step(frame, time)` returns the annotated frame;
`save()` writes the run to SQLite; `result()` gives a plain summary for printing.

**`app/hud.py`** — draws on the camera frame with OpenCV: the instruction bar, the hold
bar, the big GET READY countdown, the live angle, and the measured shank line. Pose runs
on the raw frame and the image is mirrored afterwards. Mirroring first would swap
MediaPipe's left and right legs.

**`app/recorder.py`** — saves the annotated video. Motion-JPEG `.avi` by default, because
it opens in any Windows player; after closing it reads the file back to prove it works.

**`kinematics/pose_estimator.py`** — wraps MediaPipe Pose with the settings in
`config/default.yaml`. Returns image landmarks (for drawing) and world landmarks (metres,
for measuring).

**`kinematics/calibrator.py`** — from about 1.5 s of sitting still, records hip width and
femur length (body scale) and each leg's resting shank angle (the zero for rotation).

**`kinematics/rotation.py`** — the core rotation idea, a digital version of the clinical
Craig's test. With the thigh still, rotating the hip swings the lower leg sideways, so the
shank's angle from vertical, seen from the front, is the hip rotation. Positive = internal,
negative = external. The shank is long and well tracked; the hip joint centre is not.

**`profiling/profiler.py`** — turns the four locked extremes into a `HipProfile`: average
IR and ER, total arc, rotation bias (IR minus ER), left/right symmetry, and a pattern
(IR-dominant, balanced or ER-dominant, with a 5° dead zone).

**`recommendation/stance.py`** — turns the profile into a starting stance with plain rules
whose constants live in `config/stance_rules.yaml`. More external-rotation dominance
means more toe-out and a wider stance; a small total arc damps the widening. It flags
left/right asymmetry and writes a sentence explaining the advice.

**`squat/signals.py`** — per-frame body measurements from the landmarks: knee, hip and
ankle angles, trunk lean, stance width, knee width, hip height, depth (0 standing,
1 thighs parallel), foot stagger, left/right drop. Lengths are divided by femur length or
hip width so body size cancels out. Also holds `FrameState`, what the squat runner
reports each frame.

**`squat/rule_classifier.py`** — the squat check. Signals are smoothed (0.3 s median, then
an exponential filter), and a state machine follows STANDING → DESCENDING → BOTTOM →
ASCENDING → STANDING. Each completed cycle gets 9 checks: depth, knee flexion, hip
motion, descent, bottom, ascent, return to standing, both legs bending together, and an
upright enough trunk. A cycle that passes 7 of the 9 (and is physically plausible) counts
as a squat. Every threshold is in `RuleConfig`. `RuleSquatRunner` is the live interface;
`stance_recommendation` gives NARROW / MODERATE / WIDE from the confirmed squats (bands at
1.4 and 2.1 hip widths), and only when the camera sees you from the front (hips within
20° of square-on). It is rule-based because training a GRU on this data gave poor
learning curves (85% of the windows were not squats).

**`squat/kinematics.py`** — single-frame geometry: joint angles, knee flexion, trunk lean,
depth ratio, knee-over-foot deviation, and `detect_stance` (stance width and toe-out from
the feet).

**`storage/store.py`** — the SQLite database (`data/stancesense.db`). Tables: `sessions`,
`hip_profiles`, `stance_recs`, `rep_results`, `squat_runs`. Saves a whole run in one
call and reads it back for the dashboard. SQLite needs no server and the file sits in the
project folder.

**`ui/report.py`** — turns stored results into readable text and tables for the dashboard
and the printouts (profile line, stance line, rep summary, rep table, depth verdict, a
warning when a rotation arc is implausibly large).

**`ingestion/video_stream.py`** — opens the webcam with the camera settings from
`config/default.yaml`, so every program opens the camera the same way.

**`datasets/mmfit.py`** — reads MM-Fit workouts (used by the offline demo), converts their
skeleton to MediaPipe's landmark layout, cuts out the labelled squat sets, and estimates
body scale (`estimate_scale`, also used by the data-preparation scripts).

**`datasets/multiview_fitness.py`** — reads the Multi-View dataset: finds clips in the zip,
parses subject / exercise / good-or-bad / camera view from the file name, extracts a clip,
applies the duplicate rules, and loads the preprocessing manifest.

**`common/types.py`** — the small data classes passed between modules: `HipProfile`,
`StanceRec`, `RepResult`. **`common/geometry.py`** — tiny vector-angle helpers.

### 4.3 Tests (`tests/`)

Run with `.venv\Scripts\python.exe -m pytest -q`. Each file tests one part:

| File | What it checks |
|---|---|
| `test_app_controller.py` | the whole guided flow: countdowns, locking at the real extreme, leg visibility, the stand-up gate, text fits the camera window |
| `test_app_streamlit.py` | the dashboard loads and its buttons work (Streamlit's AppTest) |
| `test_desktop_app.py` | the launcher keeps the server local, finds a free port, has its icon |
| `test_live_mirror.py` | pose runs before the image is mirrored (left/right not swapped) |
| `test_recording.py` | the saved video opens again |
| `test_hud.py` | camera-window text: banners fit the frame, readouts clear the progress bar |
| `test_rule_classifier.py` | real squats pass, shallow dips and standing still don't, frame rate doesn't matter, stance bands |
| `test_signals.py` | body signals point the right way and ignore turning and body size |
| `test_rotation.py` | rotation sign convention and angle maths |
| `test_profiler.py`, `test_stance.py` | profile maths and stance rules |
| `test_squat.py` | stance width / toe-out detection, femur calibration |
| `test_mmfit.py`, `test_multiview_fitness.py` | the two dataset readers and duplicate rules |
| `test_storage.py`, `test_ui_report.py` | database round trips and text formatting |
| `test_video_stream.py` | opening the camera: a clear error when it is missing, release on failure, reading frames |

---

## 5. Folders and data files

| Path | What it is |
|---|---|
| `config/default.yaml` | camera (source, size, fps) and MediaPipe settings |
| `config/stance_rules.yaml` | constants of the starting-stance rules |
| `assets/` | app icon (`.ico` for Windows, `.png` for the dashboard) |
| `data/stancesense.db` | **your saved assessments** (History tab) |
| `data/mm-fit/` | MM-Fit workouts for the offline demo |
| `data/A Multi-View Raw Video Dataset.../` | the evaluation dataset (zip) |
| `data/processed_multiview/` | cached MediaPipe landmarks per video, `manifest.csv`, `duplicates.json` |
| `data/*.md`, `subjects.csv`, `goniometer.csv` | consent / data-collection forms |
| `models/rule_classifier/` | evaluation results: frozen rule config, metrics, per-video CSVs, real-time check |
| `models/multiview_scale_correction.json` | per-person body scale, from `rebuild_multiview_scale.py` |
| `models/experiments/multiview_A.json` | kept only because the report quotes the dataset's class balance from it |
| `models/pose_speed.json` | measured speed of the pretrained pose model on this laptop (for the report) |
| `demo_output/` | offline-demo figures and summaries |
| `recordings/` | videos of your live assessments |
| `docs/tarp_report/` | the final TARP report (.docx, .pdf, figures); `previous_version/` keeps the copy before the 8 Oct update |
| `docs/DA3_*`, `docs/tarp_submissions/` | earlier submitted reports and reviews |
| `.streamlit/live_status.json` | status the camera window writes for the dashboard |
| `Dockerfile`, `docker-compose.yml`, `requirements-docker.txt` | Docker setup (dashboard + tools) |
| `requirements.txt` | Python packages |
| `.claude/launch.json` | dev-server entry for the Claude Code preview (port 8517) |

---

## 6. Removed in the October 2026 clean-up

The earlier GRU and MLP models, their training and evaluation scripts, the old FSM rep
counter, MM-Fit rep-count verification, older live tools and report builders, their saved
results and figures, and the old documentation (`PROJECT_INFO.md`,
`METHODOLOGY_UPDATE.md`, `MMFIT_VERIFICATION.md`) were moved out of this folder. They
are in `..\StanceSense-RT_removed_2026-10-08\`, at the same relative paths, with a list
in `MOVED_FILES.txt`. Nothing was deleted. Once you are sure nothing there is needed,
you can delete that folder.

---

## 7. If something goes wrong

- **"Could not open camera source 0".** Another program has the webcam, or it's a
  different camera index. Close Teams/Zoom/the Camera app, or set `camera.source` in
  `config/default.yaml` (1, 2, ...).
- **The app window doesn't appear.** Look in `%LOCALAPPDATA%\StanceSense-RT\launcher.log`
  and `dashboard.log`.
- **A leg never locks.** The border turns amber and says NOT MEASURING when the hip,
  knee or ankle isn't visible. Move back so both legs are fully in frame, in good light.
- **`ml_dtypes` / `jax` error on import.** `.venv\Scripts\python.exe -m pip uninstall -y jax jaxlib`
  (MediaPipe's pose estimation doesn't use them).
- **Start is greyed out in Docker.** Expected. Run the camera on Windows (section 2.5).
