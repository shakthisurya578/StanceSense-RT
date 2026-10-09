# StanceSense-RT — Project Context (for Claude Code)

> Agent-oriented context. The human guide is `README.md` (how to run it, what every
> file does). Environment: Windows 11, Python 3.11, CPU only, `.venv` at the project root.

## What it is

One webcam + MediaPipe BlazePose. A seated functional hip-rotation test (IR/ER per leg)
gives a hip profile and a starting squat stance. Then a squat check: a **rule-based**
biomechanical classifier (no machine learning) counts reps, confirms each is a squat
(9 checks, threshold 7/9) and gives NARROW / MODERATE / WIDE stance advice from a
front-facing camera. Results go to SQLite (`data/stancesense.db`) and a Streamlit
dashboard, packaged as a Windows desktop app (`scripts/desktop_app.py`) and in Docker
(dashboard only; the webcam stays on the host).

The project is **rule-based only**. The earlier GRU / MLP verifiers, the FSM rep counter
and the MM-Fit verification were removed on 2026-10-08 and moved to
`..\StanceSense-RT_removed_2026-10-08\` (list in `MOVED_FILES.txt`). Don't reintroduce
them or quote their numbers.

## Map

```
scripts/  desktop_app.py (+ install_desktop_app.ps1, make_app_icon.py)   the desktop app
          app.py            dashboard: Assess / History / Verification / Offline demo
          live_assessment.py camera window (Start launches it as its own process)
          demo_end_to_end_mmfit.py   no-camera demo on MM-Fit
          dedupe_multiview.py -> preprocess_multiview.py -> rebuild_multiview_scale.py
              -> eval_rule_classifier.py (--split dev | eval) -> e2e_rule_classifier.py
          make_tarp_figures.py, capture_report_media.py, build_tarp_report.py,
          tarp_finalize.ps1   TARP report (front-matter details are constants in build_tarp_report.py)
src/stancesense/
  app/       controller.py (guided state machine, UI-free), live_session.py, hud.py, recorder.py
  kinematics/ pose_estimator.py, calibrator.py, rotation.py (shank angle = hip rotation)
  profiling/profiler.py, recommendation/stance.py (+ config/stance_rules.yaml)
  squat/     rule_classifier.py (RuleConfig, RuleSquatRunner, stance_recommendation),
             signals.py (frame_signals, FrameState), kinematics.py
  storage/store.py, ui/report.py, ingestion/video_stream.py
  datasets/  mmfit.py (offline demo), multiview_fitness.py (evaluation data, duplicate rules)
models/rule_classifier/   frozen config, metrics_{dev,eval}.json, results_*.csv, e2e_realtime.json
```

## Facts that are easy to break

- **The only pretrained model** is MediaPipe BlazePose (pose_detection + pose_landmark_full,
  frozen, inside the mediapipe package). Nothing is trained by the project.

- **Evaluation split:** people split once with `default_rng(0)`: 6 development, 20
  evaluation. Rules may only be tuned on development people; the evaluation split is
  scored once with frozen rules. Duplicates (`data/processed_multiview/duplicates.json`,
  47 files) are removed before the split.
- **Mirroring:** pose runs on the raw frame; the image is mirrored only for display
  (mirroring first swaps MediaPipe's left/right).
- **Holds are in seconds, not frames.** Each rotation step has a 4 s countdown; an extreme
  locks within 3 deg of the step's furthest point, held still for 2 s.
- **Camera-window text** uses OpenCV's ASCII-only Hershey font: no em dashes or degree
  signs, and every line must fit 640 px (tested in `test_app_controller.py`).
- **Stance advice** needs a front view (hip line within 20 deg of square-on).
- **Storage schema** keeps a `gru_prob` column so older sessions still load; leave it.
- Rotation is functional and qualitative: never claim angle accuracy or bone version.

## Run / test

```powershell
.venv\Scripts\python.exe -m pytest -q                       # 138 tests, ~10 s
.venv\Scripts\python.exe -m streamlit run scripts/app.py     # dashboard
.venv\Scripts\python.exe scripts/live_assessment.py         # camera window
.venv\Scripts\python.exe scripts/eval_rule_classifier.py --split eval   # writes models/rule_classifier/
```

Long runs: launch detached (PowerShell `Start-Process`); background shells get killed.
Figures and report numbers are generated from saved results; never hand-edit them.
