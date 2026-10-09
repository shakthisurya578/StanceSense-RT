# Data & Verification Protocol — StanceSense-RT

> **Change (Aug 2026).** Recording our own squat clips is not currently possible,
> so the squat rep/depth engine is now **verified on the public MM-Fit dataset**
> instead of self-recorded video. The earlier goniometer accuracy study
> (Study A) is **dropped** — StanceSense-RT reports functional hip rotation
> *qualitatively* and no longer claims a validated rotation-angle accuracy.

## 1. What is verified, and with what data

| Item | Purpose | Source |
|---|---|---|
| **Squat rep counting** | Does the FSM count reps correctly? | **MM-Fit** per-set rep labels |
| **Squat recognition** | Does the GRU tell a squat from other movement? | **MM-Fit** exercise labels |
| ~~Rotation vs goniometer~~ | ~~Accuracy of the app's rotation angle~~ | **Dropped** (no ground truth) |

The seated hip-rotation measurement still drives the stance recommendation; it is
simply presented as a functional signal, not an accuracy-validated angle.

## 2. MM-Fit dataset

- **Get it:** download `mm-fit.zip` from https://mmfit.github.io/ and extract into
  `data/mm-fit/` so the layout is `data/mm-fit/w00/…`, `data/mm-fit/w01/…`, etc.
  (21 workouts, w00–w20). If the archive unpacks into a nested `data/mm-fit/mm-fit/`,
  move the `wNN` folders up one level. **All 21 are present and used** — the
  results in `MMFIT_VERIFICATION.md` cover the complete release.
- **What we use:** per workout, `wNN_pose_3d.npy` (3-D pose, 17 Human3.6M joints)
  and `wNN_labels.csv` (`start_frame, end_frame, reps, exercise`). Squat rows are
  labelled `squats`. Other modalities (IMU, RGB-D) are not needed.
- **Adapter:** `src/stancesense/datasets/mmfit.py` maps the MM-Fit skeleton onto
  the MediaPipe joints the squat kinematics expect (+z-up → +y-down), so the same
  geometry the live pipeline runs is exercised on real data.

## 3. Reproduce the verification

```bash
# 1) rep/depth engine vs ground-truth rep counts (no training at all)
python scripts/verify_squats_mmfit.py --data data/mm-fit --mode adaptive
python scripts/verify_squats_mmfit.py --data data/mm-fit --mode fixed        --out data/labeled/mmfit_squat_verification_fixed.csv

# 2) train + evaluate the GRU squat verifier on held-out subjects
python scripts/train_squat_gru.py --data data/mm-fit --backend keras --epochs 40

# 3) leave-one-subject-out across all 21 workouts (the headline metric)
python scripts/train_squat_gru.py --data data/mm-fit --loso --backend keras --epochs 40

# 3b) what the verifier flags, per exercise (does it confuse lunges for squats?)
python scripts/analyze_verifier_by_exercise.py

# 4) the whole chain on one workout, no camera needed
python scripts/demo_end_to_end_mmfit.py --workout w04
```

Results are written to `data/labeled/` (FSM per-set CSV), `models/` (GRU metrics,
weights, and the feature-normalisation constants inference reuses),
`data/stancesense.db` (the SQLite store the dashboard reads), and `demo_output/`.
See `MMFIT_VERIFICATION.md` for the current numbers. All runs are CPU-only and
seeded (`--seed`, default 0).

## 4. Subject split

The active GRU (Multi-View model) is evaluated **person-independently**:
leave-one-person-out over 25 people, with duplicate recordings removed first, so
no person, clip or window is in both training and test.

The retired MM-Fit GRU was evaluated **leave-one-workout-out** (no workout in both
training and test). That is NOT person-independent: MM-Fit's 21 workouts come from
only 10 people. The FSM rep counter is threshold-based and uses no training.

## 5. Live capture (the app's own data)

Running `streamlit run scripts/app.py` and pressing Start produces one row per
assessment in `data/stancesense.db` — the hip profile, the stance recommendation
and every rep — plus an annotated video in `recordings/`. This is the app doing
its job, not a research dataset: it is unlabelled, unblinded, single-subject, and
carries no ground truth to score against. **No claim in any of these documents is
derived from it.**

Three conditions must hold for a live reading to mean anything, and the app now
enforces the first two rather than trusting them:

| Condition | How it is enforced |
|---|---|
| The leg being measured is actually visible | hip, knee and ankle must all exceed MediaPipe visibility 0.6, else nothing is recorded and the window shows "NOT MEASURING" |
| The lifter is upright before reps are counted | counting is withheld until a standing posture has been held; otherwise rising from the seated rotation stage books a phantom rep |
| Pose runs on the unmirrored frame | mirroring swaps the estimator's left/right labels, so the mirror is applied after inference (see `METHODOLOGY_UPDATE.md`, "Validity of the live measurement path") |

## 6. Not currently collected

Self-recorded seated-rotation clips, squat clips, and goniometer readings are on
hold as a *validation* source — there is no ground truth to compare against, and
the rotation-accuracy study is dropped regardless. `subjects.csv`,
`goniometer.csv`, and `CONSENT_AND_SAFETY.md` remain as templates should live
collection resume later.
