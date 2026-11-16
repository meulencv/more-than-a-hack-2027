# Reproducing the submission

The submission is `evaluation/results/v29_final.csv`
(sha256 `19cf15870dec4d7ae88a507128e23a8d04dec3e8d9cc5cc3dbcbf748be10203c`, 199,319 rows).

One command regenerates it from the organisers' data bundle and the fitted objects in
`artifacts/`:

```bash
.venv/bin/python scripts/reproduce_submission.py
```

It takes about 11 minutes on an Apple M5 Max (10 min 33 s and 11 min 27 s in two runs), prints the sha256 of the CSV it wrote, and exits with
status 0 only if that is the sha256 above. The method is described in
`docs/TECHNICAL_REPORT.md`.

No ground truth of the challenge match is read by this command or by any training step below
(neither `evaluation/gt/` nor any hand label). How the public 5-minute window was used during
development (candidates scored with the official scorer as a pipeline check, inspection, and an
earlier default whose 0.7 prior was informed by the window and was replaced by the equal prior for
that reason) is stated in section 6 of `docs/TECHNICAL_REPORT.md`. The submitted file has not been
scored on the public window.

## 1. Environment

| | |
|---|---|
| Machine of the reference run | Apple M5 Max, 64 GB, macOS 27.0.1 (arm64) |
| Python | CPython 3.12.15 |
| Packages | `requirements-submission.lock.txt` (torch 2.15.0, numpy 2.5.4, scikit-learn 1.9.1, opencv-python-headless 5.0.0.93, pandas 3.1.1, scipy 1.18.1, joblib 1.6.0) |

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-submission.lock.txt
```

The lock file is the exact package list of the environment the reference run used. Installing it
into a new virtual environment was not tested for this document (it needs a download).

## 2. Data

Unpack the organisers' bundle into `data/` with the names it arrives with (or point
`MTAH_DATA_DIR` at it; see `data/README.md`). The command reads these seven files:

```
data/tactical/BAR-ATH_tactical.mp4               5.2 GB   ball detector
data/tactical/BAR-ATH_tactical_atd.txt           196 MB   tracking
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_homography.csv    23 MB
data/broadcast/BAR-ATH_broadcast_atd.txt         142 MB   second camera: ball and camera cues
data/broadcast/BAR-ATH_broadcast_atd.xml
data/broadcast/BAR-ATH_broadcast_homography.csv  15 MB
```

The broadcast video and the Smart Tagging file are not used. The six tracking and homography files
are checked against the sha256 values recorded in `artifacts/frozen/selection.json`, and the video
against `artifacts/MANIFEST.json`; the command stops if one of them is a different file.

The pipeline keeps nothing from the match in the repository. Every intermediate (parsed tracking,
camera alignment, ball detections, ball track, network outputs) is rebuilt in the work directory,
which is under the ignored `data/` folder.

## 3. The command

```bash
.venv/bin/python scripts/reproduce_submission.py                    # evaluation/results/v29_final.csv
.venv/bin/python evaluation/validate_submission.py evaluation/results/v29_final.csv
.venv/bin/python scripts/reproduce_submission.py --team-name NAME   # also evaluation/results/NAME.csv
```

The organisers ask for a CSV named after the team. `--team-name NAME` writes
`<output-dir>/NAME.csv` with the same bytes as the variant's CSV, and only when that CSV has the
expected sha256.

| Option | Default | Meaning |
|---|---|---|
| `--variant` | `final` | one or more of `final`, `prior70`, `blend_hedge`, `blend_bet`, `safe` |
| `--team-name NAME` | none | also write `<output-dir>/NAME.csv` with the same bytes; needs exactly one variant |
| `--output-dir` | `evaluation/results` | the CSV is `<output-dir>/v29_<variant>.csv` |
| `--work-dir` | `data/work/repro` | every intermediate is rebuilt here (75 MB) |
| `--artifacts` | `artifacts` | folder with the fitted objects |
| `--detections FILE` | none | skip the detector pass and reuse the `detections_net3.npz` an earlier run left in its work directory |
| `--detector-device` | `mps` | `cpu` runs the detector in float32; see section 5 |

An existing CSV is never replaced. If the file is already there with the same bytes the run
succeeds; if it holds other bytes the new result is left next to it as `v29_<variant>.part.csv`
and the exit status is 1. The same holds for `NAME.csv`: a file with other bytes is kept and the
exit status is 1.

### Variants

`final` is the submission. The other four were studied and not submitted; they stay available
with `--variant`:

```bash
.venv/bin/python scripts/reproduce_submission.py --variant prior70       # evaluation/results/v29_prior70.csv
.venv/bin/python scripts/reproduce_submission.py --variant blend_hedge   # evaluation/results/v29_blend_hedge.csv
.venv/bin/python scripts/reproduce_submission.py --variant blend_bet     # evaluation/results/v29_blend_bet.csv
.venv/bin/python scripts/reproduce_submission.py --variant safe          # evaluation/results/v29_safe.csv
```

The verifier target is `w x gain under the consistent reading + (1 - w) x gain under the
alternative reading` (section 6.4).

| Variant | Networks | Verifier target | File | sha256 |
|---|---|---|---|---|
| `final` (the submission) | frozen | w = 0.5, the equal prior | `v29_final.csv` | `19cf15870dec4d7ae88a507128e23a8d04dec3e8d9cc5cc3dbcbf748be10203c` |
| `prior70` | frozen | w = 0.7 | `v29_prior70.csv` | `ed3e87d911e3070662b96550d3759a0801422790622c7a252998ab878dfaea7a` |
| `blend_hedge` | frozen + retrained, 50/50 | w = 0.7 | `v29_blend_hedge.csv` | `2d1ac7a7063a6d83ca59e7b757f89b973a32ec0bcfbc1df9286181ca404b82ff` |
| `blend_bet` | frozen + retrained, 50/50 | w = 1.0 (consistent only) | `v29_blend_bet.csv` | `8219bb6a2d4409eba1c0adfc81358d073f8b9e0b93583a76305948abdda00294` |
| `safe` | frozen | minimum of the two | `v29_safe.csv` | `1dafcd2e633e618812c516bcb66870767f0c7b7dc939e82f442538b6e1104f71` |

`prior70` is the default of the earlier version of this script, which wrote it to
`v29_final.csv`. Its 0.7 was informed by the public window of the challenge, which is why it is
no longer the submission (`docs/TECHNICAL_REPORT.md`, section 6). A `v29_final.csv` with the
sha256 `ed3e87d9...` is that earlier file.

The study code (`possession/v29_final.py`, `possession/v29_lab.py`) and its outputs under
`data/work/v29` use older names for four of the candidates: `hedge_frozen` (= `prior70`), `hedge`
(= `blend_hedge`), `bet` (= `blend_bet`) and `safe`. The CSVs that study wrote are
`evaluation/results/v29_hedge_frozen.csv`, `v29_hedge.csv`, `v29_bet.csv` and `v29_safe.csv`;
the expected sha256 values of those four variants are the sha256 of those files. The study run
fitted and scored the verifier of `final` (target `w0.50` of `possession.v29_lab` on the frozen
base) but wrote no CSV for it. Its expected sha256 is that of the CSV this command wrote in run A
of section 4, which is also what `possession.v29_final.verify` gives when it is applied with the
same verifier to the fused probabilities saved by the study run.

## 4. What the command does, and how long it takes

| # | Step | Device | Time (M5 Max, all runs below) | Rebuilt in the work directory |
|---|---|---|---|---|
| 1 | Check the input files; parse both tracking files | CPU | 5-7 s | `tracking_cache/` (64 MB) |
| 2 | Camera alignment: broadcast frame -> tactical frame (`possession/camera_sync.py`) | CPU | 13-18 s | `alignment.npz`, `alignment.json` |
| 3 | Ball detector over the 158,857 graded frames of the tactical video | MPS, float16, 6 processes | 9 min 13 s to 11 min 4 s | `detections_net3.npz` (12 MB) |
| 4 | Link detections into a ball track that fills the gaps of the tracking's ball | CPU | 7-13 s | `video_ball_f2.npz` |
| 5 | Frozen four-network ensemble on the filled ball | CPU, 6 threads | 22-27 s | `frozen_probabilities.npz` |
| 6 | Camera fusion, 25 Hz decoder, spell verifier, CSV | CPU | 3-4 s | the CSV |
| | Total | | 10 min 33 s and 11 min 27 s (runs A and C below) | 75 MB |

The camera alignment, the detections, the ball track and the network outputs are frame-level data
of the match, so none of them is shipped; they are recomputed every time.

`blend_hedge` and `blend_bet` also run the retrained ensemble in step 5 (38-43 s for both ensembles,
`retrained_probabilities.npz`) and blend the two outputs 50/50 before the camera fusion. `final`,
`prior70` and `safe` do not read `artifacts/retrained/`.

After each step the command compares what it produced with a content digest of the same
intermediate in the study run (`artifacts/MANIFEST.json`, key `expected`; all variants share the
intermediates up to the fused probabilities). If
the final CSV differs, the first check reported as `DIFFERENT` is the step where the run left the
reference. The timings and all checks are written to `<work-dir>/reproduce_report.json`.

### Verification (16 November, Apple M5 Max)

With the script as published (default variant `final`, the equal prior). Runs A and C are complete
runs of the default command, each into an empty work directory, with no cache of an earlier run
read.

| Run | Where | Command | Wall-clock | Result |
|---|---|---|---|---|
| A | working repository | `scripts/reproduce_submission.py --work-dir data/work/repro_final_w50` | 10 min 33 s | `v29_final.csv` written new, sha256 `19cf1587...`. This is the run that established that sha256: the manifest had none for `final` yet, so the exit status was 1 by design |
| B | working repository, detections of run A reused | `scripts/reproduce_submission.py --detections data/work/repro_final_w50/detections_net3.npz --work-dir data/work/repro_final_w50_b --output-dir data/work/repro_final_w50_b/out --team-name TEAM_NAME_TEST` | 53 s | `v29_final.csv` and `TEAM_NAME_TEST.csv` written new in another folder, both byte-identical to the file of run A, exit status 0 |
| C | working repository | `scripts/reproduce_submission.py --work-dir data/work/repro_final_w50_c` | 11 min 27 s | the CSV has the expected sha256 (the file of run A was found with the same bytes), exit status 0 |
| D | working repository, detections of run A reused | `scripts/reproduce_submission.py --variant final prior70 blend_hedge blend_bet safe --detections data/work/repro_final_w50/detections_net3.npz --work-dir data/work/repro_final_w50_all` | 1 min 15 s | all five CSVs have the expected sha256, exit status 0 |
| E | copy of the published files only, detections of run A reused | `scripts/reproduce_submission.py --detections <run A>/detections_net3.npz --team-name TEAM_NAME_TEST` | 54 s | `v29_final.csv` and `TEAM_NAME_TEST.csv` written new, both with the expected sha256, exit status 0 |

In the five runs every intermediate had the digest of the study run: parsed tracking, camera
alignment, detections, ball track, network outputs and fused probabilities.

- Run A is the default command. Its CSV passes `evaluation/validate_submission.py` (199,319 rows).
  The detector pass took 9 min 40 s (10 min 29 s in run C). Its sha256 was then recorded in the
  manifest (`--collect`), and runs B to E were checked against it.
- The sha256 does not rest on this command alone: applying `possession.v29_final.verify` (the
  study code) with the same verifier to the fused probabilities saved by the study run
  (`data/work/v29/run`) gives a CSV with the same bytes. The same check gives the bytes of
  `v29_hedge_frozen.csv` for the w = 0.7 verifier.
- Run B repeats everything after the detector pass and writes into an empty folder, so its two
  files are new files and not the file of run A found again.
- Run E started from a copy of only the files git would publish (tracked files plus the new ones),
  without `evaluation/gt/`, without `artifacts/retrained/` and without the verifiers of the other
  four variants, with `data/` holding only its README and the bundle reached through
  `MTAH_DATA_DIR`. It shows that the submission needs no file outside the published ones and the
  data bundle, and none of the fitted objects of the other variants. It is not a complete run: the
  detector pass was not repeated in the copy.
- Run D moved no file: the five CSVs were already in `evaluation/results/` with the same bytes
  (`v29_prior70.csv` is the file the earlier version of the script had written as `v29_final.csv`).

Logs and reports: `data/work/repro_final_w50/`, `repro_final_w50_b/`, `repro_final_w50_c/`,
`repro_final_w50_all/` and `repro_final_w50_clean/` (run E), each with `run.log` and
`reproduce_report.json`.

Earlier the same day, with two earlier versions of the script. They differed in the names of the
variants and in which one was the default, not in any computing step. In the first table the
default was the variant now called `prior70`, which that version wrote to `v29_final.csv`:

| Run | Command (names of that version) | Wall-clock | Result |
|---|---|---|---|
| 4 | `scripts/reproduce_submission.py --work-dir data/work/repro_final` | 11 min 32 s | the CSV now called `v29_prior70.csv` written new, expected sha256 (`ed3e87d9...`), exit status 0 |
| 5 | default command with `--team-name TEAM_NAME_TEST`, from a copy of the published files only | 11 min 15 s | the same CSV and `TEAM_NAME_TEST.csv` written new, both with that sha256, exit status 0 |
| 6 | `scripts/reproduce_submission.py --variant final blend_hedge blend_bet safe --detections data/work/repro_final/detections_net3.npz --work-dir data/work/repro_final_all` | 1 min 30 s | all four CSVs had the expected sha256, exit status 0 |

Before that the default was the variant now called `blend_hedge`:

| Run | Command (names of that version) | Wall-clock | Result |
|---|---|---|---|
| 1 | `scripts/reproduce_submission.py --work-dir data/work/repro_check` | 10 min 49 s | `v29_hedge.csv` (now `blend_hedge`) identical |
| 2 | `scripts/reproduce_submission.py --variant hedge bet safe hedge_frozen --work-dir data/work/repro_check_all` | 10 min 26 s | all four CSVs identical, among them `v29_hedge_frozen.csv` (now `prior70`) |
| 3 | default command, from a copy of the published files only | 12 min 43 s (machine busy) | `v29_hedge.csv` written new, expected sha256, exit status 0 |

In runs 1 to 6 every intermediate had the digest of the study run. Their logs and reports are
`data/work/repro_check*/`, `data/work/repro_final/`, `data/work/repro_final_clean/` and
`data/work/repro_final_all/`.

## 5. Hardware

The ball detector pass (step 3) is the only part that needs a GPU, and the part most tied to the
hardware. The submission was computed with it on Apple MPS in float16.

- **Apple Silicon with MPS.** Verified on one machine (M5 Max): the detections are identical
  across runs, and so is the CSV. Another Apple Silicon model, another macOS version or another
  torch build were not tested.
- **No MPS.** The command stops at step 3 and says so. `--detector-device cpu` runs the detector
  in float32 on the CPU. That takes about a day (roughly 1.4 frames per second for 158,857
  frames), and the result is not guaranteed to be the same: on a 300-frame sample, float32 and
  float16 put 2 of 973 detections on different sides of the 0.4 confidence threshold. Whether
  that changes the CSV was not tested.
- **Steps 1, 2, 4, 5 and 6** run on the CPU (numpy, scikit-learn, torch with 6 threads). They were
  verified on the same machine only. The network ensembles give slightly different floats on
  MPS (maximum difference 4.2e-07), which is why the command always runs them on the CPU.

So the byte-identical CSV is established for one machine: an Apple M5 Max with MPS, macOS 27.0.1
and the package versions of the lock file. On any other hardware the command still reports, step
by step, whether each intermediate equals the reference.

The ball detections are frame-level data of the match, so they are not shipped; that is why the
detector pass has to be repeated.

## 6. What is in `artifacts/` and how it was trained

`artifacts/MANIFEST.json` lists every file with its size, sha256, origin and the variants that
read it (`needed_by`; per variant under `variants`). Total 6.2 MB. None of the files is
frame-level data of the match.

| File | What | Trained on | Read by |
|---|---|---|---|
| `ball_detector/net3.pt` | ball detector for the tactical video (0.7 MB) | the challenge video, with the tracking's own ball as pseudo-label; no human label | all variants |
| `frozen/*.pt`, `frozen/selection.json` | four possession networks (2.3 MB) | six DFL/Sportec open matches, native possession labels | all variants |
| `verifier/frozen_v27_w0.50.joblib` | spell verifier of the submission (0.16 MB) | three Metrica open sample games, event-derived references | `final` |
| `verifier/` the other four `.joblib` | spell verifiers of the other variants (0.16 MB each) | the same | one variant each (section 6.4) |
| `retrained/*.pt`, `retrained/selection.json` | the same four networks, retrained (2.3 MB) | the same six matches, labels converted by rule to controlled possession | `blend_hedge`, `blend_bet` |

The submission reads the first three rows: seven files, 3.2 MB. The command checks only the files
of the requested variants against the manifest, so the default command runs without
`artifacts/retrained/` and without the other four verifiers.

Labelled data, all external to the challenge match:

- **DFL/Sportec open tracking data**: Bassek, Rein, Weber, Memmert (2025), *An integrated dataset
  of spatiotemporal and event data in elite soccer*, figshare,
  doi 10.6084/m9.figshare.28196177, CC BY 4.0. Six full matches (J03WMX, J03WN1, J03WOH, J03WPY,
  J03WOY, J03WQQ) with the provider's per-frame ball possession and ball status. Pinned in
  `docs/sportec_v6_manifest.json`.
- **Metrica Sports open sample data**: github.com/metrica-sports/sample-data at revision
  `e706dd5`, three games with tracking and event data. Pinned in
  `docs/external_data_manifest.json`. Used only to fit and test the spell verifier and to choose
  between pipeline variants, never to train a network.

The challenge match contributes no label anywhere. Two training steps read its unlabelled inputs:
the ball detector (video, tracking ball, homography), and the sensor degradation used to make the
external games look like automatic tracking (`possession/degrade.py`: run lengths of ball
visibility, player-to-ball distance and number of tracked players of the challenge tracking).

The commands below are the ones recorded with each training run. They were not re-run for this
document. Training on MPS is not bit-reproducible, which is why the weights are shipped.

### 6.1 Ball detector (`ball_detector/net3.pt`)

Three rounds of self-training on the tactical video. One-minute blocks (every eighth minute) are
held out of training and used to check the detector against the tracking's own ball.

```bash
.venv/bin/python -m possession.video_ball extract
.venv/bin/python -m possession.video_ball snap
.venv/bin/python -m possession.v26_ball train --data seed --out net1 --epochs 8
.venv/bin/python -m possession.v26_ball mine --net net1 --out mined --no-columns --workers 8
.venv/bin/python -m possession.v26_ball train --data mined --out net2 --epochs 12 --no-columns
.venv/bin/python -m possession.v26_ball mine --net net2 --out mined2 --no-columns --workers 8
.venv/bin/python -m possession.v26_ball mine --net net2 --out mined3 --workers 8
.venv/bin/python -m possession.v26_ball train --data mined2,mined3 --out net3 --epochs 16
```

### 6.2 Frozen ensemble (`frozen/`)

Two temporal networks (`standard`, `physics`) and two trajectory networks (seeds 2027, 2028), each
fitted on the six DFL matches seen through clean, moderate and stress sensor views. Protocol:
`docs/PROTOCOLO_V13.json`; results: `docs/RESULTADOS_MODELO_FINAL.md`.

```bash
.venv/bin/python scripts/fetch_v6_native.py
OMP_NUM_THREADS=6 .venv/bin/python scripts/prepare_final_training.py
OMP_NUM_THREADS=6 .venv/bin/python -m possession.v13_cv --final-only --output-root data/work/retrained_final
```

### 6.3 Retrained ensemble (`retrained/`; not used by the submission)

The same recipe, games, seeds and steps. The only change is the target: the DFL native label
switches possession much more often than the challenge's controlled possession does (7.1 to 11.0
possessions per live minute in the six matches, against about 4 to 6 in the Metrica open games;
in practice at almost every touch, which is our reading of those statistics and not the
dataset's stated definition), so it is converted to controlled possession by a tracking-only rule
(`possession/v28d_relabel.py`: a possession between two possessions of the other team counts only
if the ball stays within 1.5 m of a player of that team for at least 1.2 s).

```bash
.venv/bin/python -m possession.v28d_relabel write
.venv/bin/python -m possession.v28d_train --component standard --device cpu
.venv/bin/python -m possession.v28d_train --component physics --device cpu
.venv/bin/python -m possession.v28d_train --component trajectory2027 --device mps
.venv/bin/python -m possession.v28d_train --component trajectory2028 --device mps
.venv/bin/python -m possession.v28d_train --selection
```

### 6.4 Spell verifier (`verifier/`)

A small gradient-boosting regressor that scores the removal of each predicted possession of 8 s or
less. It is fitted on the outputs of the same ensemble on the three Metrica games (each in three
sensor conditions) against references derived from the Metrica event data. The target is
`w * gain under the "consistent" reference + (1 - w) * gain under the "alternative" reference`.

A possession is removed when its score reaches 0 (the best-scored first, in up to three passes).

| File | Variant | Networks | Target |
|---|---|---|---|
| `frozen_v27_w0.50.joblib` | `final` (the submission) | frozen | w = 0.5 |
| `frozen_v27_w0.70.joblib` | `prior70` | frozen | w = 0.7 |
| `blend_v27_w0.70.joblib` | `blend_hedge` | frozen + retrained, 50/50 | w = 0.7 |
| `blend_v27_w1.00.joblib` | `blend_bet` | frozen + retrained, 50/50 | w = 1.0 |
| `frozen_v27_robust.joblib` | `safe` | frozen | minimum of the two gains |

w = 0.5 is the equal prior between the two readings: it is not fitted, and it uses no information
about the challenge match. The fit reads only the Metrica open games; its seed is fixed
(`random_state=0`).

The verifier of the submission is fitted on the outputs of the frozen ensemble on the Metrica
games (`possession.v16_transfer`). The command list is the same for all five files, because
`v29_lab table` and `fit` process the frozen and the blend base and all targets (w = 1.0, 0.85,
0.7, 0.5 and the minimum) in one pass. The shipped `frozen_v27_w0.50.joblib` comes from that pass
(16 November, before the equal prior was chosen for the submission).

```bash
.venv/bin/python scripts/fetch_external.py
.venv/bin/python -m possession.v16_transfer          # frozen ensemble on the Metrica games
.venv/bin/python -m possession.v27_lab prepare       # event-derived references
.venv/bin/python -m possession.v28d_data             # network inputs of the Metrica games
.venv/bin/python -m possession.v28d_eval predict --tag retrained --selection data/work/v28d/selection.json
.venv/bin/python -m possession.v28d_eval blend --tag blend --sources frozen retrained
.venv/bin/python -m possession.v28a_lab signals
.venv/bin/python -m possession.v28a_lab table
.venv/bin/python -m possession.v29_lab table
.venv/bin/python -m possession.v29_lab hedge         # leave-one-game-out table, data/work/v29/hedge.json
.venv/bin/python -m possession.v29_lab fit           # data/work/v29/verifiers/*.joblib
```

These commands give the consistent, original and alternative columns of the leave-one-game-out
table. The mixture column (q = 0.5) also needs `python -m possession.v29_lab mix` before `hedge`.
That step imports two scripts of an internal review (`rv_common.py`, `rv_refs.py`, which read only
the Metrica open games) from the ignored folder `data/work/review_v27_reference`. They are not in
this repository, so the mixture column cannot be regenerated from the published files; without
`mix`, `hedge` writes the table without it. The values are in `docs/evidence/hedge.json`.

Checked again for the submission's verifier, from the saved candidate table
(`data/work/v29/table_frozen.npz`, 2,483 candidate removals):

- the leave-one-game-out row of the w = 0.5 target (and of w = 0.7 and 1.0) was recomputed and is
  identical to the row in `data/work/v29/hedge.json` (copy in `docs/evidence/hedge.json`);
- the final fit was repeated twice. Both fits predict exactly the same values as the shipped file
  on the 2,483 candidates and on 200,000 random inputs. The `.joblib` bytes are not the same from
  one fitting session to another (the pickle depends on the session, not the model), so the sha256
  in the manifest identifies the shipped file and cannot be reproduced by fitting again.

Loading the `.joblib` files needs the scikit-learn version of the lock file.

### 6.5 Rebuilding `artifacts/`

```bash
.venv/bin/python scripts/reproduce_submission.py --collect
```

copies the files above from `data/work/` into `artifacts/` and rewrites the manifest. It needs the
working tree in which the models were trained.

## 7. What must not be published

`data/` and `evaluation/results/*.csv` are ignored by git and must stay so: the work directory
holds frame-level exports of the match, and a results file is a set of labels for the whole match.
That includes `v29_final.csv` and the team-named copy written by `--team-name`: the CSV is sent to
the organisers, not committed. `docs/evidence/` holds only aggregate numbers (statistics of the
external games, of the held-out minutes of the ball detector and of whole prediction files), no
per-frame value of the match.
`.gitignore` makes one exception for model files, the `artifacts/` folder.

The only frame-level files of the match under version control are the ones the organisers' own
repository ships: the four `evaluation/results/baseline_*.csv` and the public ground-truth window
`evaluation/gt/BAR-ATH_possession_gt_public.csv`. Nothing in this pipeline reads them.
