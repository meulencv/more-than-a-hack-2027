"""One command from the data bundle in data/ to the submission CSV.

    .venv/bin/python scripts/reproduce_submission.py                        # evaluation/results/v29_final.csv
    .venv/bin/python scripts/reproduce_submission.py --team-name NAME       # ... and evaluation/results/NAME.csv

The submission is the variant `final`. Four alternatives that were studied and not submitted
stay available:

    .venv/bin/python scripts/reproduce_submission.py --variant prior70      # evaluation/results/v29_prior70.csv
    .venv/bin/python scripts/reproduce_submission.py --variant blend_hedge  # evaluation/results/v29_blend_hedge.csv
    .venv/bin/python scripts/reproduce_submission.py --variant blend_bet    # evaluation/results/v29_blend_bet.csv
    .venv/bin/python scripts/reproduce_submission.py --variant safe         # evaluation/results/v29_safe.csv
    .venv/bin/python scripts/reproduce_submission.py --variant final prior70 blend_hedge blend_bet safe

The spell verifier is fitted on the three Metrica open games to the target
w * gain_consistent + (1 - w) * gain_alternative (two readings of the written definition):

    final        frozen four-network ensemble; w = 0.5, the equal prior between the two readings
    prior70      frozen four-network ensemble; w = 0.7 (the default of an earlier version of this script)
    blend_hedge  0.5 frozen + 0.5 retrained ensemble; w = 0.7
    blend_bet    0.5 frozen + 0.5 retrained ensemble; w = 1.0 (consistent target only)
    safe         frozen ensemble; target = the smaller of the two gains

Only two things are read: the organisers' data bundle (data/tactical, data/broadcast, or
MTAH_DATA_DIR) and the fitted objects in artifacts/ (network weights and the spell verifier,
none of which is frame-level data of the match). `final`, `prior70` and `safe` read neither
artifacts/retrained/ nor the verifiers of the other variants. Every intermediate is rebuilt in
--work-dir (default data/work/repro); no cache of an earlier run under data/work is read.

Steps (wall-clock on an Apple M5 Max in brackets; see REPRODUCE.md):
    1 tracking    check the input files, parse both ATD files                    [~10 s]
    2 alignment   automatic broadcast -> tactical frame mapping                  [~15 s]
    3 detect      ball detector over the tactical video, MPS float16, 6 procs    [~10 min]
    4 link        detections -> ball track that fills the tracking's gaps        [~10 s]
    5 networks    frozen four-network ensemble, CPU (blend variants: + retrained) [~25 s; ~40 s]
    6 decode      camera fusion, 25 Hz decoder, spell verifier, CSV              [~5 s]

Each intermediate is compared with the content digest recorded in artifacts/MANIFEST.json, so
a run that does not end in the expected CSV says at which step it left the reference run. The
exit status is 0 only when every requested CSV has the expected sha256.

No ground truth of the challenge match is read (public or private).

Maintainers only: `--collect` copies the fitted objects from data/work into artifacts/ and
rewrites the manifest (it needs the working tree that trained them).
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ARTIFACTS = Path("artifacts")
SUBMISSION = "final"
# name -> networks, verifier file, and the name the same candidate has in the study code
# (possession.v29_final.CANDIDATES, possession.v29_lab, data/work/v29): the CSV that study wrote
# is evaluation/results/v29_<lab>.csv and holds the bytes this script must reproduce.
# `final` (w = 0.5) has lab=None: the study run fitted its verifier (target w0.50 of
# possession.v29_lab) and scored it on the open games, but wrote no CSV for it. Its reference
# is the CSV this script wrote, evaluation/results/v29_final.csv.
VARIANTS = {
    "final": dict(base="frozen", verifier="frozen_v27_w0.50.joblib", lab=None),
    "prior70": dict(base="frozen", verifier="frozen_v27_w0.70.joblib", lab="hedge_frozen"),
    "blend_hedge": dict(base="blend", verifier="blend_v27_w0.70.joblib", lab="hedge"),
    "blend_bet": dict(base="blend", verifier="blend_v27_w1.00.joblib", lab="bet"),
    "safe": dict(base="frozen", verifier="frozen_v27_robust.joblib", lab="safe"),
}
DETECT_WORKERS, DETECT_BATCH = 6, 4      # the reference pass; changing them changes the float16 batches


# -- small helpers ---------------------------------------------------------------------------

def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def content_digest(arrays):
    """sha256 of named arrays (dtype, shape, bytes): unlike the sha256 of an .npz file it does
    not depend on the zlib build or on zip metadata."""
    import numpy as np
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(f"{name}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def npz_digest(path, keys=None):
    import numpy as np
    with np.load(path) as saved:
        return content_digest({k: saved[k] for k in (keys or saved.files)})


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class Run:
    """Timing and reference checks of one reproduction."""

    def __init__(self, work, manifest):
        self.work, self.expected = work, manifest.get("expected", {})
        self.report = dict(steps={}, checks={}, environment=environment())
        self.started = time.time()

    def step(self, name):
        run = self

        class Timer:
            def __enter__(self):
                self.t = time.time()
                print(f"\n[{name}] ...", flush=True)

            def __exit__(self, kind, *_):
                if kind is None:
                    run.report["steps"][name] = round(time.time() - self.t, 1)
                    print(f"[{name}] {run.report['steps'][name]} s", flush=True)
                    run.save()
        return Timer()

    def check(self, name, value):
        expected = self.expected.get(name)
        state = "not recorded" if expected is None else ("same" if expected == value else "DIFFERENT")
        self.report["checks"][name] = dict(value=value, expected=expected, state=state)
        print(f"    check {name}: {state}" + ("" if state != "DIFFERENT" else f"  (got {value[:16]}, reference {expected[:16]})"), flush=True)
        return state != "DIFFERENT"

    def save(self):
        self.report["seconds_total"] = round(time.time() - self.started, 1)
        (self.work / "reproduce_report.json").write_text(json.dumps(self.report, indent=2))


def environment():
    import numpy
    import sklearn
    import torch
    import cv2
    return dict(python=platform.python_version(), platform=platform.platform(), machine=platform.machine(),
                torch=torch.__version__, numpy=numpy.__version__, scikit_learn=sklearn.__version__, opencv=cv2.__version__,
                mps=bool(torch.backends.mps.is_available()))


def use_work_cache(work):
    """Every tracking parse goes to <work>/tracking_cache (the modules default to data/work/cache)."""
    from possession import data, video_ball
    original = data.load_tracking
    cache = work / "tracking_cache"

    def load_tracking(source="tactical", cache_dir=None):
        return original(source, cache_dir=cache)
    data.load_tracking = load_tracking
    video_ball.load_tracking = load_tracking


def needed_by(relative):
    """Variants that read the file artifacts/<relative>."""
    folder, name = relative.split("/", 1)
    if folder == "retrained":
        return [v for v, spec in VARIANTS.items() if spec["base"] == "blend"]
    if folder == "verifier":
        return [v for v, spec in VARIANTS.items() if spec["verifier"] == name]
    return list(VARIANTS)                                            # ball detector and frozen ensemble


def read_manifest(artifacts, variants):
    """The manifest, after checking the fitted objects the requested variants read."""
    path = artifacts / "MANIFEST.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing: the fitted objects of the submission are not in this checkout")
    manifest = json.loads(path.read_text())
    for row in manifest["files"]:
        if not set(needed_by(row["path"])) & set(variants):
            continue
        target = artifacts / row["path"]
        if not target.exists() or file_digest(target) != row["sha256"]:
            raise SystemExit(f"{target} is missing or differs from {path}")
    return manifest


def copy_as(source, target):
    """Byte copy that, like the CSV writer, never replaces a file holding other bytes.
    Returns False when such a file was found (and kept)."""
    source, target = Path(source), Path(target)
    if target.exists():
        return file_digest(target) == file_digest(source)
    temporary = target.with_name(target.name + ".part")
    shutil.copyfile(source, temporary)
    temporary.replace(target)
    return True


# -- the steps -------------------------------------------------------------------------------

def step_tracking(run):
    from possession.sources import challenge_match
    from possession.competitive_bundle import camera_match, input_signature
    signature = input_signature()
    reference_signature = json.loads((run.artifacts / "frozen/selection.json").read_text())["input_files_sha256"]
    if signature != reference_signature:
        changed = [k for k in reference_signature if signature.get(k) != reference_signature[k]]
        raise SystemExit("The data bundle is not the one the submission was computed from; different files: " + ", ".join(changed))
    from games import asset_paths
    video = Path(asset_paths()["video"])
    recorded = run.manifest["inputs"]["tactical/" + video.name]
    if not video.exists() or video.stat().st_size != recorded["bytes"] or file_digest(video) != recorded["sha256"]:
        raise SystemExit(f"{video} is missing or is not the video the submission was computed from")
    reference = challenge_match()
    other, valid = camera_match(reference, run.work)
    run.check("tracking_tactical", content_digest(dict(players=reference.players, ball=reference.ball, team=reference.team)))
    run.check("tracking_broadcast", content_digest(dict(players=other.players, ball=other.ball, team=other.team)))
    return reference, other, valid


def step_alignment(run, reference, other, valid):
    """The automatic camera mapping (possession.camera_sync), recomputed; it is frame-level
    data of the match, so it is not shipped."""
    from possession.competitive_bundle import align, mapping_digest
    result = align(reference, other, valid, run.work, None)          # writes <work>/alignment.npz and .json
    run.check("alignment_mapping", mapping_digest(result))
    run.check("alignment_arrays", npz_digest(run.work / "alignment.npz"))
    run.check("alignment_runs", json_digest(json.loads((run.work / "alignment.json").read_text())["runs"]))
    return run.work / "alignment.npz"


def step_detect(run, device):
    """Ball detector over every graded frame of the tactical video."""
    import torch
    from possession import v26_ball
    out = (run.work / "detections_net3.npz").resolve()
    half = device == "mps"
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("The detector pass of the submission ran on Apple MPS in float16 and this machine has no MPS.\n"
                         "`--detector-device cpu` runs it in float32 on the CPU: about a day of computing, and the\n"
                         "detections (so possibly the CSV) are not guaranteed to be the same. See REPRODUCE.md.")
    if device != "mps":
        print("    WARNING: detector on", device, "in float32; the reference pass is MPS float16 and the result may differ", flush=True)
    net = (run.artifacts / "ball_detector/net3").resolve()            # v26_ball appends ".pt"; an absolute path leaves its ROOT
    v26_ball.detect(argparse.Namespace(workers=DETECT_WORKERS, limit=0, device=device, batch=DETECT_BATCH, net=str(net),
                                       half=half, out=str(out)))
    return out


def step_link(run, detections):
    from possession import v26_track
    out = (run.work / "video_ball_f2.npz").resolve()
    v26_track.link(argparse.Namespace(detections=str(Path(detections).resolve()), no_flight_filter=True, out=str(out),
                                      threshold=None, min_weight=3.0, max_far=0.05))
    run.check("video_ball", npz_digest(out, ["ball", "confidence", "source"]))
    return out


def step_networks(run, video_ball, alignment, names):
    import numpy as np
    import torch
    from possession import v29_final as final
    from possession.competitive_bundle import CompetitiveBundle
    from possession.v12_compare import match_record
    torch.set_num_threads(6)
    match, fill = final.filled_match(run.work, video_ball=video_ball, alignment=alignment)
    run.report["ball_fill"] = fill
    record = match_record(match)
    out = {}
    for name in names:
        selection = json.loads((run.artifacts / name / "selection.json").read_text())
        for spec in selection["components"]:                         # the checkpoints next to this selection.json
            spec["path"] = str(run.artifacts / name / Path(spec["path"]).name)
        final.check_fingerprints(selection)
        p, e = CompetitiveBundle(selection, "cpu").probabilities(record)
        np.savez_compressed(run.work / f"{name}_probabilities.npz", probabilities=p, edges=e, frames=record["frames"])
        out[name] = (p, e, record["frames"])
        run.check(f"{name}_probabilities", content_digest(dict(probabilities=p, edges=e, frames=record["frames"])))
    return out


def step_decode(run, triples, alignment, variants, output_dir, team_name=None):
    import joblib
    import numpy as np
    from possession import v28a_core as k28
    from possession import v29_final as final
    if "retrained" in triples:
        triples = dict(triples, blend=final.bases(triples["frozen"], triples["retrained"])["blend"])
    fused = {}
    results = {}
    for name in variants:
        base = VARIANTS[name]["base"]
        if base not in fused:
            probabilities, blocks, n = final.fused(triples[base], alignment=alignment)
            fused[base] = (probabilities, triples[base][1], triples[base][2], blocks, n)
            run.check(f"fused_{base}", content_digest(dict(probabilities=probabilities)))
        saved = joblib.load(run.artifacts / "verifier" / VARIANTS[name]["verifier"])
        if saved["features"] != k28.FEATURES or saved["base"] != base:
            raise SystemExit("The verifier in artifacts/ does not belong to this code")
        after, before, row = final.verify(*fused[base], saved)
        path = Path(output_dir) / f"v29_{name}.csv"
        kept = None
        try:
            sha = final.write_csv(path, after)                       # never replaces a file that holds other bytes
        except RuntimeError:                                         # the new result stays next to it as .part.csv
            kept, path = path, path.with_suffix(".part.csv")
            sha = file_digest(path)
        expected = run.expected.get(f"csv_v29_{name}")
        run.check(f"csv_v29_{name}", sha)
        results[name] = dict(csv=str(path), sha256=sha, expected=expected, identical=expected is not None and sha == expected,
                             existing_file_with_other_bytes_kept=None if kept is None else str(kept),
                             rows=int(len(after)), counts={k: int((final.LABELS[after] == k).sum()) for k in "HADX"},
                             spells_decoder=row["before"]["spells"], spells_final=row["after"]["spells"], removals=row["removals"])
        if team_name:                                                # the same bytes under the name the organisers ask for
            target = Path(output_dir) / f"{team_name}.csv"
            team = dict(csv=str(target), variant=name, written=False, reason=None)
            if expected is None:
                team["reason"] = "the manifest records no sha256 for this variant"
            elif sha != expected:
                team["reason"] = "the CSV of this run is not the expected one"
            elif not copy_as(path, target):
                team["reason"] = "that file exists with other bytes and was kept"
            else:
                team.update(written=True, sha256=file_digest(target))
            results[name]["team_csv"] = team
    run.report["results"] = results
    return results


def reproduce(args):
    os.chdir(REPO)
    os.environ["OMP_NUM_THREADS"] = "6"                              # as in the reference run; set before numpy / torch load
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.artifacts, args.variant)
    use_work_cache(work)
    run = Run(work, manifest)
    run.artifacts, run.manifest = args.artifacts, manifest
    print("environment", json.dumps(run.report["environment"]), flush=True)
    for key in ("python", "torch", "numpy", "scikit_learn", "opencv"):
        if manifest["environment"].get(key) != run.report["environment"][key]:
            print(f"    WARNING: {key} {run.report['environment'][key]} here, {manifest['environment'].get(key)} in the reference run", flush=True)
    names = ["frozen"] + (["retrained"] if any(VARIANTS[v]["base"] == "blend" for v in args.variant) else [])
    with run.step("1 tracking"):
        reference, other, valid = step_tracking(run)
    with run.step("2 alignment"):
        alignment = step_alignment(run, reference, other, valid)
    del reference, other, valid
    if args.detections:
        detections = args.detections
        run.report["detections_reused_from"] = str(detections)
        print(f"\n[3 detect] skipped: detections read from {detections}", flush=True)
    else:
        with run.step("3 detect"):
            detections = step_detect(run, args.detector_device)
    run.check("detections", npz_digest(detections))
    with run.step("4 link"):
        video_ball = step_link(run, detections)
    with run.step("5 networks"):
        triples = step_networks(run, video_ball, alignment, names)
    with run.step("6 decode"):
        results = step_decode(run, triples, alignment, args.variant, args.output_dir, args.team_name)
    run.report["challenge_labels_read"] = False
    run.save()
    print("\nsteps (s):", json.dumps(run.report["steps"]), "total", run.report["seconds_total"])
    different = [k for k, v in run.report["checks"].items() if v["state"] == "DIFFERENT"]
    if different:
        print("first intermediate that differs from the reference run:", different[0], "| all:", ", ".join(different))
    ok = True
    for name, row in results.items():
        what = "the submission" if name == SUBMISSION else f"the reference `{name}` (not the submission)"
        state = "NO REFERENCE sha256 in the manifest for " if row["expected"] is None else ("IDENTICAL to " if row["identical"] else "DIFFERENT from ")
        print(f"{row['csv']}  sha256 {row['sha256']}  " + state + what)
        ok &= row["identical"]
        if row["existing_file_with_other_bytes_kept"]:
            print(f"    NOT WRITTEN to {row['existing_file_with_other_bytes_kept']}: that file holds other bytes and was kept")
            ok = False
        team = row.get("team_csv")
        if team and team["written"]:
            print(f"{team['csv']}  sha256 {team['sha256']}  the same bytes" + ("" if name == SUBMISSION else f"  (variant `{name}`, NOT the submission)"))
        elif team:
            print(f"    NOT WRITTEN to {team['csv']}: {team['reason']}")
            ok = False
    return 0 if ok else 1


# -- maintainers: build artifacts/ from the working tree that trained the models ---------------

SOURCES = [
    # (path under artifacts/, source in the working tree, what produced it)
    ("ball_detector/net3.pt", "data/work/v26/net3.pt",
     "possession.video_ball extract/snap + possession.v26_ball train/mine (three self-training rounds on the tactical "
     "video; pseudo-labels = the tracking's own ball; no human label). REPRODUCE.md section 6.1"),
    ("verifier/blend_v27_w0.70.joblib", "data/work/v29/verifiers/blend_v27_w0.70.joblib",
     "possession.v29_lab table + fit: spell verifier on the blend base, target 0.7*consistent + 0.3*alternative (variant blend_hedge). Section 6.4"),
    ("verifier/blend_v27_w1.00.joblib", "data/work/v29/verifiers/blend_v27_w1.00.joblib",
     "possession.v29_lab table + fit: spell verifier on the blend base, consistent target (variant blend_bet). Section 6.4"),
    ("verifier/frozen_v27_robust.joblib", "data/work/v29/verifiers/frozen_v27_robust.joblib",
     "possession.v29_lab table + fit: spell verifier on the frozen base, min(consistent, alternative) target (variant safe). Section 6.4"),
    ("verifier/frozen_v27_w0.50.joblib", "data/work/v29/verifiers/frozen_v27_w0.50.joblib",
     "possession.v29_lab table + fit: spell verifier on the frozen base, target 0.5*consistent + 0.5*alternative "
     "(variant final, the submission). Section 6.4"),
    ("verifier/frozen_v27_w0.70.joblib", "data/work/v29/verifiers/frozen_v27_w0.70.joblib",
     "possession.v29_lab table + fit: spell verifier on the frozen base, target 0.7*consistent + 0.3*alternative "
     "(variant prior70). Section 6.4"),
]
ENSEMBLES = {
    "frozen": ("docs/FINAL_SELECTION.json",
               "scripts/prepare_final_training.py + possession.v13_cv --final-only: four networks on the six DFL/Sportec open matches "
               "with their native possession labels. Section 6.2"),
    "retrained": ("data/work/v28d/selection.json",
                  "possession.v28d_train: the same four-network recipe on the same six DFL/Sportec matches, labels "
                  "relabelled algorithmically to controlled possession (possession.v28d_relabel). Section 6.3"),
}


def collect(args):
    """Copy the fitted objects into artifacts/ and record digests of the reference run."""
    import numpy as np
    os.chdir(REPO)
    root = args.artifacts
    rows = []

    def add(relative, source, what):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or file_digest(target) != file_digest(source):
            shutil.copyfile(source, target)
        rows.append(dict(path=relative, bytes=target.stat().st_size, sha256=file_digest(target), needed_by=needed_by(relative),
                         copied_from=str(source), produced_by=what))

    for relative, source, what in SOURCES:
        add(relative, source, what)
    for name, (selection_path, what) in ENSEMBLES.items():
        selection = json.loads(Path(selection_path).read_text())
        for spec in selection["components"]:
            relative = f"{name}/{Path(spec['path']).name}"
            if file_digest(spec["path"]) != spec["sha256"]:
                raise SystemExit("Checkpoint changed: " + spec["path"])
            add(relative, spec["path"], what)
            spec["path"] = str(root / relative)
        target = root / name / "selection.json"
        target.write_text(json.dumps(selection, indent=2) + "\n")
        rows.append(dict(path=f"{name}/selection.json", bytes=target.stat().st_size, sha256=file_digest(target),
                         needed_by=needed_by(f"{name}/selection.json"), copied_from=selection_path,
                         produced_by="ensemble definition; only the checkpoint paths were rewritten to artifacts/"))
    # Digests of the intermediates of the run that produced the submission (the original files
    # under data/work, read only here), so that a reproduction can say where it diverges.
    from possession.sources import challenge_match
    from possession.competitive_bundle import camera_match
    v29 = Path("data/work/v29/run")
    tactical = challenge_match()
    broadcast, _ = camera_match(tactical, v29)
    frozen = json.loads(Path(ENSEMBLES["frozen"][0]).read_text())
    expected = dict(
        tracking_tactical=content_digest(dict(players=tactical.players, ball=tactical.ball, team=tactical.team)),
        tracking_broadcast=content_digest(dict(players=broadcast.players, ball=broadcast.ball, team=broadcast.team)),
        alignment_mapping=frozen["alignment_mapping_sha256"],
        alignment_arrays=npz_digest("data/work/v11/challenge_alignment.npz"),
        alignment_runs=json_digest(json.loads(Path("data/work/v11/challenge_alignment.json").read_text())["runs"]),
        detections=npz_digest("data/work/v26/detections_net3.npz"),
        video_ball=npz_digest("data/work/v26/video_ball_f2.npz", ["ball", "confidence", "source"]),
        frozen_probabilities=npz_digest(v29 / "frozen_probabilities.npz", ["probabilities", "edges", "frames"]),
        retrained_probabilities=npz_digest(v29 / "retrained_probabilities.npz", ["probabilities", "edges", "frames"]),
        fused_frozen=npz_digest(v29 / "fused_frozen.npz", ["probabilities"]),
        fused_blend=npz_digest(v29 / "fused_blend.npz", ["probabilities"]))
    variants = {}
    for name, spec in VARIANTS.items():
        if spec["lab"]:                                              # the CSV the study run (possession.v29_final) wrote
            reference_csv, origin = Path("evaluation/results") / f"v29_{spec['lab']}.csv", "study run (possession.v29_final)"
        else:                                                        # no CSV in the study run: the one this script wrote
            reference_csv, origin = Path("evaluation/results") / f"v29_{name}.csv", "scripts/reproduce_submission.py"
        sha = file_digest(reference_csv) if reference_csv.exists() else None
        if sha is None:
            print(f"NOTE: {reference_csv} does not exist yet: no sha256 recorded for `{name}`; run the variant, then --collect again")
        else:
            expected[f"csv_v29_{name}"] = sha
        variants[name] = dict(csv=f"evaluation/results/v29_{name}.csv", sha256=sha,
                              submission=name == SUBMISSION, networks=spec["base"], verifier="verifier/" + spec["verifier"],
                              artifacts=[row["path"] for row in rows if name in row["needed_by"]],
                              name_in_study_code=spec["lab"], reference_csv=str(reference_csv), reference_csv_written_by=origin)
    from games import asset_paths
    video = Path(asset_paths()["video"])
    inputs = {"tactical/" + video.name: dict(bytes=video.stat().st_size, sha256=file_digest(video))}
    manifest = dict(
        note="Fitted objects needed to regenerate the submission and the four alternatives that were not submitted. None of "
             "them is frame-level data of the challenge match: network weights, gradient-boosting spell verifiers fitted on "
             "the open Metrica games, and two ensemble definitions.",
        submission=f"evaluation/results/v29_{SUBMISSION}.csv", submission_variant=SUBMISSION,
        submission_sha256=expected.get(f"csv_v29_{SUBMISSION}"), variants=variants,
        variants_note="`artifacts` lists what each variant reads. The submission (`final`), `prior70` and `safe` do not read "
                      "retrained/. `name_in_study_code` is the name of the same candidate in possession.v29_final / "
                      "possession.v29_lab and in data/work/v29; the study run wrote no CSV for `final` (verifier target w0.50 "
                      "of possession.v29_lab on the frozen base), so its reference is the CSV scripts/reproduce_submission.py "
                      "wrote.",
        environment=environment(), files=rows, inputs=inputs,
        inputs_note="the tactical video; the six tracking and homography files are fingerprinted in frozen/selection.json",
        expected=expected,
        expected_note="content digests (scripts/reproduce_submission.py content_digest) of the intermediates of the study run "
                      "(data/work/v29/run: every variant shares them up to the fused probabilities), and sha256 of the CSV "
                      "files. They identify the step at which another run diverges. retrained_probabilities and fused_blend "
                      "exist only for the blend variants.")
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("artifacts:", len(rows), "files,", sum(r["bytes"] for r in rows), "bytes; expected digests:", len(expected))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", nargs="+", choices=list(VARIANTS), default=[SUBMISSION],
                        help="final (the submission, default); prior70, blend_hedge, blend_bet, safe (studied, not submitted)")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"), help="CSV = <output-dir>/v29_<variant>.csv")
    parser.add_argument("--team-name", default=None, metavar="NAME",
                        help="also write <output-dir>/NAME.csv with the same bytes (the organisers ask for a CSV named after "
                             "the team); needs exactly one variant")
    parser.add_argument("--work-dir", type=Path, default=Path("data/work/repro"), help="every intermediate is rebuilt here")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--detector-device", choices=["mps", "cpu"], default="mps",
                        help="mps = float16, the reference pass; cpu = float32, about a day, result not guaranteed identical")
    parser.add_argument("--detections", type=Path, default=None,
                        help="skip step 3 and read ball detections saved by an earlier run (<work-dir>/detections_net3.npz)")
    parser.add_argument("--collect", action="store_true", help="maintainers: rebuild artifacts/ from data/work")
    args = parser.parse_args()
    args.variant = list(dict.fromkeys(args.variant))
    if args.team_name is not None:
        name = args.team_name[:-4] if args.team_name.lower().endswith(".csv") else args.team_name
        if not name.strip() or name != Path(name).name or name.startswith("."):
            parser.error("--team-name must be a plain file name (no folder)")
        if len(args.variant) != 1:
            parser.error("--team-name needs exactly one --variant")
        if name == f"v29_{args.variant[0]}":
            parser.error("--team-name is the name of the variant's own CSV")
        args.team_name = name
    if args.collect:
        collect(args)
    else:
        sys.exit(reproduce(args))
