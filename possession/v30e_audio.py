"""V30e: per-video-frame audio features of both recordings (stadium sound; no speech content).

    .venv/bin/python -m possession.v30e_audio extract      # both recordings -> data/work/v30e/audio_<source>.npz
    .venv/bin/python -m possession.v30e_audio sync         # audio-only broadcast -> tactical offset per half

Short-time spectrum: 22.05 kHz mono mix (and L/R separately for the whistle band), Hann window
of 1024 samples (46 ms), hop 441 samples = two columns per video frame. Column 2k is centred on
the middle of video frame k (sample 882 k + 441).

Per half-frame column (arrays have 2 * n_video_frames rows):
    band_db[:, j]     log power in BANDS[j]
    tone_db           whistle band (3.3-4.7 kHz): strongest bin over the median of the side bands
    tone_narrow_db    the same bin over the median of the whistle band itself (a tone, not a hiss)
    tone_hz           frequency of that bin (parabolic interpolation)
    harmonic_db       power at twice that frequency over the median around it
    tone_lr_db        left minus right level of the tone
    whistle_spectrum_db   the 65 bins of the whistle band (float16 dB), side_db = median of the side bands
    flux              half-wave rectified spectral flux of the log spectrum, 0.3-8 kHz
    level_db          broadband level
"""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
from games import asset_paths, source_frames

ROOT = Path("data/work/v30e")
FPS = 25
RATE = 22050
WINDOW = 1024
HOP = 441
BANDS = ((80, 300), (300, 1000), (1000, 2000), (2000, 3300), (3300, 4700), (4700, 6500), (6500, 9500))
WHISTLE = (3300.0, 4700.0)
SIDES = ((2200.0, 3100.0), (5000.0, 6200.0))


def decode(video):
    """Stereo PCM (n, 2) float32 of a recording's audio stream."""
    command = ["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "2", "-ar", str(RATE), "-f", "s16le", "-"]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype="<i2").reshape(-1, 2)


def power_columns(signal, first, count):
    """|STFT|^2 of columns first..first+count-1; column c is centred at sample c * HOP + HOP."""
    window = np.hanning(WINDOW).astype(np.float32)
    centre = (np.arange(first, first + count) * HOP + HOP)
    start = centre - WINDOW // 2
    index = start[:, None] + np.arange(WINDOW)[None, :]
    valid = (index >= 0) & (index < len(signal))
    frames = np.where(valid, signal[np.clip(index, 0, len(signal) - 1)], 0).astype(np.float32) / 32768.0
    return (np.abs(np.fft.rfft(frames * window, axis=1)) ** 2).astype(np.float32)


def features(pcm, n_frames, chunk=20000):
    freq = np.fft.rfftfreq(WINDOW, 1.0 / RATE)
    n = 2 * n_frames
    mono = pcm.astype(np.int32).sum(axis=1) // 2
    left, right = pcm[:, 0], pcm[:, 1]
    in_band = np.flatnonzero((freq >= WHISTLE[0]) & (freq < WHISTLE[1]))
    side = np.zeros(len(freq), dtype=bool)
    for low, high in SIDES:
        side |= (freq >= low) & (freq < high)
    wide = (freq >= 300) & (freq < 8000)
    out = dict(band_db=np.zeros((n, len(BANDS)), np.float32), tone_db=np.zeros(n, np.float32),
               tone_narrow_db=np.zeros(n, np.float32), tone_hz=np.zeros(n, np.float32), harmonic_db=np.zeros(n, np.float32),
               tone_lr_db=np.zeros(n, np.float32), flux=np.zeros(n, np.float32), level_db=np.zeros(n, np.float32),
               whistle_spectrum_db=np.zeros((n, len(in_band)), np.float16), side_db=np.zeros(n, np.float32))
    last_log = None
    rows = np.arange(0)
    for first in range(0, n, chunk):
        count = min(chunk, n - first)
        p = power_columns(mono, first, count) + 1e-12
        sl = slice(first, first + count)
        rows = np.arange(count)
        for j, (low, high) in enumerate(BANDS):
            out["band_db"][sl, j] = 10 * np.log10(p[:, (freq >= low) & (freq < high)].mean(axis=1))
        out["level_db"][sl] = 10 * np.log10(p[:, freq >= 80].mean(axis=1))
        out["whistle_spectrum_db"][sl] = 10 * np.log10(p[:, in_band])
        out["side_db"][sl] = 10 * np.log10(np.median(p[:, side], axis=1))
        k = in_band[0] + p[:, in_band].argmax(axis=1)
        peak = p[rows, k]
        out["tone_db"][sl] = 10 * np.log10(peak / np.median(p[:, side], axis=1))
        out["tone_narrow_db"][sl] = 10 * np.log10(peak / np.median(p[:, in_band], axis=1))
        a, b, c = np.log(p[rows, k - 1]), np.log(peak), np.log(p[rows, k + 1])
        shift = np.clip(0.5 * (a - c) / (a - 2 * b + c - 1e-9), -0.5, 0.5)
        hz = (k + shift) * RATE / WINDOW
        out["tone_hz"][sl] = hz
        h = np.clip(np.round(2 * hz * WINDOW / RATE).astype(int), 0, len(freq) - 3)
        around = np.stack([p[rows, np.clip(h + d, 0, len(freq) - 1)] for d in range(-12, 13)], axis=1)
        out["harmonic_db"][sl] = 10 * np.log10(around[:, 10:15].max(axis=1) / np.median(around, axis=1))
        pl = power_columns(left, first, count)[rows, k] + 1e-12
        pr = power_columns(right, first, count)[rows, k] + 1e-12
        out["tone_lr_db"][sl] = 10 * np.log10(pl / pr)
        log = np.log(p[:, wide])
        before = np.concatenate([log[:1] if last_log is None else last_log[None], log[:-1]])
        out["flux"][sl] = np.maximum(log - before, 0).mean(axis=1)
        last_log = log[-1]
    return out


def extract(args):
    ROOT.mkdir(parents=True, exist_ok=True)
    for source in ("tactical", "broadcast"):
        pcm = decode(asset_paths(source=source)["video"])
        n = source_frames(source)
        out = features(pcm, n)
        np.savez_compressed(ROOT / f"audio_{source}.npz", samples=len(pcm), **out)
        print("AUDIO", source, "samples", len(pcm), "seconds", round(len(pcm) / RATE, 2), "video frames", n,
              "median level dB", round(float(np.median(out["level_db"])), 1), flush=True)


def envelope(band_db):
    """Zero-mean fast envelope (per half-frame column) used to align the two sound tracks."""
    x = band_db - np.convolve(band_db, np.ones(101) / 101, mode="same")
    return x.astype(np.float64)


def lag_search(a, b, lags):
    """Normalised correlation of a[t] with b[t + lag] for each lag (columns)."""
    out = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        lo = max(0, -lag)
        hi = min(len(a), len(b) - lag)
        if hi - lo < 500:
            continue
        x, y = a[lo:hi], b[lo + lag:hi + lag]
        out[i] = float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-12))
    return out


def sync(args):
    """Broadcast -> tactical offset from the SOUND alone, per half and per stretch of a half.

    Independent of possession.camera_sync (player constellations). offset = broadcast frame
    minus tactical frame, in video frames (resolution half a frame).
    """
    from games import GAME, half_video_blocks
    t = np.load(ROOT / "audio_tactical.npz")
    b = np.load(ROOT / "audio_broadcast.npz")
    expected = {1: -4014, 2: -26920}
    report = {}
    for half, first, last in half_video_blocks(GAME):
        rows = {}
        for band in (4, 5, 2):
            ta = envelope(t["band_db"][:, band])
            ba = envelope(b["band_db"][:, band])
            pieces = np.linspace(first, last + 1, 7).astype(int)
            found = []
            for a, z in zip(pieces[:-1], pieces[1:]):
                segment = ta[2 * a:2 * z]
                lags = np.arange(-150, 151)                      # half-frames around the camera alignment
                best = []
                for lag in lags:
                    start = 2 * (a + expected[half]) + lag
                    if start < 0 or start + len(segment) > len(ba):
                        best.append(np.nan)
                        continue
                    y = ba[start:start + len(segment)]
                    best.append(float((segment * y).sum() / np.sqrt((segment ** 2).sum() * (y ** 2).sum() + 1e-12)))
                best = np.array(best)
                if np.isfinite(best).any():
                    k = int(np.nanargmax(best))
                    found.append(dict(first=int(a), last=int(z), offset_frames=expected[half] + lags[k] / 2.0,
                                      correlation=round(float(best[k]), 3),
                                      second_best=round(float(np.nanmax(np.delete(best, range(max(0, k - 4), min(len(best), k + 5))))), 3)))
            rows[f"band_{BANDS[band][0]}_{BANDS[band][1]}"] = found
        report[str(half)] = rows
        for name, found in rows.items():
            print("SYNC half", half, name, [(f["offset_frames"], f["correlation"], f["second_best"]) for f in found], flush=True)
    (ROOT / "audio_sync.json").write_text(json.dumps(dict(camera_alignment=expected, audio=report), indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["extract", "sync"])
    args = parser.parse_args()
    dict(extract=extract, sync=sync)[args.step](args)
