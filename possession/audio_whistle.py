"""Referee whistles from the audio track of the tactical recording.

The tactical video carries stadium sound on the submission's own timeline. A whistle is a
narrow tone between 3.3 and 4.7 kHz, but so are the crowd's whistles, so the referee's is
identified by its own signature: the two published kick-offs are each preceded by a whistle
that can only be the referee's, and those two blasts give the pitch, the warble (spread of the
pitch) and the minimum length and prominence that every other blast is compared against.

Label-free check on the challenge game (possession labels never read): of the blasts that match
the template while the decoder says "in play", about 85% are followed by a decoded stoppage
within five seconds; for blasts at the crowd's pitch that figure is about 20%.
"""
import subprocess
import numpy as np
from scipy.signal import stft

FPS = 25
RATE = 22050                    # 882 samples per video frame
WINDOW = 2048
BAND = (3300.0, 4700.0)
REFERENCE = ((2000.0, 3000.0), (5200.0, 6500.0))
EVENT_DB = 18.0                 # prominence over the neighbouring bands that counts as a tone
MERGE_GAP = 6
KICKOFF_LEAD = 75               # the kick-off whistle sounds within 3 s before the kick
PITCH_TOLERANCE = 100.0         # Hz around the kick-off whistles
RELATIVE_FLOOR = 0.5            # minimum warble and length, relative to the kick-off whistles
TONE_SLACK = 8.0                # dB below the weaker kick-off whistle


def decode_audio(video_path):
    """Mono PCM of a recording's audio track, decoded by ffmpeg."""
    command = ["ffmpeg", "-v", "error", "-i", str(video_path), "-vn", "-ac", "1",
               "-ar", str(RATE), "-f", "s16le", "-"]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def tone_track(video_path, n_frames):
    """Per video frame: prominence (dB) of the strongest tone in the whistle band, and its pitch."""
    hop = RATE // FPS
    freq, _, spectrum = stft(decode_audio(video_path), fs=RATE, nperseg=WINDOW, noverlap=WINDOW - hop,
                             window="hann", boundary=None, padded=False)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    del spectrum
    in_band = (freq >= BAND[0]) & (freq < BAND[1])
    reference = np.zeros(len(freq), dtype=bool)
    for low, high in REFERENCE:
        reference |= (freq >= low) & (freq < high)
    floor = np.median(power[reference], axis=0) + 1e-12
    peak = power[in_band].argmax(axis=0) + np.flatnonzero(in_band)[0]
    prominence = 10 * np.log10(power[peak, np.arange(power.shape[1])] / floor + 1e-12)
    # STFT column k is centred WINDOW/2 samples in: video frame k + 1 at 882 samples per frame.
    shift = int(round((WINDOW / 2) / hop))
    covered = min(len(prominence), n_frames - shift)
    tone = np.zeros(n_frames)
    pitch = np.full(n_frames, np.nan)
    tone[shift:shift + covered] = prominence[:covered]
    pitch[shift:shift + covered] = freq[peak][:covered]
    return tone, pitch


def tone_events(tone, pitch, first, last):
    """Tonal blasts in [first, last): start, length, peak prominence, pitch at the peak, warble."""
    loud = tone[first:last] > EVENT_DB
    d = np.diff(np.r_[0, loud.astype(np.int8), 0])
    runs = []
    for a, z in zip(np.flatnonzero(d == 1) + first, np.flatnonzero(d == -1) + first):
        if runs and a - runs[-1][1] < MERGE_GAP:
            runs[-1][1] = z
        else:
            runs.append([a, z])
    events = []
    for a, z in runs:
        if z - a < 2:
            continue
        sounding = tone[a:z] > EVENT_DB
        k = a + int(np.argmax(tone[a:z]))
        events.append(dict(start=int(a), length=int(z - a), tone=float(tone[k]), pitch=float(pitch[k]),
                           spread=float(np.std(pitch[a:z][sounding]))))
    return events


def referee_whistles(tone, pitch, blocks, kickoffs):
    """Start frames of blasts that match the kick-off whistles, inside the graded halves."""
    templates = []
    for kickoff in kickoffs:
        before = tone_events(tone, pitch, kickoff - KICKOFF_LEAD, kickoff + 1)
        if not before:
            raise ValueError("No whistle found before the published kick-off at {}".format(kickoff))
        templates.append(max(before, key=lambda e: e["tone"]))
    low = min(t["pitch"] for t in templates) - PITCH_TOLERANCE
    high = max(t["pitch"] for t in templates) + PITCH_TOLERANCE
    spread = RELATIVE_FLOOR * min(t["spread"] for t in templates)
    length = RELATIVE_FLOOR * min(t["length"] for t in templates)
    loud = min(t["tone"] for t in templates) - TONE_SLACK
    starts = []
    for (_, first, last), kickoff in zip(sorted(blocks, key=lambda b: b[1]), sorted(kickoffs)):
        for event in tone_events(tone, pitch, max(first, kickoff + 1), last + 1):
            if (low <= event["pitch"] <= high and event["spread"] >= spread
                    and event["length"] >= length and event["tone"] >= loud):
                starts.append(event["start"])
    return np.array(sorted(starts), dtype=int), dict(templates=templates, pitch=[low, high], spread=spread,
                                                     length=length, tone=loud)
