"""Common in-memory representation of a match for v3 (challenge and external games).

Every source is converted to the SAME units so features are comparable across datasets:

* positions in metres, origin at one corner, x along the pitch length;
* player speed is always recomputed from positions (the challenge ATD speed channel is
  declared as km/h in its XML while DATA.md calls it m/s; external CSVs have no speed);
* visibility is an absolute count of tracked players, never a fraction of slots
  (the challenge has 46 slots, external games 28-36).

Nothing here reads ground truth or event data of the challenge game.
"""
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

FPS = 25
EXTERNAL_PITCH = (105.0, 68.0)
EXTERNAL_DIR = Path("data/external/metrica_sample")


@dataclass
class Match:
    name: str
    players: np.ndarray          # (n, slots, 2) metres, NaN when not tracked
    team: np.ndarray             # (slots,) 0 = home (H), 1 = away (A)
    ball: np.ndarray             # (n, 2) metres, NaN when not tracked
    pitch: tuple
    blocks: list                 # [(period, first_frame, last_frame)] inclusive, frames to label
    meta: dict = field(default_factory=dict)

    def copy_with(self, **changes):
        return replace(self, **changes)


def challenge_match(cache_dir="data/work/cache"):
    """The Barcelona-Athletic tactical tracking. ATD frame_id f -> video frame f-1."""
    from games import GAMES, GAME, half_video_blocks
    from .data import load_tracking
    players, ball, slots = load_tracking(cache_dir=cache_dir)
    pitch = tuple(GAMES[GAME]["pitch"])
    scale = np.array(pitch, dtype=np.float32)
    return Match("BAR-ATH", players[:, :, :2] * scale, slots.astype(np.int8),
                 ball[:, :2] * scale, pitch, list(half_video_blocks(GAME)))


def _csv_game(root, game):
    home = pd.read_csv(root / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv", skiprows=2)
    away = pd.read_csv(root / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv", skiprows=2)
    if not np.array_equal(home["Frame"], away["Frame"]):
        raise ValueError("External home/away frames are not aligned")
    n = len(home)
    if not np.array_equal(home["Frame"].to_numpy(), np.arange(1, n + 1)):
        raise ValueError("External frames do not start at 1 and increase by 1")
    parts, teams = [], []
    for k, frame in enumerate((home, away)):
        xy = frame.iloc[:, 3:-2].to_numpy(dtype=np.float32).reshape(n, -1, 2)
        parts.append(xy)
        teams.append(np.full(xy.shape[1], k, dtype=np.int8))
    players = np.concatenate(parts, axis=1) * np.array(EXTERNAL_PITCH, dtype=np.float32)
    ball = home.iloc[:, -2:].to_numpy(dtype=np.float32) * np.array(EXTERNAL_PITCH, dtype=np.float32)
    raw = pd.read_csv(root / f"Sample_Game_{game}_RawEventsData.csv")
    events = pd.DataFrame({
        "team": raw["Team"].map({"Home": 0, "Away": 1}).astype(int),
        "type": raw["Type"].str.upper(),
        "subtype": raw["Subtype"].fillna("").str.upper(),
        "period": raw["Period"].astype(int),
        # Event frames index the tracking file, whose first frame is 1.
        "start": raw["Start Frame"].astype(int) - 1,
        "end": raw["End Frame"].astype(int) - 1,
    })
    return players, np.concatenate(teams), ball, events


def _epts_game(root):
    meta = ET.parse(root / "Sample_Game_3_metadata.xml").getroot()
    team_ids = [t.get("id") for t in meta.iter("Team")]
    player_team = {p.get("id"): team_ids.index(p.get("teamId")) for p in meta.iter("Player")}
    channel_player = {c.get("id"): c.get("playerId") for c in meta.iter("PlayerChannel")}
    order = list(player_team)
    slot = {pid: i for i, pid in enumerate(order)}
    specs = []
    for spec in meta.iter("DataFormatSpecification"):
        ids = []
        for ref in spec.iter("PlayerChannelRef"):
            pid = channel_player[ref.get("playerChannelId")]
            if not ids or ids[-1] != pid:
                ids.append(pid)
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")),
                      np.array([slot[p] for p in ids])))
    lines = (root / "Sample_Game_3_tracking.txt").read_text().splitlines()
    n = len(lines)
    players = np.full((n, len(order), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    spec_index = 0
    for i, line in enumerate(lines):
        frame_s, body, ball_s = line.split(":")
        frame = int(frame_s)
        if frame != i + 1:
            raise ValueError("Game 3 tracking frames are not consecutive")
        while not specs[spec_index][0] <= frame <= specs[spec_index][1]:
            spec_index += 1
        values = np.array(body.replace(";", ",").split(","), dtype=np.float32).reshape(-1, 2)
        players[i, specs[spec_index][2]] = values
        ball[i] = np.array(ball_s.split(","), dtype=np.float32)
    players *= np.array(EXTERNAL_PITCH, dtype=np.float32)
    ball *= np.array(EXTERNAL_PITCH, dtype=np.float32)
    data = json.loads((root / "Sample_Game_3_events.json").read_text())["data"]
    rows = []
    for e in data:
        sub = e.get("subtypes")
        if isinstance(sub, dict):
            sub = sub["name"]
        elif isinstance(sub, list):
            sub = "-".join(s["name"] for s in sub)
        rows.append({"team": team_ids.index(e["team"]["id"]), "type": e["type"]["name"].upper(),
                     "subtype": (sub or "").upper(), "period": int(e["period"]),
                     "start": int(e["start"]["frame"]) - 1, "end": int(e["end"]["frame"]) - 1})
    return players, np.array([player_team[p] for p in order], dtype=np.int8), ball, pd.DataFrame(rows)


def external_match(game, root=EXTERNAL_DIR):
    """Metrica sample game 1, 2 or 3 with its approximate event-derived reference labels."""
    from .reference import event_reference
    root = Path(root)
    if game in (1, 2):
        players, team, ball, events = _csv_game(root, game)
    elif game == 3:
        players, team, ball, events = _epts_game(root)
    else:
        raise ValueError("Unknown external game")
    labels, blocks = event_reference(events, len(ball))
    return Match(f"metrica_sample_{game}", players, team, ball, EXTERNAL_PITCH, blocks,
                 {"labels": labels, "events": events})
