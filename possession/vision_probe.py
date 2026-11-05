"""Local multimodal feasibility benchmark; outputs are NOT used as ground truth.

Uses deterministic, uniformly spaced samples outside the public five minutes.
Only model-generated descriptions are stored. Nothing is manually labelled.
"""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from games import GAME, asset_paths, half_video_blocks

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "play_status": {"type": "string", "enum": ["live", "dead", "unknown"]},
        "ball_visible": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "controlled_by_kit": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "evidence": {"type": "string"},
    },
    "required": ["play_status", "ball_visible", "controlled_by_kit", "confidence", "evidence"],
}

PROMPT = """These four football video frames are in chronological order, 0.48 seconds apart.
Assess the CENTRAL time using the whole sequence, not just proximity in one image.
Do not assume a missing or airborne ball means a stoppage. A deflection is not control.
If the ball is too small to resolve, explicitly say unknown. Do not invent details.
Return only JSON with keys: play_status (live/dead/unknown), ball_visible (yes/no/uncertain),
controlled_by_kit (short color description or unknown), confidence (low/medium/high),
evidence (one short sentence describing only visible evidence).
Return ONE object about the central time, not an array and not one answer per image."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("data/work/vision_probe"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Selection depends solely on published half bounds, never on labels or scores.
    eligible = np.concatenate([np.arange(max(a, 18685), b - 50)
                               for _, a, b in half_video_blocks(GAME)])
    if args.limit < 1:
        parser.error("--limit must be positive")
    centers = eligible[(np.linspace(0.2, 0.8, args.limit) * (len(eligible) - 1)).astype(int)]
    cap = cv2.VideoCapture(asset_paths()["video"])
    batches = []
    for center in centers:
        paths = []
        for offset in (-18, -6, 6, 18):
            index = int(center + offset)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
            if not ok:
                raise RuntimeError("Cannot decode frame {}".format(index))
            path = args.output_dir / (str(index) + ".jpg")
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.append(str(path))
        batches.append((int(center), paths))
    cap.release()
    from mlx_vlm import load, generate, apply_chat_template
    from mlx_vlm.utils import load_config
    from mlx_vlm.structured import build_json_schema_logits_processor
    import mlx.core as mx
    started = time.perf_counter()
    print("Loading local vision-language weights...", flush=True)
    model, processor = load(args.model)
    model_config = load_config(args.model)
    load_seconds = time.perf_counter() - started
    for center, paths in batches:
        mx.random.seed(0)
        prompt = apply_chat_template(processor, model_config, PROMPT,
                                     num_images=len(paths), enable_thinking=False)
        started = time.perf_counter()
        constraint = build_json_schema_logits_processor(processor.tokenizer, SCHEMA)
        result = generate(model, processor, prompt, image=paths, max_tokens=300,
                          temperature=0.0, verbose=False, logits_processors=[constraint])
        try:
            parsed = json.loads(result.text)
            schema_valid = (isinstance(parsed, dict) and set(parsed) == set(SCHEMA["required"])
                            and all(isinstance(v, str) for v in parsed.values())
                            and all(parsed[k] in SCHEMA["properties"][k]["enum"]
                                    for k in ("play_status", "ball_visible", "confidence")))
        except (json.JSONDecodeError, TypeError):
            parsed, schema_valid = None, False
        record = {"center_frame": center, "images": paths, "prompt": PROMPT,
                  "model": args.model, "load_seconds": load_seconds,
                  "seconds": time.perf_counter() - started, "text": result.text,
                  "peak_memory_gb": result.peak_memory,
                  "generation_tokens_per_second": result.generation_tps,
                  "parsed": parsed, "schema_valid": schema_valid,
                  "finish_reason": result.finish_reason,
                  "used_for_submission": False, "ground_truth_read": False}
        print(json.dumps(record), flush=True)
        with (args.output_dir / "results.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        mx.clear_cache()


if __name__ == "__main__":
    main()
