"""Video helpers for matched comparison pages: probe, byte-budget transcode and inline embedding."""

import base64
import hashlib
import json
import math
import subprocess
from pathlib import Path


STYLE = """<style>
*{box-sizing:border-box}body{margin:0;background:#faf9f6;color:#18251f;font:14px system-ui;line-height:1.45;padding:20px}
.eyebrow{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:#66736a;margin:0 0 8px}h1,h2{font-weight:650;letter-spacing:-.03em;margin:0 0 7px}h1{font-size:28px}h2{font-size:24px}.sub{color:#66736a;margin:0 0 18px}
.films{display:grid;gap:14px;grid-template-columns:repeat(var(--columns,3),minmax(0,1fr))}.film{background:#fff;border:1px solid #e3e6df;border-radius:12px;overflow:hidden;min-width:0}.filmhead{padding:13px;min-height:70px;display:flex;align-items:center;justify-content:space-between;gap:7px}.name{font-weight:650;font-size:13px}.tag{font-size:10px;font-weight:650;border-radius:5px;padding:4px 6px;white-space:nowrap;background:#edf0ee;color:#69736b}.tag.pass{color:#276343;background:#e6f2e9}.tag.fail{color:#985c42;background:#f8ede7}
.media{width:100%;aspect-ratio:4/3;overflow:hidden;background:#0e1513}.media video{width:100%;height:100%;display:block;object-fit:contain}.zoomed .media video{transform:scale(2);transform-origin:50% 60%}.filmfoot{padding:12px 13px;color:#66736a;font-size:12px}.metric{display:flex;align-items:baseline;justify-content:space-between;gap:8px;margin:5px 0}.metric b{color:#1e2d25;font-size:17px;font-variant-numeric:tabular-nums}.absent{grid-column:1/-1;display:grid;grid-template-columns:minmax(280px,.8fr) 2fr;align-items:center}.absent .filmhead{min-height:0}.absent .metric{display:inline-flex;margin-left:15px;gap:10px}.absent .metric b{font-size:13px}
.controls{margin:16px 0 0;display:flex;gap:8px;align-items:center;flex-wrap:wrap}button,select{border:1px solid #d5dcd3;border-radius:7px;background:#fff;color:#27362b;font:inherit;padding:8px 11px;cursor:pointer}button.primary{background:#254a37;color:#fff;border-color:#254a37}input[type=range]{accent-color:#254a37;flex:1;min-width:120px}.time{font-variant-numeric:tabular-nums;color:#66736a;font-size:12px;min-width:90px}.footer{margin-top:12px;border-top:1px solid #e1e6de;padding-top:12px;color:#778079;font-size:11px}
table{width:100%;border-collapse:collapse;font-size:13px}th{text-align:left;font-weight:550;color:#66736a}td,th{padding:12px 8px;border-bottom:1px solid #e2e6df;vertical-align:top}td:first-child{font-weight:600;min-width:140px}.tablewrap{overflow-x:auto}section+section{margin-top:32px;padding-top:24px;border-top:1px solid #dce2d9}.notice{border-left:3px solid #c7d4c9;padding:8px 12px;background:#f0f3ee;color:#57675c}
@media(max-width:700px){body{padding:12px}h1{font-size:24px}h2{font-size:21px}.films{grid-template-columns:1fr}.absent{display:block}.absent .metric{margin-left:0}.controls{position:sticky;bottom:0;background:#faf9f6;padding:10px 0}}
</style>"""


def probe(path: Path) -> dict:
    data = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,duration:format=duration",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )
    streams = data.get("streams", [])
    if not streams:
        raise ValueError(f"No real video stream: {path}")
    stream = streams[0]
    duration = float(stream.get("duration") or data.get("format", {}).get("duration", 0))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid video duration: {path}")
    return dict(width=int(stream["width"]), height=int(stream["height"]), duration_s=duration)


def format_number(value, unit: str, multiplier: float = 1.0) -> str:
    try:
        number = float(value) * multiplier
        if math.isfinite(number):
            return f"{number:.1f} {unit}"
    except (TypeError, ValueError):
        pass
    return "—"


def transcode(
    source: Path,
    destination: Path,
    budget: int,
    *,
    crop: str | None = None,
    compact: bool = True,
    duration: float | None = None,
) -> dict:
    info = probe(source)
    seconds = min(info["duration_s"], duration) if duration else info["duration_s"]
    levels = (
        [
            (512, 12, 29),
            (480, 10, 30),
            (420, 10, 32),
            (420, 8, 33),
            (360, 6, 35),
            (320, 4, 38),
            (288, 3, 40),
            (256, 3, 42),
            (224, 2, 44),
        ]
        if compact
        else [(960, 25, 24), (800, 25, 26), (720, 20, 28), (640, 15, 29), (512, 15, 31)]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    for width, fps, crf in levels:
        filters = [crop] if crop else []
        filters += [f"fps={fps}", f"scale=min({width}\\,iw):-2"]
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-t",
            str(seconds),
            "-vf",
            ",".join(filters),
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            "slow" if compact else "fast",
            "-threads",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        subprocess.run(command, check=True, timeout=120)
        if destination.stat().st_size <= budget:
            break
    else:
        # Last bounded fallback uses an explicit bitrate; time remains unchanged.
        bitrate = max(3000, int((budget - 6000) * 8 * 0.85 / seconds))
        bounded = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-t",
            str(seconds),
            "-vf",
            ",".join(filters),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-threads",
            "2",
            "-b:v",
            str(bitrate),
            "-maxrate",
            str(bitrate),
            "-bufsize",
            str(bitrate * 2),
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        subprocess.run(bounded, check=True, timeout=120)
        if destination.stat().st_size > budget:
            raise ValueError(f"Real clip cannot fit byte budget: {destination.stat().st_size}>{budget}")
    return dict(
        source=str(source),
        path=str(destination),
        bytes=destination.stat().st_size,
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        source_probe=info,
        preview_probe=probe(destination),
        crop=crop,
        retiming=False,
        compact=compact,
    )


def encoded(path: Path) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing actual MP4: {path}")
    return "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
