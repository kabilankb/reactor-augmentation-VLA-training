#!/usr/bin/env python3
"""Local read-only dashboard for the reactor-augmentation fruit/env pipeline.

Serves dataset stats (coverage, target dE, drop reasons), the scene prompt
files, preview images, and a live tail of build logs -- all read fresh off
disk on every request, so it reflects an in-progress `build_fruit_addition_dataset.py`
run without needing a restart.

Stdlib only, no new dependencies. Read-only: it never triggers a build or
touches REACTOR_API_KEY.

    python3 webui/server.py [--port 8787]
    open http://localhost:8787/
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import subprocess
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent

# Named log files the dashboard can tail. Add an entry here for every new
# build run -- deliberately NOT an arbitrary-path parameter, to keep /api/log
# from being a local file-read gadget.
LOGS: dict[str, Path] = {
    "env50 build": Path(
        "/tmp/claude-1000/-home-zeux-reactor-augmentation/"
        "a1270aa0-700b-4e02-a89a-e3b67846db74/scratchpad/build_env50.log"
    ),
    "fruit95 build": Path(
        "/tmp/claude-1000/-home-zeux-reactor-augmentation/"
        "a1270aa0-700b-4e02-a89a-e3b67846db74/scratchpad/build_95.log"
    ),
}
# Also pick up anything dropped in the repo's own logs/ dir, so future runs
# just need `> logs/whatever.log` to show up automatically.
for p in sorted((ROOT / "logs").glob("*.log")):
    LOGS.setdefault(p.stem, p)

ALLOWED_FILE_EXTS = {".png", ".jpg", ".jpeg", ".json", ".jsonl"}


def read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def read_jsonl(p: Path):
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def dataset_summary(d: Path) -> dict | None:
    info = read_json(d / "meta" / "info.json")
    if info is None:
        return None
    episodes = {e["episode_index"]: e for e in read_jsonl(d / "meta" / "episodes.jsonl")}
    augs = read_jsonl(d / "meta" / "augmentations.jsonl")
    dropped = read_json(d / "meta" / "dropped.json") or []
    ep_rows = []
    for a in sorted(augs, key=lambda x: x["episode_index"]):
        ep_rows.append({
            "episode_index": a["episode_index"],
            "scene_id": a.get("scene_id"),
            "fruits_added": a.get("fruits_added"),
            "prompt": a.get("prompt"),
            "task": a.get("task"),
            "frames_kept": a.get("frames_kept"),
            "frames_before_truncation": a.get("frames_before_truncation"),
            "truncated": a.get("truncated"),
            "coverage": a.get("coverage"),
            "target_dE_median": a.get("target_dE_median"),
            "target_dE_p90": a.get("target_dE_p90"),
        })
    return {
        "name": d.name,
        "path": str(d.relative_to(ROOT)),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "episodes": ep_rows,
        "dropped": dropped,
    }


def api_datasets() -> dict:
    out = []
    ds_dir = ROOT / "datasets"
    if ds_dir.exists():
        for d in sorted(ds_dir.iterdir()):
            if d.is_dir():
                s = dataset_summary(d)
                if s:
                    out.append(s)
    return {"datasets": out}


def api_scenes() -> dict:
    out = []
    for p in sorted(ROOT.glob("fruit_scenes*.json")):
        scenes = read_json(p)
        if scenes is not None:
            out.append({"file": p.name, "scenes": scenes})
    return {"scene_files": out}


def api_previews() -> dict:
    out = []
    for d in sorted(ROOT.glob("out_*")):
        if not d.is_dir():
            continue
        images = sorted(f.name for f in d.iterdir()
                         if f.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if images:
            out.append({"dir": d.name, "images": images})
    return {"preview_dirs": out}


def proc_running(pattern: str) -> bool:
    try:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, timeout=2)
        return r.returncode == 0
    except Exception:
        return False


def api_logs() -> dict:
    out = []
    for name, path in LOGS.items():
        if not path.exists():
            out.append({"name": name, "exists": False})
            continue
        text = path.read_text(errors="replace")
        tail = "\n".join(text.splitlines()[-400:])
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        out.append({
            "name": name,
            "exists": True,
            "path": str(path),
            "mtime": mtime,
            "tail": tail,
            "running": proc_running("build_fruit_addition_dataset.py")
                       or proc_running("lingbot_scene_explore.py"),
        })
    return {"logs": out}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter default logging
        pass

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, p: Path) -> None:
        data = p.read_bytes()
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/" or path == "/index.html":
            return self._file(Path(__file__).with_name("dashboard.html"))
        if path == "/api/datasets":
            return self._json(api_datasets())
        if path == "/api/scenes":
            return self._json(api_scenes())
        if path == "/api/previews":
            return self._json(api_previews())
        if path == "/api/logs":
            return self._json(api_logs())
        if path.startswith("/files/"):
            rel = path[len("/files/"):]
            target = (ROOT / rel).resolve()
            if (target.suffix.lower() in ALLOWED_FILE_EXTS
                    and str(target).startswith(str(ROOT) + "/")
                    and target.exists() and target.is_file()):
                return self._file(target)
            return self._json({"error": "not found"}, 404)

        self._json({"error": "not found"}, 404)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"  reactor-augmentation dashboard: http://localhost:{args.port}/")
    print(f"  serving repo root: {ROOT}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
