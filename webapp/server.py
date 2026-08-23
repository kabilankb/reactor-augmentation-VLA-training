#!/usr/bin/env python3
"""Local control panel for the Reactor -> LeRobot -> LeIsaac pipeline.

Runs on localhost only. Wraps the scripts already in this project plus the
LeIsaac install at ~/leisaac, so nothing here reimplements the pipeline — it
shells out to the same commands you would type.

    python webapp/server.py            # http://127.0.0.1:8080

Long jobs (fetching, augmenting, Isaac Sim) run as background subprocesses with
their logs streamed to the browser, because an Isaac Sim launch takes minutes
and a request must not block on it.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
LEISAAC = Path.home() / "leisaac"
LEISAAC_PY = Path.home() / "miniconda3" / "envs" / "leisaac" / "bin" / "python"
ENV_FILE = ROOT / ".env"
JOBS_DIR = ROOT / "webapp" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Reactor x LeIsaac control panel")

# ---------------------------------------------------------------- job runner

JOBS: dict[str, dict] = {}
_lock = threading.Lock()


def start_job(name: str, cmd: list[str], cwd: Path, env_extra: dict | None = None) -> str:
    """Spawn a subprocess and tee its output to a log file."""
    jid = uuid.uuid4().hex[:8]
    log = JOBS_DIR / f"{jid}.log"
    env = os.environ.copy()
    env.update(read_env())
    if env_extra:
        env.update(env_extra)
    # Unbuffered, so the browser sees progress instead of one blob at the end.
    env["PYTHONUNBUFFERED"] = "1"

    fh = log.open("w")
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    with _lock:
        JOBS[jid] = {"id": jid, "name": name, "cmd": " ".join(shlex.quote(c) for c in cmd),
                     "cwd": str(cwd), "log": str(log), "pid": proc.pid,
                     "started": time.time(), "proc": proc, "status": "running"}

    def waiter():
        rc = proc.wait()
        fh.close()
        with _lock:
            JOBS[jid]["status"] = "done" if rc == 0 else f"failed ({rc})"
            JOBS[jid]["returncode"] = rc
            JOBS[jid]["ended"] = time.time()

    threading.Thread(target=waiter, daemon=True).start()
    return jid


def job_view(j: dict) -> dict:
    return {k: v for k, v in j.items() if k != "proc"}


# ------------------------------------------------------------------- config

def read_env() -> dict:
    out = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


class KeyIn(BaseModel):
    api_key: str


@app.get("/api/config")
def get_config():
    key = read_env().get("REACTOR_API_KEY", "")
    return {"has_key": bool(key),
            # Never return the key itself — the browser has no use for it and
            # the server is the only thing that should hold it.
            "masked": (key[:6] + "…" + key[-4:]) if len(key) > 12 else "",
            "leisaac": LEISAAC.exists(), "leisaac_python": LEISAAC_PY.exists()}


@app.post("/api/config")
def set_config(body: KeyIn):
    key = body.api_key.strip()
    if not key.startswith("rk_"):
        raise HTTPException(400, "Reactor keys start with rk_")
    env = read_env()
    env["REACTOR_API_KEY"] = key
    ENV_FILE.write_text("".join(f"{k}={v}\n" for k, v in env.items()))
    ENV_FILE.chmod(0o600)
    return {"ok": True}


# ----------------------------------------------------------------- datasets

SAMPLE_DATASETS = [
    {"repo": "LightwheelAI/leisaac-pick-orange", "task": "Grab orange and place into plate",
     "robot": "so101_follower", "episodes": 60, "video_key": "observation.images.front",
     "note": "AV1 video — decoded via ffmpeg fallback"},
    {"repo": "nvidia/Arena-GR1-Manipulation-Task", "task": "Open the microwave",
     "robot": "GR1 humanoid", "episodes": 50, "video_key": "observation.images.ego_view",
     "note": "GR00T-LeRobot layout under lerobot/"},
    {"repo": "lerobot/svla_so101_pickplace", "task": "SO-101 pick and place",
     "robot": "so101_follower", "episodes": None, "video_key": "observation.images.front",
     "note": "official LeRobot sample"},
]


@app.get("/api/hf/samples")
def hf_samples():
    return SAMPLE_DATASETS


@app.get("/api/hf/info")
def hf_info(repo: str):
    """Episode count and video keys for any LeRobot repo, straight from the Hub."""
    import urllib.request
    # HuggingFace rejects requests with no User-Agent, so send one. Some repos
    # (the GR1 one) nest the LeRobot tree under lerobot/, so try both.
    info = None
    errs = []
    for prefix in ("", "lerobot/"):
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{prefix}meta/info.json"
        req = urllib.request.Request(url, headers={"User-Agent": "reactor-augmentation/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                info = json.loads(r.read())
                break
        except Exception as exc:
            errs.append(f"{prefix or '<root>'}: {exc}")
    if info is None:
        raise HTTPException(400, f"could not read meta/info.json for {repo} — " + "; ".join(errs))
    keys = [k for k in info.get("features", {}) if k.startswith("observation.images")]
    return {"repo": repo, "fps": info.get("fps"), "robot_type": info.get("robot_type"),
            "total_episodes": info.get("total_episodes"), "total_frames": info.get("total_frames"),
            "video_keys": keys}


class FetchIn(BaseModel):
    repo: str
    episode: int = 0
    video_key: str = "observation.images.front"
    name: str | None = None


@app.post("/api/hf/fetch")
def hf_fetch(body: FetchIn):
    name = body.name or f"{body.repo.split('/')[-1]}_ep{body.episode}"
    cmd = [sys.executable, str(ROOT / "webapp" / "fetch_episode.py"),
           "--repo", body.repo, "--episode", str(body.episode),
           "--video-key", body.video_key, "--name", name]
    return {"job": start_job(f"fetch {body.repo} ep{body.episode}", cmd, ROOT)}


@app.get("/api/episodes")
def episodes():
    out = []
    d = ROOT / "episodes"
    if d.exists():
        for e in sorted(d.iterdir()):
            man = e / "manifest.jsonl"
            if man.exists():
                rows = man.read_text().splitlines()
                first = json.loads(rows[0]) if rows else {}
                out.append({"name": e.name, "frames": len(rows), "task": first.get("task"),
                            "action_dim": len(first.get("action", []))})
    return out


def _episode_dir(name: str) -> Path:
    if "/" in name or "\\" in name or name in (".", ".."):
        raise HTTPException(400, "bad episode name")
    d = ROOT / "episodes" / name
    if not (d / "manifest.jsonl").exists():
        raise HTTPException(404, "no such source episode")
    return d


@app.get("/api/source/{name}/frame/{seq}")
def source_frame(name: str, seq: int):
    """One raw recorded frame -- the real footage, pre-augmentation."""
    d = _episode_dir(name)
    p = d / "rgb" / f"f{seq:06d}.png"
    if not p.exists():
        raise HTTPException(404, f"no frame {seq}")
    return FileResponse(p, media_type="image/png")


@app.get("/api/source/{name}/actions")
def source_actions(name: str):
    """Same {n, dim, series} shape as /api/actions, so the frontend plot() is shared."""
    import numpy as np
    d = _episode_dir(name)
    rows = [json.loads(l) for l in (d / "manifest.jsonl").read_text().splitlines()]
    a = np.stack([np.asarray(r["action"], float) for r in rows])
    return {"n": int(a.shape[0]), "dim": int(a.shape[1]),
            "series": [a[:, i].tolist() for i in range(a.shape[1])],
            "task": rows[0].get("task") if rows else None,
            "source_seq": [r["seq"] for r in rows]}


# ---------------------------------------------------------------- augmenting

SCENES_GLOB = "fruit_scenes*.json"


@app.get("/api/scene_files")
def scene_files():
    return sorted(p.name for p in ROOT.glob(SCENES_GLOB))


@app.get("/api/scenes/{filename}")
def scene_file(filename: str):
    # Only ever read fruit_scenes*.json directly under ROOT -- no path
    # components allowed, so this can't be turned into an arbitrary file read.
    if "/" in filename or "\\" in filename or not filename.startswith("fruit_scenes") \
            or not filename.endswith(".json"):
        raise HTTPException(400, "not a scenes file")
    p = ROOT / filename
    if not p.exists():
        raise HTTPException(404, "no such scenes file")
    return json.loads(p.read_text())


class AugmentIn(BaseModel):
    episode: str
    scenes: list[dict]          # [{id, fruits, prompt}]
    out_name: str
    count: int = 96
    batch: int = 96
    coverage_floor: float = 0.75
    target_de_ceiling: float = 30.0


@app.post("/api/augment")
def augment(body: AugmentIn):
    """Add-distractor-fruit build, gated on the ORANGE surviving per frame.

    Shells out to build_fruit_addition_dataset.py -- the script this whole
    project actually validated (coverage floor + per-frame, per-orange target
    dE gate with truncation), not the older coverage-only build_fruit_datasets.py.
    """
    if not read_env().get("REACTOR_API_KEY"):
        raise HTTPException(400, "set the Reactor API key first")
    spec = JOBS_DIR / f"scenes_{uuid.uuid4().hex[:8]}.json"
    spec.write_text(json.dumps(body.scenes, indent=2))
    cmd = [sys.executable, str(ROOT / "build_fruit_addition_dataset.py"),
           "--episode", str(ROOT / "episodes" / body.episode),
           "--scenes", str(spec), "--count", str(body.count), "--batch", str(body.batch),
           "--coverage-floor", str(body.coverage_floor),
           "--target-de-ceiling", str(body.target_de_ceiling),
           "--out", str(ROOT / "datasets" / body.out_name),
           "--preview", str(ROOT / "webapp" / "static" / "previews" / body.out_name)]
    return {"job": start_job(f"augment -> {body.out_name}", cmd, ROOT)}


# ------------------------------------------------------------- hub publishing

class HfPushIn(BaseModel):
    dataset: str
    repo_id: str
    private: bool = False


@app.post("/api/hf/push")
def hf_push(body: HfPushIn):
    ds_dir = ROOT / "datasets" / body.dataset
    if not (ds_dir / "meta" / "info.json").exists():
        raise HTTPException(404, f"no dataset at {ds_dir}")
    if "/" not in body.repo_id:
        raise HTTPException(400, "repo_id must be namespace/name")
    cmd = [sys.executable, str(ROOT / "webapp" / "push_hf.py"),
           "--dataset", str(ds_dir), "--repo-id", body.repo_id]
    if body.private:
        cmd.append("--private")
    return {"job": start_job(f"push {body.dataset} -> {body.repo_id}", cmd, ROOT)}


# ----------------------------------------------------------------- model info

MODELS = [
    {"id": "xmax/x2", "name": "X2", "category": "Streaming Video Editing",
     "used_for": "Editing real recorded episode frames in place (add distractor "
                 "fruit, restyle lighting). Preserves the source arm motion, so "
                 "output frames can carry the original action/state labels."},
    {"id": "reactor/lingbot-world-2", "name": "LingBot World 2",
     "category": "Action Controlled World Generation",
     "used_for": "Generating a new navigable world from a seed image + prompt, "
                 "driven by camera movement it controls. No video input, so it "
                 "cannot preserve an existing episode's motion -- not used for "
                 "building this project's paired LeRobot datasets."},
]


@app.get("/api/models")
def models():
    return MODELS


# --------------------------------------------------------------- visualising

@app.get("/api/datasets")
def datasets():
    out = []
    d = ROOT / "datasets"
    for c in sorted(d.iterdir()) if d.exists() else []:
        info = c / "meta" / "info.json"
        if info.exists():
            i = json.loads(info.read_text())
            out.append({"name": c.name, "episodes": i.get("total_episodes"),
                        "frames": i.get("total_frames"), "fps": i.get("fps"),
                        "robot": i.get("robot_type")})
    return out


@app.get("/api/dataset/{name}")
def dataset(name: str):
    root = ROOT / "datasets" / name
    if not (root / "meta" / "info.json").exists():
        raise HTTPException(404, "no such dataset")
    info = json.loads((root / "meta" / "info.json").read_text())
    aug_file = root / "meta" / "augmentations.jsonl"
    aug = [json.loads(l) for l in aug_file.read_text().splitlines()] if aug_file.exists() else []
    eps_file = root / "meta" / "episodes.jsonl"
    eps = [json.loads(l) for l in eps_file.read_text().splitlines()] if eps_file.exists() else []
    vkeys = [k for k in info.get("features", {}) if k.startswith("observation.images")]
    return {"info": info, "augmentations": aug, "episodes": eps,
            "video_key": vkeys[0] if vkeys else "observation.images.front"}


@app.get("/api/video/{name}/{ep}")
def video(name: str, ep: int, key: str = "observation.images.front"):
    p = ROOT / "datasets" / name / "videos" / "chunk-000" / key / f"episode_{ep:06d}.mp4"
    if not p.exists():
        raise HTTPException(404, f"no video at {p}")
    return FileResponse(p, media_type="video/mp4")


@app.get("/api/actions/{name}/{ep}")
def actions(name: str, ep: int):
    """Action trace for the plot under the player."""
    import pandas as pd
    p = ROOT / "datasets" / name / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
    if not p.exists():
        raise HTTPException(404, "no such episode")
    df = pd.read_parquet(p)
    import numpy as np
    a = np.stack([np.asarray(x, float) for x in df["action"]])
    return {"n": int(a.shape[0]), "dim": int(a.shape[1]),
            "series": [a[:, i].tolist() for i in range(a.shape[1])],
            "source_seq": df["source_seq"].tolist() if "source_seq" in df else None}


# -------------------------------------------------------------------- leisaac

LEISAAC_TASKS = ["LeIsaac-SO101-PickOrange-v0", "LeIsaac-SO101-PickOrange-Mimic-v0",
                 "LeIsaac-SO101-LiftCube-v0", "LeIsaac-SO101-AssembleHamburger-v0"]


class LeIsaacIn(BaseModel):
    action: str                 # teleop | inference | convert
    task: str = LEISAAC_TASKS[0]
    headless: bool = True
    policy_type: str = "gr00tn1.5"
    policy_host: str = "localhost"
    policy_port: int = 5555
    checkpoint: str | None = None
    instruction: str | None = None
    dataset: str | None = None


@app.get("/api/leisaac/tasks")
def leisaac_tasks():
    return {"tasks": LEISAAC_TASKS, "root": str(LEISAAC), "python": str(LEISAAC_PY),
            "available": LEISAAC.exists() and LEISAAC_PY.exists()}


@app.post("/api/leisaac/run")
def leisaac_run(body: LeIsaacIn):
    if not LEISAAC_PY.exists():
        raise HTTPException(400, f"leisaac python not found at {LEISAAC_PY}")
    py = str(LEISAAC_PY)
    if body.action == "inference":
        cmd = [py, "scripts/evaluation/policy_inference.py", "--task", body.task,
               "--policy_type", body.policy_type, "--policy_host", body.policy_host,
               "--policy_port", str(body.policy_port)]
        if body.checkpoint:
            cmd += ["--policy_checkpoint_path", body.checkpoint]
        if body.instruction:
            cmd += ["--policy_language_instruction", body.instruction]
    elif body.action == "teleop":
        cmd = [py, "scripts/environments/teleoperation/teleop_se3_agent.py", "--task", body.task]
    elif body.action == "convert":
        if not body.dataset:
            raise HTTPException(400, "convert needs a dataset")
        cmd = [py, "scripts/convert/isaaclab2lerobot.py", "--task_name", body.task]
    else:
        raise HTTPException(400, f"unknown action {body.action}")
    if body.headless:
        cmd.append("--headless")
    return {"job": start_job(f"leisaac {body.action}", cmd, LEISAAC)}


@app.get("/api/gpu")
def gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        gpus = []
        for line in out.splitlines():
            n, used, total, util = [x.strip() for x in line.split(",")]
            gpus.append({"name": n, "used_mb": int(used), "total_mb": int(total), "util": int(util)})
        return {"gpus": gpus}
    except Exception as exc:
        return {"gpus": [], "error": str(exc)}


# ----------------------------------------------------------------------- jobs

@app.get("/api/jobs")
def jobs():
    with _lock:
        return sorted((job_view(j) for j in JOBS.values()), key=lambda j: -j["started"])


@app.get("/api/jobs/{jid}")
def job(jid: str, tail: int = 400):
    with _lock:
        j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "no such job")
    log = Path(j["log"])
    text = log.read_text(errors="replace") if log.exists() else ""
    lines = text.splitlines()
    return {**job_view(j), "log_tail": "\n".join(lines[-tail:]), "log_lines": len(lines)}


@app.post("/api/jobs/{jid}/stop")
def stop(jid: str):
    with _lock:
        j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "no such job")
    try:
        # Kill the whole process group: Isaac Sim spawns children that outlive
        # a bare terminate on the parent.
        os.killpg(os.getpgid(j["pid"]), signal.SIGTERM)
    except ProcessLookupError:
        pass
    return {"ok": True}


# ------------------------------------------------------------------ frontend

STATIC = Path(__file__).parent / "static"
(STATIC / "previews").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text()


if __name__ == "__main__":
    import uvicorn
    print("  Reactor x LeIsaac control panel  ->  http://127.0.0.1:8080")
    print(f"  project: {ROOT}")
    print(f"  leisaac: {LEISAAC} ({'found' if LEISAAC.exists() else 'MISSING'})")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")
