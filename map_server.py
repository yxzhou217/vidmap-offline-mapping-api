"""map_server.py — 离线建图 API: 上传一段视频 → VidMap 建图 → 产出 COLMAP 模型 + 定位 npz 地图 + 检索库

长任务异步服务(建图 30~60 分钟,不可能同步等)。提交后立即返回 job_id,轮询 status,完成后下载产物。

流水线(每个 job 依次执行,全局同时只跑一个,其余排队):
  1. vidmap.run              (vidmap 环境, GPU)  视频 → COLMAP 模型 rec/
  2. export_rec_npz.py       (vidmap 环境, CPU)  rec/ → rec.npz (names/c2w/内参)
  3. make_map_npz_generic.py (map 环境, CPU)     rec.npz + 抽帧 → <scene>_map.npz
  4. build_map_db.py         (map 环境, 小GPU)   map.npz → <scene>_db.npz (DINO 描述子)
  5. visualization.html      (vidmap 环境, CPU)  rec/ → 3D 可视化 HTML (可选)

产物位置:
  任务目录:  map_jobs/<job_id>/          (视频、rec/、日志、rec.zip、html)
  地图注册:  output/<scene>_map.npz 和 <scene>_db.npz  (首帧定位 API 直接用)

用法:
    python map_server.py --port 8200

环境变量配置(都有默认值):
    VIDMAP_DIR       VidMap 仓库路径(默认: 与本仓库并列的 ../vidmap)
    VIDMAP_PYTHON    vidmap conda 环境的 python(默认: 当前 python)
    MAP_PYTHON       map conda 环境的 python(默认: 当前 python)
    LINGBOT_MAP_DIR  lingbot-map 仓库路径(转 npz 要 import lingbot_map;不设则沿用当前 PYTHONPATH)
    MAP_JOBS_DIR     任务目录(默认: ./map_jobs)
    MAP_OUTPUT_DIR   地图注册目录(默认: ./output)
"""

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

# ---- 路径常量(均可用环境变量覆盖) ----
REPO_DIR = Path(__file__).resolve().parent
VIDMAP_DIR = Path(os.environ.get("VIDMAP_DIR", REPO_DIR.parent / "vidmap"))
VIDMAP_PY = os.environ.get("VIDMAP_PYTHON", sys.executable)
MAP_PY = os.environ.get("MAP_PYTHON", sys.executable)
LINGBOT_MAP_DIR = os.environ.get("LINGBOT_MAP_DIR", "")
JOBS_ROOT = Path(os.environ.get("MAP_JOBS_DIR", REPO_DIR / "map_jobs"))
OUTPUT_DIR = Path(os.environ.get("MAP_OUTPUT_DIR", REPO_DIR / "output"))

STAGES = ["queued", "vidmap_frontend", "vidmap_mapping", "export", "map_npz", "db", "html", "done"]
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".MOV", ".MP4"}
MIN_FREE_VRAM_MB = 20000                     # frontend 峰值 ~15.2GB,留余量


@dataclass
class Job:
    job_id: str
    scene: str
    video_name: str
    keyframe_drift: Optional[float] = None
    make_html: bool = True
    status: str = "queued"                   # 见 STAGES, 另有 failed
    error: Optional[str] = None
    gpu: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    num_keyframes: Optional[int] = None
    num_registered: Optional[int] = None

    @property
    def dir(self) -> Path:
        return JOBS_ROOT / self.job_id

    def public(self) -> dict:
        d = asdict(self)
        d["elapsed_s"] = round((self.finished_at or time.time()) - (self.started_at or self.created_at))
        if self.status not in ("done", "failed"):
            d["log_tail"] = read_log_tail(self)
        return d


JOBS: dict[str, Job] = {}
JOB_QUEUE: "queue.Queue[str]" = queue.Queue()
LOCK = threading.Lock()


# ---------------- 工具 ----------------

def read_log_tail(job: Job, n: int = 8) -> list[str]:
    log = job.dir / "run.log"
    if not log.exists():
        return []
    lines = log.read_text(errors="replace").splitlines()
    # 剥掉 tqdm 的 \r 刷屏,只留信息行
    lines = [l for l in lines if "\r" not in l or "INFO" in l]
    return [l[-200:] for l in lines[-n:]]


def log_msg(job: Job, msg: str):
    with open(job.dir / "run.log", "a") as f:
        f.write(f"\n===== [{time.strftime('%H:%M:%S')}] {msg} =====\n")


def pick_free_gpu() -> int:
    """选一张空闲卡: 已用显存最小且总量-已用 > MIN_FREE_VRAM_MB;都不满足就报错排队。"""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True)
    best, best_free = None, -1
    for line in out.strip().splitlines():
        idx, used, total = [int(x.strip()) for x in line.split(",")]
        free = total - used
        if free > MIN_FREE_VRAM_MB and free > best_free:
            best, best_free = idx, free
    if best is None:
        raise RuntimeError("当前没有空闲显存超过 20GB 的 GPU,请稍后再试")
    return best


def run_cmd(job: Job, cmd: list[str], *, cwd: Path, gpu: Optional[int] = None, extra_env: dict = None):
    """跑一个子进程,输出追加到 job 日志;非零返回码抛异常。"""
    import os
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if extra_env:
        env.update(extra_env)
    log_msg(job, "运行: " + " ".join(str(c) for c in cmd))
    with open(job.dir / "run.log", "a") as lf:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=lf, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        tail = "\n".join(read_log_tail(job, 15))
        raise RuntimeError(f"命令失败(exit {proc.returncode}),日志尾部:\n{tail}")


def set_stage(job: Job, stage: str):
    with LOCK:
        job.status = stage
    log_msg(job, f"进入阶段: {stage}")


def sanitize_scene(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
    if not name:
        raise ValueError("场景名为空")
    return name[:64]


# ---------------- 流水线 ----------------

def run_pipeline(job: Job):
    job.started_at = time.time()
    outdir = job.dir / "output"
    rec_dir = outdir / "rec"
    video = job.dir / ("input" + Path(job.video_name).suffix)

    # 1. VidMap 建图(frontend+mapping 一体)
    set_stage(job, "vidmap_frontend")
    job.gpu = pick_free_gpu()
    cmd = [VIDMAP_PY, "-m", "vidmap.run", "--input_data", str(video), "--output", str(outdir)]
    if job.keyframe_drift:
        cmd.append(f"frontend.keyframes.selection.max_normalized_keypoint_drift={job.keyframe_drift}")
    # vidmap.run 是单进程两阶段;日志出现 "Mapper inputs ready" 即进入 mapping
    watcher = threading.Thread(target=_watch_mapping_stage, args=(job,), daemon=True)
    watcher.start()
    run_cmd(job, cmd, cwd=VIDMAP_DIR, gpu=job.gpu)

    if not (rec_dir / "images.bin").exists():
        raise RuntimeError("VidMap 跑完但没有产出 rec/images.bin,建图失败")
    # 从日志里捞注册帧数
    tail = "\n".join(read_log_tail(job, 30))
    m = re.search(r"Reconstruction complete with (\d+)/(\d+) registered", tail)
    if m:
        job.num_registered, job.num_keyframes = int(m.group(1)), int(m.group(2))

    # 2. rec → rec.npz
    set_stage(job, "export")
    run_cmd(job, [VIDMAP_PY, str(REPO_DIR / "export_rec_npz.py"), str(rec_dir), str(job.dir / "rec.npz")],
            cwd=VIDMAP_DIR)

    # 3. rec.npz + 抽帧 → 定位地图 npz
    set_stage(job, "map_npz")
    frames_dirs = [d for d in (outdir / "video_frames").glob("*") if d.is_dir()]
    if not frames_dirs:
        raise RuntimeError("找不到 video_frames 抽帧目录")
    map_npz = OUTPUT_DIR / f"{job.scene}_map.npz"
    npz_env = {"PYTHONPATH": LINGBOT_MAP_DIR + ":" + os.environ.get("PYTHONPATH", "")} if LINGBOT_MAP_DIR else None
    run_cmd(job, [MAP_PY, str(REPO_DIR / "make_map_npz_generic.py"),
                  "--rec_npz", str(job.dir / "rec.npz"),
                  "--frames_dir", str(frames_dirs[0]),
                  "--out", str(map_npz)],
            cwd=REPO_DIR, extra_env=npz_env)

    # 4. 建 DINO 检索库
    set_stage(job, "db")
    db_npz = OUTPUT_DIR / f"{job.scene}_db.npz"
    run_cmd(job, [MAP_PY, str(REPO_DIR / "build_map_db.py"),
                  "--map_npz", str(map_npz), "--out", str(db_npz)],
            cwd=REPO_DIR, gpu=job.gpu,
            extra_env={"HF_ENDPOINT": "https://hf-mirror.com"})

    # 5. 可选: 3D 可视化 HTML
    if job.make_html:
        set_stage(job, "html")
        try:
            run_cmd(job, [VIDMAP_PY, "-m", "vidmap.visualization.html",
                          "--rec", str(rec_dir), "--images", str(frames_dirs[0]),
                          "--include-cameras", "-o", str(job.dir / "map_3d.html")],
                    cwd=VIDMAP_DIR)
        except Exception as e:
            log_msg(job, f"HTML 生成失败(不阻塞): {e}")

    # 打包 rec/ 方便下载
    with zipfile.ZipFile(job.dir / "rec.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for p in rec_dir.iterdir():
            zf.write(p, p.name)

    job.status = "done"
    job.finished_at = time.time()


def _watch_mapping_stage(job: Job):
    """盯着 vidmap.run 日志, frontend 完成时把状态切到 vidmap_mapping。"""
    log = job.dir / "run.log"
    while job.status == "vidmap_frontend":
        if log.exists() and "Mapper inputs ready" in log.read_text(errors="replace"):
            with LOCK:
                if job.status == "vidmap_frontend":
                    job.status = "vidmap_mapping"
            return
        time.sleep(10)


def worker():
    while True:
        job_id = JOB_QUEUE.get()
        job = JOBS[job_id]
        try:
            run_pipeline(job)
        except Exception as e:
            job.status = "failed"
            job.error = f"{e}\n{traceback.format_exc(limit=3)}"
            job.finished_at = time.time()
            log_msg(job, f"任务失败: {e}")
        finally:
            JOB_QUEUE.task_done()


# ---------------- API ----------------

app = FastAPI(title="离线建图 API", version="1.0")


@app.on_event("startup")
def _startup():
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()


@app.get("/health")
def health():
    running = [j for j in JOBS.values() if j.status not in ("done", "failed")]
    return {"status": "ok", "queued": JOB_QUEUE.qsize(),
            "current": running[0].job_id if running else None}


@app.post("/map/build")
async def build(file: UploadFile = File(...),
                scene: Optional[str] = Form(None),
                keyframe_drift: Optional[float] = Form(None),
                make_html: bool = Form(True)):
    """上传视频,提交建图任务。scene 默认取文件名;keyframe_drift 调小=关键帧更密(默认 0.11)。"""
    suffix = Path(file.filename).suffix
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(400, f"不支持的视频格式 {suffix},支持: {sorted(VIDEO_SUFFIXES)}")
    try:
        scene_name = sanitize_scene(scene or Path(file.filename).stem)
    except ValueError as e:
        raise HTTPException(400, str(e))

    job = Job(job_id=time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6],
              scene=scene_name, video_name=file.filename,
              keyframe_drift=keyframe_drift, make_html=make_html)
    job.dir.mkdir(parents=True, exist_ok=False)

    # 流式写盘,视频可能几百 MB
    dst = job.dir / ("input" + suffix)
    with open(dst, "wb") as f:
        while chunk := await file.read(8 * 1024 * 1024):
            f.write(chunk)
    log_msg(job, f"视频已接收: {file.filename} ({dst.stat().st_size / 1e6:.0f} MB), 场景名: {scene_name}")

    with LOCK:
        JOBS[job.job_id] = job
    JOB_QUEUE.put(job.job_id)
    return {"job_id": job.job_id, "scene": scene_name,
            "status_url": f"/map/status/{job.job_id}",
            "预计耗时": "30~60 分钟(取决于视频长度),轮询 status_url"}


@app.get("/map/status/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job_id 不存在")
    d = job.public()
    if job.status == "done":
        d["artifacts"] = {
            "map_npz": f"/map/download/{job_id}/map_npz",
            "db_npz": f"/map/download/{job_id}/db_npz",
            "rec_zip": f"/map/download/{job_id}/rec_zip",
            "html": f"/map/download/{job_id}/html" if (job.dir / "map_3d.html").exists() else None,
            "log": f"/map/download/{job_id}/log",
        }
        d["registered_to"] = [str(OUTPUT_DIR / f"{job.scene}_map.npz"), str(OUTPUT_DIR / f"{job.scene}_db.npz")]
    return d


@app.get("/map/jobs")
def list_jobs():
    return {jid: {"scene": j.scene, "status": j.status, "video": j.video_name,
                  "created_at": time.strftime("%m-%d %H:%M:%S", time.localtime(j.created_at))}
            for jid, j in sorted(JOBS.items(), key=lambda kv: kv[1].created_at, reverse=True)}


@app.get("/map/download/{job_id}/{artifact}")
def download(job_id: str, artifact: str):
    job = JOBS.get(job_id)
    if not job or job.status != "done":
        raise HTTPException(404, "任务不存在或未完成")
    targets = {
        "map_npz": OUTPUT_DIR / f"{job.scene}_map.npz",
        "db_npz": OUTPUT_DIR / f"{job.scene}_db.npz",
        "rec_zip": job.dir / "rec.zip",
        "html": job.dir / "map_3d.html",
        "log": job.dir / "run.log",
    }
    path = targets.get(artifact)
    if path is None:
        raise HTTPException(400, f"未知产物 {artifact},可选: {sorted(targets)}")
    if not path.exists():
        raise HTTPException(404, f"产物文件不存在: {path.name}")
    return FileResponse(path, filename=path.name)


@app.delete("/map/{job_id}")
def delete_job(job_id: str):
    """删除任务目录(视频/缓存/日志)。注意: 已注册的 <scene>_map/db.npz 不会删。"""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job_id 不存在")
    if job.status not in ("done", "failed"):
        raise HTTPException(409, "任务还在跑,不能删")
    shutil.rmtree(job.dir, ignore_errors=True)
    return {"deleted": job_id, "registered_maps_kept": [f"{job.scene}_map.npz", f"{job.scene}_db.npz"]}


def main():
    parser = argparse.ArgumentParser(description="离线建图 API 服务(VidMap)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8200)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
