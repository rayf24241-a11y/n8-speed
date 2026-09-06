"""
N8 Speed inference server.

One GPU, one worker thread -> all generation jobs are serialized through a
single queue.Queue. That matches a single-replica Salad container exactly:
there is only one 5090 to hand work to, so a thread pool would just add
complexity without adding real concurrency. When you add more GPU workers
later, put a load balancer / router in front of N replicas of this same
container instead of adding concurrency inside one of them.
"""
import base64
import io
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

OUTPUT_DIR = Path("/app/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="N8 Speed")

# ---------------------------------------------------------------------------
# Model loading (once, at process startup)
# ---------------------------------------------------------------------------

print("Loading text-to-image model (SDXL-Turbo, 1-step)...", flush=True)
from diffusers import AutoPipelineForText2Image  # noqa: E402

txt2img = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
).to("cuda")


def _load_shape_pipeline():
    """
    Try the fastest checkpoint first, fall back to slower-but-known-good ones.
    The exact subfolder name for the mini-turbo checkpoint can shift between
    Hunyuan3D-2 releases -- this fallback chain means a rename upstream
    degrades speed instead of hard-crashing the server.
    """
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    candidates = [
        ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini-turbo"),
        ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0-turbo"),
        ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0"),
    ]
    last_err = None
    for repo_id, subfolder in candidates:
        try:
            print(f"Loading shape model {repo_id}/{subfolder} ...", flush=True)
            pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(repo_id, subfolder=subfolder)
            print(f"  -> loaded {repo_id}/{subfolder}", flush=True)
            # Confirmed on a real deploy: this pipeline's .to() mutates in
            # place and returns None (unlike torch.nn.Module.to(), which
            # returns self) -- `return pipe.to("cuda")` was silently
            # returning None, so every /generate call failed with
            # "'NoneType' object is not callable" at the shape_pipeline(...)
            # call site, despite the startup logs claiming success.
            pipe.to("cuda")
            return pipe
        except Exception as e:  # noqa: BLE001
            print(f"  -> failed ({e}); trying next candidate", flush=True)
            last_err = e
    raise RuntimeError(f"Could not load any Hunyuan3D shape checkpoint: {last_err}")


shape_pipeline = _load_shape_pipeline()

print("Loading texture (paint) pipeline...", flush=True)
from hy3dgen.texgen import Hunyuan3DPaintPipeline  # noqa: E402

paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")

print("N8 Speed ready.", flush=True)

# ---------------------------------------------------------------------------
# Single-worker job queue
# ---------------------------------------------------------------------------

job_queue: "queue.Queue[str]" = queue.Queue()
jobs: dict = {}
jobs_lock = threading.Lock()


class GenerateRequest(BaseModel):
    prompt: Optional[str] = None
    image_b64: Optional[str] = None
    texture: bool = False


def _run_job(job_id: str):
    job = jobs[job_id]
    req: GenerateRequest = job["request"]

    image_seconds = 0.0
    if req.image_b64:
        image = Image.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
    elif req.prompt:
        t_img = time.time()
        image = txt2img(prompt=req.prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        image_seconds = time.time() - t_img
    else:
        raise ValueError("Provide either 'prompt' or 'image_b64'")

    # Split out from the export step below -- previously this was the only
    # thing timed, so a slow mesh.export() (marching cubes / octree
    # extraction, not diffusion) would have been invisible in shape_seconds.
    t0 = time.time()
    # Hunyuan3DDiTFlowMatchingPipeline defaults to num_inference_steps=50 --
    # fine for the base checkpoint, but defeats the whole point of the
    # mini-turbo checkpoint (few-step distilled, designed for ~5 steps).
    # Starting guess; tune against the real shape_seconds this logs.
    mesh = shape_pipeline(image=image, num_inference_steps=5)[0]
    shape_seconds = time.time() - t0

    texture_seconds = 0.0
    if req.texture:
        t1 = time.time()
        mesh = paint_pipeline(mesh, image=image)
        texture_seconds = time.time() - t1

    t2 = time.time()
    out_path = OUTPUT_DIR / f"{job_id}.glb"
    mesh.export(str(out_path))
    export_seconds = time.time() - t2

    with jobs_lock:
        job["status"] = "done"
        job["result_path"] = str(out_path)
        job["image_seconds"] = round(image_seconds, 2)
        job["shape_seconds"] = round(shape_seconds, 2)
        job["texture_seconds"] = round(texture_seconds, 2)
        job["export_seconds"] = round(export_seconds, 2)
        job["total_seconds"] = round(image_seconds + shape_seconds + texture_seconds + export_seconds, 2)


def worker_loop():
    while True:
        job_id = job_queue.get()
        with jobs_lock:
            jobs[job_id]["status"] = "processing"
        try:
            _run_job(job_id)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            with jobs_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = str(e)
        job_queue.task_done()


threading.Thread(target=worker_loop, daemon=True).start()

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@app.post("/generate")
def generate(req: GenerateRequest):
    if not req.prompt and not req.image_b64:
        raise HTTPException(400, "Provide 'prompt' or 'image_b64'")

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "request": req}
    job_queue.put(job_id)

    position = job_queue.qsize()
    message = (
        "A lot of people are using N8 Speed right now. This might take longer."
        if position > 1
        else "Generating..."
    )
    return {"job_id": job_id, "queue_position": position, "message": message}


@app.get("/status/{job_id}")
def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    resp = {"status": job["status"]}
    if job["status"] == "done":
        resp["image_seconds"] = job["image_seconds"]
        resp["shape_seconds"] = job["shape_seconds"]
        resp["texture_seconds"] = job["texture_seconds"]
        resp["export_seconds"] = job["export_seconds"]
        resp["total_seconds"] = job["total_seconds"]
    if job["status"] == "error":
        resp["error"] = job["error"]
    return resp


@app.get("/result/{job_id}")
def result(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "not ready")
    return FileResponse(job["result_path"], media_type="model/gltf-binary", filename=f"{job_id}.glb")


@app.get("/health")
def health():
    return {"status": "ok"}
