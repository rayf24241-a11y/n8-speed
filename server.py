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

print("Loading text-to-image model (SDXL-Turbo)...", flush=True)
from diffusers import AutoPipelineForText2Image  # noqa: E402

txt2img = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

print("Loading background remover...", flush=True)
from hy3dgen.rembg import BackgroundRemover  # noqa: E402

# Confirmed against Hunyuan3D-2's own gradio_app.py: they run every RGB
# (non-alpha) input image through this before shape generation, always --
# not an optional extra. Skipping it is exactly why an uploaded photo's
# floor/background was getting reconstructed as 3D geometry: the shape
# pipeline has no "subject vs. environment" concept of its own, it just
# needs an image where the background is already gone.
rmbg = BackgroundRemover()

print("Loading content-safety classifier...", flush=True)
from transformers import pipeline as hf_pipeline  # noqa: E402

# A keyword check on the prompt text alone can't cover the /image_b64
# path at all -- a user can just upload inappropriate content directly,
# no prompt involved. This classifies the actual image (uploaded OR
# generated) before any GPU time is spent on the 3D pipeline. Small ViT
# model, self-hosted like everything else here (no external API key),
# negligible added VRAM/latency on top of what's already loaded.
nsfw_classifier = hf_pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=0)
NSFW_CONFIDENCE_THRESHOLD = 0.85


def _check_image_safety(image: Image.Image):
    result = max(nsfw_classifier(image), key=lambda r: r["score"])
    if result["label"] == "nsfw" and result["score"] >= NSFW_CONFIDENCE_THRESHOLD:
        raise ValueError("Image flagged as inappropriate -- request rejected.")


# First-pass, zero-GPU-cost filter on the prompt text itself -- rejects
# obvious intent immediately via HTTP 400 instead of queuing a job that
# would just get caught downstream anyway. Confirmed live: this needs
# named-object terms too, not just anatomy/act words -- a prompt asking
# for a sex toy by name generates an image of exactly that object on a
# plain background, which isn't nudity and isn't guaranteed to trip an
# NSFW-photo classifier trained mainly on explicit imagery. The keyword
# list is the actual defense for this category; _check_image_safety()
# below is a backstop for nudity-style content, not a substitute for it.
BANNED_PROMPT_TERMS = (
    "nude", "naked", "nsfw", "porn", "sexual", "genitals", "genitalia",
    "penis", "vagina", "breasts", "nipple", "loli", "shota",
    "bestiality", "gore", "decapitat", "mutilat", "corpse", "dead body",
    "dildo", "vibrator", "sex toy", "fleshlight", "buttplug", "butt plug",
    "strap-on", "strap on", "cock ring", "anal plug",
)


def _prompt_is_flagged(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(term in lowered for term in BANNED_PROMPT_TERMS)


# Confirmed live: a plain prompt like "a dog" makes SDXL-Turbo generate a
# natural photo -- dog standing on visible ground/floor, background context,
# shadow -- and the shape pipeline then reconstructs 3D geometry for
# *everything* in the image, floor included, since it has no concept of
# "subject" vs "environment". Bias toward an isolated single object instead,
# the same technique already used for prompt suffixing in
# Hunyuan3D-Output\Generate-Model.ps1's GtagStyleSuffix.
#
# Confirmed live again: "a man" with only the floor/background suffix still
# generated a man sitting in a chair -- rembg doesn't strip that, since a
# chair someone is sitting on is contiguous with the foreground subject in
# the alpha mask, not "background" in any sense it can detect. Same for the
# largest-connected-component mesh cleanup: a chair fused to a seated
# figure is one connected component, not separable debris. The only real
# fix is stopping the prop from being generated in the first place.
#
# Confirmed live a third time: "a man" generated a legless result -- the
# "studio product photography, centered" framing biases a PERSON subject
# toward a head-and-shoulders portrait crop, so the 2D image itself likely
# never had legs in frame in the first place; the shape model can't
# reconstruct what it never saw. Added explicit full-body framing language.
# Faces are the hardest thing for both stages to get right: SDXL-Turbo at
# very few steps tends to blur or garble fine facial detail in the source
# image, and the shape model can only reconstruct 3D geometry for detail
# that was actually legible in that 2D image. Biasing the prompt toward a
# sharp, well-defined face gives the shape model something real to work
# from instead of a blurry approximation.
TEXT_TO_IMAGE_STYLE_SUFFIX = (
    ", single isolated object, centered, plain white background, no floor, "
    "no shadow, no ground, studio product photography, clean background, "
    "no props, no furniture, no accessories, no other objects, nothing else in frame, "
    "full body, full-length, entire body visible from head to feet, "
    "detailed face, sharp facial features, clear eyes, well-defined face, high detail"
)


def _load_shape_pipeline():
    """
    Try the best-quality checkpoint we have time budget for first, fall back
    to smaller/older ones if it's missing. The exact subfolder name can
    shift between Hunyuan3D-2 releases -- this fallback chain means a
    rename upstream degrades quality instead of hard-crashing the server.
    """
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    candidates = [
        # Confirmed live: the 0.6B mini-turbo checkpoint (previously first
        # here, chosen purely for speed) produced a cat with a head as big
        # as its body and misplaced legs -- a real anatomy/proportion
        # failure, not a step-count artifact (already at 25 steps). The
        # full 1.1B turbo checkpoint is a materially bigger model with the
        # same "turbo" few-step distillation, so it's not a speed cliff --
        # only ~2x shape_seconds (measured ~8s -> expect ~16s), comfortably
        # inside the ~30s budget, for meaningfully better anatomy.
        ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0-turbo"),
        ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini-turbo"),
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


def _load_paint_pipeline():
    """
    Same fallback-chain idea as the shape loader. Confirmed against
    Hunyuan3D-2's own examples/fast_texture_gen_multiview.py: the turbo
    checkpoint needs its subfolder specified explicitly -- from_pretrained()
    with no subfolder silently loads the slow, non-distilled base model
    instead. That's a real, credible reason a texture job could genuinely
    take many minutes (not necessarily a hang): we were never running the
    fast checkpoint in the first place.
    """
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    candidates = ["hunyuan3d-paint-v2-0-turbo", "hunyuan3d-paint-v2-0"]
    last_err = None
    for subfolder in candidates:
        try:
            print(f"Loading paint model tencent/Hunyuan3D-2/{subfolder} ...", flush=True)
            pipe = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2", subfolder=subfolder)
            print(f"  -> loaded {subfolder}", flush=True)
            return pipe
        except Exception as e:  # noqa: BLE001
            print(f"  -> failed ({e}); trying next candidate", flush=True)
            last_err = e
    raise RuntimeError(f"Could not load any Hunyuan3D paint checkpoint: {last_err}")


paint_pipeline = _load_paint_pipeline()

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
        full_prompt = req.prompt + TEXT_TO_IMAGE_STYLE_SUFFIX
        # Raised from 2: SDXL-Turbo supports up to ~4 steps at
        # guidance_scale=0 and each step is cheap (well under 1s total
        # either way). Faces are the most step-starved detail in a fast
        # generation -- more steps means less blur/garbling in exactly the
        # feature the shape model most needs a clean source image for.
        image = txt2img(prompt=full_prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        image_seconds = time.time() - t_img
    else:
        raise ValueError("Provide either 'prompt' or 'image_b64'")

    # Covers both paths -- an uploaded image never went through the prompt
    # keyword filter in /generate at all, and a text prompt that dodges
    # that filter still has to produce an actual image, which lands here.
    _check_image_safety(image)

    # Confirmed against Hunyuan3D-2's own gradio_app.py: it runs every RGB
    # image through background removal before shape generation, unconditionally.
    # Applies to BOTH paths -- an uploaded photo has a real background to
    # strip, and our own SDXL-Turbo image benefits too even with the prompt
    # suffix biasing it toward a plain background already.
    t_rmbg = time.time()
    image_nobg = rmbg(image.convert("RGB"))
    rembg_seconds = time.time() - t_rmbg

    # NOTE: shape_seconds times the whole shape_pipeline() call, which
    # internally does denoising AND mesh extraction (VAE decode + marching
    # cubes via vae.latents2mesh()) in one black box -- export_seconds below
    # only covers OUR mesh.export() glb write, not that internal extraction.
    # So this can't separate "slow diffusion" from "slow mesh extraction"
    # on its own; see the two params tuned below for that instead.
    t0 = time.time()
    mesh = shape_pipeline(
        image=image_nobg,
        # Raised from 12: with rembg now handling background removal
        # properly (rather than relying on the prompt suffix alone) and a
        # ~30s quality budget instead of ~10s, there's room to push detail
        # further. Not a measured number -- shape_seconds bundles a step-
        # dependent diffusion cost with a step-independent mesh-extraction
        # cost that can't be separated from here. Tune against the real
        # number this logs.
        num_inference_steps=25,
        # Lowered back from 384 (Hunyuan3D-2's own default) to 256. 384
        # isn't just slower on average, it's UNPREDICTABLE: confirmed live
        # that since we never pin a seed, each SDXL-Turbo image is
        # different, and Hunyuan3D-2's octree-based adaptive mesh
        # refinement scales with surface complexity -- one "a cup" call
        # measured shape_seconds=24.5s, another measured 299.7s (12x) with
        # no code change, just a harder random shape to refine. 256 was
        # fast and consistent across every test run (no outliers seen),
        # trading some mesh detail for an actual time ceiling instead of
        # an average.
        octree_resolution=256,
    )[0]
    shape_seconds = time.time() - t0

    # Mesh cleanup: marching-cubes extraction at this resolution leaves
    # small floating fragments disconnected from the main body (visible as
    # scattered debris around generated models). Keep only the largest
    # connected component -- the actual object -- and drop the rest.
    t_clean = time.time()
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda m: len(m.vertices))
    cleanup_seconds = time.time() - t_clean

    simplify_seconds = 0.0
    texture_seconds = 0.0
    if req.texture:
        # Confirmed live: the paint pipeline's very first step is
        # hy3dgen.texgen.utils.uv_warp_utils.mesh_uv_wrap(), which calls
        # xatlas.parametrize() -- a synchronous, single-threaded C++ call
        # that prints zero progress and holds the GIL for its entire
        # duration. UV atlas packing cost scales hard with face count, and
        # raising octree_resolution to 384 this session produced a dense
        # enough mesh that this call ran long enough to freeze the whole
        # process -- including /health -- for 10+ minutes with no error and
        # no log line, exactly matching the original "hang". Not a checkpoint
        # issue (turbo vs base share this same code path).
        #
        # Fix: decimate before texturing. Texture detail comes from the UV
        # texture map, not mesh density, so a lower-poly mesh for texturing
        # than for shape is the standard game/VFX pipeline tradeoff anyway,
        # not just a workaround. 40k faces is comfortably inside xatlas's
        # fast range regardless of how dense the shape output was.
        t_simplify = time.time()
        if len(mesh.faces) > 40000:
            mesh = mesh.simplify_quadric_decimation(face_count=40000)
        simplify_seconds = time.time() - t_simplify

        t1 = time.time()
        mesh = paint_pipeline(mesh, image=image_nobg)
        texture_seconds = time.time() - t1

    t2 = time.time()
    out_path = OUTPUT_DIR / f"{job_id}.glb"
    mesh.export(str(out_path))
    export_seconds = time.time() - t2

    with jobs_lock:
        job["status"] = "done"
        job["result_path"] = str(out_path)
        job["image_seconds"] = round(image_seconds, 2)
        job["rembg_seconds"] = round(rembg_seconds, 2)
        job["shape_seconds"] = round(shape_seconds, 2)
        job["cleanup_seconds"] = round(cleanup_seconds, 2)
        job["simplify_seconds"] = round(simplify_seconds, 2)
        job["texture_seconds"] = round(texture_seconds, 2)
        job["export_seconds"] = round(export_seconds, 2)
        job["total_seconds"] = round(
            image_seconds + rembg_seconds + shape_seconds + cleanup_seconds
            + simplify_seconds + texture_seconds + export_seconds, 2
        )


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
    if req.prompt and _prompt_is_flagged(req.prompt):
        raise HTTPException(400, "Prompt rejected: inappropriate content is not allowed.")

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
        resp["rembg_seconds"] = job["rembg_seconds"]
        resp["shape_seconds"] = job["shape_seconds"]
        resp["cleanup_seconds"] = job["cleanup_seconds"]
        resp["simplify_seconds"] = job["simplify_seconds"]
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
