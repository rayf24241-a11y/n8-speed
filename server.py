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
import os
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

OUTPUT_DIR = Path("/app/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="N8 Speed")

# Every real, billable endpoint sits behind this -- otherwise the RunPod proxy
# URL alone is the entire access control, and anyone who finds it can queue
# unlimited paid-tier generations for free. Deliberately fail CLOSED: if
# N8_SPEED_API_KEY isn't set on the pod, every gated request is refused
# rather than silently left open, so a missed env var can't quietly disable
# auth the way it could with a fail-open check. /health stays ungated -- it
# leaks no capability and needs to work for external uptime monitoring.
API_KEY = os.environ.get("N8_SPEED_API_KEY")


def require_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    if not API_KEY:
        raise HTTPException(500, "Server misconfigured: N8_SPEED_API_KEY not set")
    if x_api_key != API_KEY:
        raise HTTPException(401, "Missing or invalid X-API-Key header")

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


print("Loading content reviewer (moondream2)...", flush=True)
from transformers import AutoModelForCausalLM  # noqa: E402

# The keyword filter and NSFW classifier above can only catch what they were
# specifically built for -- neither has any concept of "is there a chair in
# this shot" or "is the whole body visible". Every one of those composition
# bugs this session (chair, legless man, headless monkey) got caught by a
# human testing it and reported back one at a time. moondream2 is a small
# (~2B) self-hosted vision-language model -- actual reasoning over the
# image, not a narrow classifier -- that can be ASKED about arbitrary
# problems instead of needing a dedicated model trained per problem. Runs
# self-hosted on this same GPU (no external API key), ~2GB VRAM, sub-second
# per query.
#
# Confirmed live: loading this with device_map={"": "cuda"} crashed the
# ENTIRE server at import time -- "AttributeError: 'HfMoondream' object has
# no attribute 'all_tied_weights_keys'", a version mismatch between
# moondream2's custom modeling code and transformers' device-map loading
# path (caching_allocator_warmup expects a newer interface the custom code
# doesn't implement). Loading without device_map and moving to cuda
# afterward -- the same pattern already used for every other model in this
# file -- skips that code path entirely. Also wrapped in try/except: this
# is an optional add-on layer, and a load failure here (this one or any
# future transformers/model version mismatch) must never be able to take
# the whole service down the way it just did. reviewer_model is None when
# unavailable; _ai_review_image() no-ops in that case.
reviewer_model = None
try:
    reviewer_model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2", revision="2025-06-21", trust_remote_code=True
    ).to("cuda")
    print("  -> content reviewer loaded", flush=True)
except Exception:  # noqa: BLE001
    traceback.print_exc()
    print("  -> content reviewer failed to load; continuing without it", flush=True)

REVIEWER_QUESTION = (
    "Look at this image carefully and check for problems. Answer with exactly "
    "one word first -- PASS or FAIL -- then a short reason on the same line. "
    "FAIL if ANY of these are true: the image contains nudity, sexual content, "
    "or graphic violence; there is more than one person or animal subject; the "
    "subject's body is cut off and not fully visible from head to feet; there "
    "is a chair, furniture, or any other object besides the single subject. "
    "Otherwise answer PASS."
)


def _ai_review_image(image: Image.Image):
    """
    Runs AFTER the keyword filter and NSFW classifier, not instead of them --
    this is a generalist model, not a substitute for a dedicated classifier
    on the highest-stakes (safety) check. Fails open on an unexpected error
    (including the model never having loaded in the first place): an issue
    in this quality layer shouldn't take down generation entirely when the
    hard safety gates above already ran.
    """
    if reviewer_model is None:
        return
    try:
        answer = reviewer_model.query(image, REVIEWER_QUESTION)["answer"]
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return
    if answer.strip().upper().startswith("FAIL"):
        raise ValueError(f"Image flagged by content reviewer: {answer.strip()}")


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
#
# Confirmed live and reproducible: the literal prompt "a man" reliably
# generated a crouching figure reaching toward a box, hands fused into one
# blob where they met near the ground. Two contributing causes, both from
# the 2D reference image, not the shape model: (1) nothing biased the pose
# toward standing, so SDXL-Turbo picked a crouch-and-reach pose where the
# two hands overlap in the 2D image -- the shape model then has no depth
# information to separate what already reads as one shape; (2) hands are
# small, fine detail exactly like faces above, so the same blur/garbling
# problem applies to fingers. Same fix as the face case: bias toward a pose
# and rendering where hands are unambiguous. Note guidance_scale=0.0 below
# means the model doesn't follow this prompt tightly (that's Turbo's
# intended, recommended setting), so this reduces the failure rate, it
# doesn't eliminate it -- fused hands are a known hard limitation of
# single-image 3D reconstruction generally, not something a prompt alone
# fully solves.
TEXT_TO_IMAGE_STYLE_SUFFIX = (
    ", single isolated object, centered, plain white background, no floor, "
    "no shadow, no ground, studio product photography, clean background, "
    "no props, no furniture, no accessories, no other objects, nothing else in frame, "
    "full body, full-length, entire body visible from head to feet, "
    "standing upright, arms relaxed at sides, hands away from each other and away from the body, "
    "hands and fingers clearly separated and distinct, not touching, not overlapping, not fused together, "
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


MIN_TARGET_FACES = 1000
MAX_TARGET_FACES = 200000


class GenerateRequest(BaseModel):
    prompt: Optional[str] = None
    image_b64: Optional[str] = None
    texture: bool = False
    # Caps the exported mesh's triangle count via post-generation decimation
    # (does not change shape-generation cost -- octree_resolution is what
    # drives that). None = no cap, keep the raw marching-cubes output.
    target_faces: Optional[int] = None


OUTPUT_MAX_AGE_SECONDS = 24 * 60 * 60


def _sweep_old_outputs():
    # Nothing ever deleted these -- every generated .glb sat on the pod's
    # disk forever. Runs once per job (cheap: a single directory listing)
    # instead of on a separate timer, since the worker thread is already the
    # one serialization point for everything this process does.
    cutoff = time.time() - OUTPUT_MAX_AGE_SECONDS
    for f in OUTPUT_DIR.glob("*.glb"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def _run_job(job_id: str):
    _sweep_old_outputs()
    job = jobs[job_id]
    req: GenerateRequest = job["request"]

    image_seconds = 0.0
    if req.image_b64:
        image = Image.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
    elif req.prompt:
        t_img = time.time()
        full_prompt = req.prompt + TEXT_TO_IMAGE_STYLE_SUFFIX
        # Raised from 4: same reasoning as the earlier 2->4 bump, extended
        # to hands -- confirmed live that fused-hands failures trace back to
        # small/blurry finger detail in the 2D reference image, the same
        # step-starvation problem faces had. Each step is still well under
        # 1s, so this adds a small fraction of a second, not a real cost.
        image = txt2img(prompt=full_prompt, num_inference_steps=6, guidance_scale=0.0).images[0]
        image_seconds = time.time() - t_img
    else:
        raise ValueError("Provide either 'prompt' or 'image_b64'")

    # Covers both paths -- an uploaded image never went through the prompt
    # keyword filter in /generate at all, and a text prompt that dodges
    # that filter still has to produce an actual image, which lands here.
    _check_image_safety(image)

    t_review = time.time()
    _ai_review_image(image)
    review_seconds = time.time() - t_review

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
        # Raised from 25: confirmed live the diffusion loop itself is only
        # ~2.4s of an ~8s shape_seconds total (logged as "Diffusion
        # Sampling:: 25/25" completing in ~2s), so step count was never the
        # dominant cost -- this mainly buys denoising stability, not the
        # facial-distortion fix below.
        num_inference_steps=35,
        # Raised from 256 -- explicitly requested higher quality ("faces
        # look distorted") at a 1.5x time budget. octree_resolution (marching
        # cubes voxel grid density), not step count, is what actually
        # determines whether fine geometry like facial features gets
        # captured or comes out blobby -- confirmed by where the time
        # actually goes (see note above).
        #
        # NOT going back to 384: confirmed live that setting is UNPREDICTABLE,
        # not just slower -- since we never pin a seed, each SDXL-Turbo image
        # is different, and Hunyuan3D-2's octree-based adaptive refinement
        # scales with surface complexity. One "a cup" call at 384 measured
        # shape_seconds=24.5s, another measured 299.7s (12x) with no code
        # change, just a harder random shape. 300 is a deliberately
        # moderate step up from the fast/consistent 256 baseline, estimated
        # (from the 256->384 scaling actually observed) to land near 1.5x
        # average total time -- an estimate, not a guarantee, given the
        # same variance risk applies at any resolution above 256, just
        # less severely. Watch real shape_seconds numbers and adjust.
        octree_resolution=300,
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

    # User-requested polycount cap. Runs before the texture-only 40k safety
    # cap below (that one exists purely to keep xatlas fast, not for quality
    # control) so a request already at or under 40k also speeds up texturing
    # for free instead of decimating twice.
    decimate_seconds = 0.0
    if req.target_faces and len(mesh.faces) > req.target_faces:
        t_decimate = time.time()
        mesh = mesh.simplify_quadric_decimation(face_count=req.target_faces)
        decimate_seconds = time.time() - t_decimate

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
        job["review_seconds"] = round(review_seconds, 2)
        job["rembg_seconds"] = round(rembg_seconds, 2)
        job["shape_seconds"] = round(shape_seconds, 2)
        job["cleanup_seconds"] = round(cleanup_seconds, 2)
        job["decimate_seconds"] = round(decimate_seconds, 2)
        job["simplify_seconds"] = round(simplify_seconds, 2)
        job["texture_seconds"] = round(texture_seconds, 2)
        job["export_seconds"] = round(export_seconds, 2)
        job["face_count"] = len(mesh.faces)
        job["total_seconds"] = round(
            image_seconds + review_seconds + rembg_seconds + shape_seconds
            + cleanup_seconds + decimate_seconds + simplify_seconds + texture_seconds + export_seconds, 2
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


@app.post("/generate", dependencies=[Depends(require_api_key)])
def generate(req: GenerateRequest):
    if not req.prompt and not req.image_b64:
        raise HTTPException(400, "Provide 'prompt' or 'image_b64'")
    if req.prompt and _prompt_is_flagged(req.prompt):
        raise HTTPException(400, "Prompt rejected: inappropriate content is not allowed.")
    if req.target_faces is not None and not (MIN_TARGET_FACES <= req.target_faces <= MAX_TARGET_FACES):
        raise HTTPException(
            400, f"target_faces must be between {MIN_TARGET_FACES} and {MAX_TARGET_FACES}"
        )

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


@app.get("/status/{job_id}", dependencies=[Depends(require_api_key)])
def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    resp = {"status": job["status"]}
    if job["status"] == "done":
        resp["image_seconds"] = job["image_seconds"]
        resp["review_seconds"] = job["review_seconds"]
        resp["rembg_seconds"] = job["rembg_seconds"]
        resp["shape_seconds"] = job["shape_seconds"]
        resp["cleanup_seconds"] = job["cleanup_seconds"]
        resp["decimate_seconds"] = job["decimate_seconds"]
        resp["simplify_seconds"] = job["simplify_seconds"]
        resp["texture_seconds"] = job["texture_seconds"]
        resp["export_seconds"] = job["export_seconds"]
        resp["face_count"] = job["face_count"]
        resp["total_seconds"] = job["total_seconds"]
    if job["status"] == "error":
        resp["error"] = job["error"]
    return resp


@app.get("/result/{job_id}", dependencies=[Depends(require_api_key)])
def result(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "not ready")
    return FileResponse(job["result_path"], media_type="model/gltf-binary", filename=f"{job_id}.glb")


@app.get("/health")
def health():
    return {"status": "ok"}
