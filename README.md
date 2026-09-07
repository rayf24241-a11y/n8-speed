# N8 Speed

Fast text/image -> 3D mesh, optional texturing, served over HTTP from a
RunPod RTX 4090 pod.

## What's here

- `Dockerfile` -- CUDA 12.4 base, PyTorch cu124 (targets the RTX 4090 we're
  actually running on -- a 12.8 base got flat-out rejected on real RunPod
  hosts whose drivers only support up to 12.4; a 5090 needs a separate
  12.8-based image later, not this one), Hunyuan3D-2 cloned + its compiled
  CUDA extensions (shape AND texture pipelines both run on this GPU). Model
  weights are **not** baked in (see below).
- `server.py` -- FastAPI app: one job queue, one worker thread (matches one
  GPU). `/generate` queues a job, `/status/{id}` polls it, `/result/{id}`
  downloads the `.glb`. Loading the models at process start triggers their
  download from Hugging Face automatically if they're not already cached.
- `.github/workflows/build.yml` -- builds and pushes the image on every push
  to `main`. This exists because building locally hit a wall twice: a
  Docker Desktop bug on the dev machine, and separately, this build is
  simply too large (CUDA base + PyTorch + Hunyuan3D-2, tens of GB once you
  add weights) to fit on a disk-limited machine reliably. Building on
  GitHub's own runners sidesteps both problems.

## Why weights aren't baked into the image

Originally they were. That blew the disk budget on *every* build
environment tried -- the local dev machine's C: drive, and then GitHub
Actions' own runner ("No space left on device", confirmed in that build's
failure logs). Baking in even a trimmed set of checkpoints plus the CUDA
toolchain landed north of 60GB, which nothing here could reliably hold
through a full Docker build.

Instead, `server.py`'s `from_pretrained()` calls download straight from
Hugging Face into `HF_HOME` the first time the container actually starts.

**This makes a persistent volume for `HF_HOME` (`/app/hf-cache`) mandatory
on RunPod, not optional.** Without one, every pod restart re-downloads
~15-20GB of weights from scratch before it can serve a single request --
several minutes of dead time on every restart, not just the first one. With
a volume mounted at `/app/hf-cache`, that download happens exactly once
across the pod's whole lifetime.

## Known risks before the first real run

I don't have a way to hit huggingface.co/tencent/Hunyuan3D-2mini from here to
confirm the *exact* current subfolder name for the turbo checkpoint, and I
can't run CUDA locally to test the pipeline end to end (no NVIDIA GPU on the
dev machine). `server.py` tries three checkpoints in order (mini-turbo ->
base turbo -> base) and logs which one it actually loaded, so a naming drift
degrades speed rather than crashing. **Read the container's startup logs on
first deploy** -- if it fell back past the first candidate, open
https://huggingface.co/tencent/Hunyuan3D-2mini and fix the subfolder string in
`server.py`'s `_load_shape_pipeline()`.

Also unverified: the exact `pipeline(image=image)[0]` call signature matches
the Hunyuan3D-2 README pattern from memory. If the container throws on the
first `/generate` call, the pod's logs will show the real signature error --
it's a one-line fix in `server.py`.

## Texturing: local paint pipeline, background removal, the real hang found

A first real `texture: true` call made the whole process unresponsive for
10+ minutes -- even `/health` stopped answering, with no CUDA OOM logged.
Two things were fixed here, and only the second one turned out to be the
actual cause.

**Checkpoint (real bug, not the hang):** `server.py` was loading the paint
pipeline with `Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")`
and no `subfolder`, silently loading the slow, non-distilled base
checkpoint. Tencent's own `examples/fast_texture_gen_multiview.py` loads it
with `subfolder="hunyuan3d-paint-v2-0-turbo"` instead. Fixed via
`_load_paint_pipeline()`'s fallback chain -- but confirmed live this alone
did NOT fix the hang: with the turbo checkpoint correctly loaded, the exact
same freeze recurred (container logs went completely silent, `/health`
timed out at the network level for 10+ minutes, right as texturing began).

**The actual hang:** `Hunyuan3DPaintPipeline.__call__()`'s first real step
is `mesh_uv_wrap()`, which calls `xatlas.parametrize()` for UV unwrapping --
a synchronous, single-threaded C++ call that prints zero progress and holds
the GIL for its entire duration. UV atlas packing cost scales hard with
face count. This session also raised `octree_resolution` from 256 to 384
for shape quality, producing a dense enough mesh that xatlas silently ran
long enough to freeze the whole process (not just the job -- a GIL-holding
call blocks every thread, including the one serving `/health`). Confirmed
by reading Hunyuan3D-2's own `hy3dgen/texgen/pipelines.py` and
`uv_warp_utils.py` source, and by watching container logs go dead at
exactly that call.

**Fix:** decimate the mesh to 40k faces (via `simplify_quadric_decimation`,
needs the `fast-simplification` pip package) before texturing, only when
`texture: true`. Texture detail comes from the UV texture map, not mesh
density, so a lower-poly mesh for texturing than for shape is the standard
game/VFX pipeline tradeoff anyway, not just a workaround. Reported as
`simplify_seconds` in `/status`.

Separately (and unrelated to either bug above): an uploaded photo's
background was getting reconstructed as 3D geometry too. Confirmed against
Hunyuan3D-2's own `gradio_app.py` -- it unconditionally runs every RGB input
image through `hy3dgen.rembg.BackgroundRemover` before shape generation,
since the shape model has no "subject vs. environment" concept and needs
the background already gone. `server.py` now does the same for every image,
uploaded or generated.

Confirmed live yet again: "a man" generated a man sitting in a chair, and
neither rembg nor the largest-component mesh cleanup touched it -- a chair
someone is sitting on is contiguous with the subject in both the alpha mask
and the mesh topology, not separable background or debris. The only real
fix is stopping the prop from being generated at all: `TEXT_TO_IMAGE_STYLE_SUFFIX`
now also says "no props, no furniture, no accessories, no other objects,
nothing else in frame". This only helps the text-prompt path (SDXL-Turbo
image) -- an uploaded photo with a chair in it has no such backstop.

Confirmed live a third time: "a man" generated a legless result. The
"studio product photography, centered" framing biases a PERSON subject
toward a head-and-shoulders portrait crop -- the 2D image itself likely
never had legs in it, so the shape model had nothing to reconstruct. Added
"full body, full-length, entire body visible from head to feet" to the
suffix.

## Content review: a real reasoning model, not another narrow classifier

Every composition bug this session (chair fused to a seated figure, a
legless man, a headless-crop monkey) was found by manual testing, reported
one at a time, and fixed with a one-off prompt tweak. The keyword filter and
NSFW classifier can only catch what they were specifically built for --
neither has any concept of "is there a chair in this shot" or "is the whole
body visible", so a new failure mode always needs a human to notice it
first.

`_ai_review_image()` in `server.py` adds `vikhyatk/moondream2`, a small
(~2B parameter) self-hosted vision-language model, as a generalist reviewer
run on every image (uploaded or generated) right after the NSFW classifier.
Unlike the classifiers above, it can be *asked* about a problem in plain
language instead of needing a dedicated model trained per problem -- the
same question also checks for extra objects, multiple subjects, and
cut-off bodies, catching a case even if nobody has told the prompt suffix
to avoid it yet. Runs self-hosted on this same GPU: no external API key,
~2GB VRAM, sub-second per query, zero new pip dependencies (only needs
`torch`/`transformers`/`pillow`, all already installed).

This does **not** replace the NSFW classifier or keyword filter -- it's a
generalist model, not a substitute for a dedicated one on the highest-stakes
check. It runs as an additional layer, and fails open on an unexpected
error (an issue in this quality layer shouldn't take down generation
entirely when the hard safety gates already ran). Reported as
`review_seconds` in `/status`.

**Incident, fixed:** the very first deploy of this crashed the ENTIRE
server at import time -- `AttributeError: 'HfMoondream' object has no
attribute 'all_tied_weights_keys'` -- a version mismatch between
moondream2's custom modeling code and transformers' `device_map`-based
loading path (`caching_allocator_warmup` expects a newer interface the
custom code doesn't implement). Not a "the reviewer doesn't work" bug --
the whole container crash-looped and never served a single request. Fixed
two ways: (1) load without `device_map`, `.to("cuda")` afterward instead
-- the same pattern every other model in this file already uses, which
skips the code path that crashed; (2) wrapped the load itself in
try/except, `reviewer_model = None` on failure, `_ai_review_image()`
no-ops when it's `None`. That second part is the actually important fix --
an optional add-on layer must never be able to take the whole service down
again, regardless of whether this specific fix holds up against some
future transformers version too.

**Known limitation, not fixed:** multi-subject prompts like "a girl and a
man" aren't well supported. `TEXT_TO_IMAGE_STYLE_SUFFIX` says "single
isolated object" on purpose -- that's the same bias that stops floor/props
from leaking in -- so a two-person prompt tends to collapse to one figure.
Even if it didn't, this pipeline reconstructs one image into one mesh with
no multi-object segmentation, so two people would get welded into a single
connected 3D blob, not two separate models. There's no good fix for this
within the current one-image-in/one-mesh-out design -- generate each
subject as its own separate `/generate` call instead.

**Face quality:** faces are the hardest detail for both stages -- SDXL-Turbo
at very few steps blurs/garbles fine facial features, and the shape model
can only reconstruct what was legible in that source image. Raised
SDXL-Turbo `num_inference_steps` 2 -> 4 (still well under 1s) and added
"detailed face, sharp facial features, clear eyes, well-defined face, high
detail" to `TEXT_TO_IMAGE_STYLE_SUFFIX`.

Quality settings were also raised: shape `num_inference_steps` 12 -> 25,
SDXL-Turbo 1 -> 2 steps. `octree_resolution` was raised 256 -> 384 and then
**reverted back to 256** -- confirmed live that 384 isn't just slower on
average, it's unpredictable. We never pin a seed for the SDXL-Turbo image,
so every call generates a different shape, and Hunyuan3D-2's octree-based
adaptive mesh refinement scales with surface complexity: one "a cup" call
measured `shape_seconds`=24.5s at 384, another measured 299.7s (12x, no code
change) because that random shape happened to need much deeper refinement.
256 was fast and consistent across every test run -- no outliers seen --
trading some mesh detail for an actual time ceiling instead of an average.
Watch the real `shape_seconds`/`texture_seconds` numbers on deploy and tune
from there -- these are informed estimates, not guaranteed timings.

## Shape checkpoint: anatomy over raw speed

Confirmed live: the 0.6B `hunyuan3d-dit-v2-mini-turbo` checkpoint (the
first candidate `_load_shape_pipeline()` tried, chosen purely for speed)
generated a cat with a head as big as its body and legs in the wrong
place -- a real proportion/anatomy failure, not a step-count artifact
(already running 25 steps). Swapped the candidate order to try the full
1.1B `hunyuan3d-dit-v2-0-turbo` checkpoint first instead -- same "turbo"
few-step distillation family, so it's roughly 2x `shape_seconds` (measured
~8s -> expect ~16s), not a speed cliff, and comfortably inside the ~30s
budget. Falls back to mini-turbo only if the bigger checkpoint is missing.

## octree_resolution 256 -> 300: distorted faces, explicit 1.5x time budget

Reported live: faces were coming out visibly distorted. Confirmed the
diffusion loop itself is only ~2.4s of an ~8s `shape_seconds` total (logged
as "Diffusion Sampling:: 25/25" finishing in ~2s) -- step count was never
the dominant cost or the thing controlling facial detail.
`octree_resolution` (marching-cubes voxel grid density) is what actually
determines whether fine geometry like facial features gets captured or
comes out blobby. Also raised `num_inference_steps` 25 -> 35 for extra
denoising stability, though that's a secondary lever.

Deliberately did NOT go back to 384 (see the section above -- that
setting is unpredictable, not just slower, with a confirmed 12x outlier).
300 is a moderate step up from the fast/consistent 256 baseline, estimated
from the 256->384 scaling actually observed to land near the requested
1.5x average total time -- an estimate, not a guarantee, since the same
variance risk applies at any resolution above 256, just less severely at
300 than at 384. Watch real `shape_seconds` numbers on deploy.

## Content safety

Two layers, since a keyword filter on the prompt text can't do anything
about the `image_b64` path -- a user can upload inappropriate content
directly with no prompt involved at all.

1. `_prompt_is_flagged()` -- a zero-cost keyword check on the prompt text,
   rejects obvious intent immediately via HTTP 400 in `/generate` before
   any GPU time is spent. Cheap first layer, not the real backstop.
2. `_check_image_safety()` -- runs `Falconsai/nsfw_image_detection` (a
   small self-hosted ViT classifier, already covered by the existing
   `transformers` dependency, no external API key needed) on the actual
   image -- uploaded OR generated -- before shape generation. This is what
   actually matters: it catches inappropriate uploaded images regardless
   of prompt, and catches a generated image even if a prompt slipped past
   the keyword filter. Flagged requests fail with a clear error in
   `/status` instead of proceeding.

## Getting the image built

Push to `main` and the GitHub Actions workflow builds + pushes it
automatically -- nothing to run locally. Requires two repo secrets already
set up: `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`.

## RunPod pod settings

- GPU: RTX 4090 24GB
- **Volume:** mount a persistent volume at `/app/hf-cache` (matches
  `HF_HOME` in the Dockerfile), sized for at least ~20GB of model weights
  plus headroom.
- Container image: `rayf24241/n8-speed:latest`
- Expose port **8000** (HTTP)
- Expect the *first* start after a fresh volume to take several extra
  minutes while weights download -- that's normal, not a hang. Check logs
  for download progress.

## API

```bash
# image -> mesh only (~30s quality target)
curl -X POST http://<endpoint>/generate \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "<base64 png/jpg>"}'
# -> {"job_id": "...", "queue_position": 1, "message": "Generating..."}

curl http://<endpoint>/status/<job_id>
# -> {"status": "done", "image_seconds": 0.0, "rembg_seconds": 0.3,
#     "shape_seconds": 18.4, "simplify_seconds": 0.0, "texture_seconds": 0.0,
#     "export_seconds": 0.3, "total_seconds": 19.0}

curl http://<endpoint>/result/<job_id> -o model.glb
```

Add `"texture": true` to also run the local paint pipeline (slower, separate
`texture_seconds` reported). Add `"prompt": "a red toy car"` instead of
`image_b64` to go text -> (SDXL-Turbo image) -> mesh.

## Next steps once this is live

1. Deploy with the volume mounted, watch startup logs for the weight
   download and which shape checkpoint actually loaded.
2. Hit `/generate` a few times, note real `shape_seconds` on the 4090 --
   that's your actual number against the ~30s quality target, not a guess.
3. If shape gen is still too slow, the first lever is inference steps on the
   shape pipeline (check `hy3dgen/shapegen` for a `num_inference_steps` /
   `steps` kwarg) before considering different hardware.
4. Wire your website to `/generate` + poll `/status` + fetch `/result`.
5. Add more GPU workers behind a router for real parallelism once one
   worker isn't enough -- remember `server.py`'s job state is in-process
   memory, so this needs shared state (e.g. Redis) or sticky routing first.
