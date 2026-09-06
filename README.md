# N8 Speed

Fast text/image -> 3D mesh, optional texturing, served over HTTP from a
RunPod RTX 4090 pod.

## What's here

- `Dockerfile` -- CUDA 12.4 base, PyTorch cu124 (targets the RTX 4090 we're
  actually running on -- a 12.8 base got flat-out rejected on real RunPod
  hosts whose drivers only support up to 12.4; a 5090 needs a separate
  12.8-based image later, not this one), Hunyuan3D-2 cloned + its compiled
  CUDA extensions. Model weights are **not** baked in (see below).
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

Also unverified: the exact `pipeline(image=image)[0]` / `paint_pipeline(mesh,
image=image)` call signatures match the Hunyuan3D-2 README pattern from
memory. If the container throws on the first `/generate` call, the pod's
logs will show the real signature error -- it's a one-line fix in
`server.py`.

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
# image -> mesh only (~10s target)
curl -X POST http://<endpoint>/generate \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "<base64 png/jpg>"}'
# -> {"job_id": "...", "queue_position": 1, "message": "Generating..."}

curl http://<endpoint>/status/<job_id>
# -> {"status": "done", "image_seconds": 0.0, "shape_seconds": 8.4,
#     "texture_seconds": 0.0, "export_seconds": 0.3, "total_seconds": 8.7}

curl http://<endpoint>/result/<job_id> -o model.glb
```

Add `"texture": true` to also run the paint pipeline (slower, separate
`texture_seconds` reported). Add `"prompt": "a red toy car"` instead of
`image_b64` to go text -> (1-step SDXL-Turbo image) -> mesh.

## Next steps once this is live

1. Deploy with the volume mounted, watch startup logs for the weight
   download and which shape checkpoint actually loaded.
2. Hit `/generate` a few times, note real `shape_seconds` on the 4090 --
   that's your actual number against the 10s target, not a guess.
3. If shape gen is still too slow, the first lever is inference steps on the
   shape pipeline (check `hy3dgen/shapegen` for a `num_inference_steps` /
   `steps` kwarg) before considering different hardware.
4. Wire your website to `/generate` + poll `/status` + fetch `/result`.
5. Add more GPU workers behind a router for real parallelism once one
   worker isn't enough -- remember `server.py`'s job state is in-process
   memory, so this needs shared state (e.g. Redis) or sticky routing first.
