# N8 Speed

Fast text/image -> 3D mesh, optional texturing, served over HTTP for a
single Salad RTX 5090 container.

## What's here

- `Dockerfile` -- CUDA 12.8 base, PyTorch cu128 (required for the 5090 /
  Blackwell / sm_120), Hunyuan3D-2 cloned + its compiled CUDA extensions,
  model weights baked in at build time.
- `prefetch_models.py` -- runs during the build to pull weights into the
  image so a container never has to download anything on first request.
- `server.py` -- FastAPI app: one job queue, one worker thread (matches one
  GPU). `/generate` queues a job, `/status/{id}` polls it, `/result/{id}`
  downloads the `.glb`.

## Known risk before first real run

I don't have a way to hit huggingface.co/tencent/Hunyuan3D-2mini from here to
confirm the *exact* current subfolder name for the turbo checkpoint, and I
can't run CUDA locally to test the pipeline end to end (this machine has no
NVIDIA GPU). `server.py` tries three checkpoints in order (mini-turbo -> base
turbo -> base) and logs which one it actually loaded, so a naming drift
degrades speed rather than crashing. **Read the container's startup logs on
first deploy** -- if it fell back past the first candidate, open
https://huggingface.co/tencent/Hunyuan3D-2mini and fix the subfolder string in
`server.py`'s `_load_shape_pipeline()`.

Also unverified: the exact `pipeline(image=image)[0]` / `paint_pipeline(mesh,
image=image)` call signatures match the Hunyuan3D-2 README pattern from
memory. If the container throws on the first `/generate` call, `docker logs`
will show the real signature error -- it's a one-line fix in `server.py`.

## Build

From this directory:

```bash
docker build -t <dockerhub-username>/n8-speed:latest .
```

This will download several tens of GB (PyTorch + model weights) and take a
while -- it's a one-time cost per image version, not per container start.

## Push to a registry

Salad pulls from Docker Hub (or another registry). Simplest path to start:

```bash
docker login
docker push <dockerhub-username>/n8-speed:latest
```

Use a **private** repo on Docker Hub once this is working (Salad supports
private registries with credentials) so the baked-in weights/business logic
aren't public. Public is fine for the first test.

## Salad container group settings

On the "Image Source" screen:

- **What service are you using?** Docker Hub (or your chosen registry)
- **Public or Private Registry?** Public to start, switch to Private + add
  credentials once it works
- **Image Name:** `<dockerhub-username>/n8-speed:latest`

Rest of the container group:

- GPU: RTX 5090 32GB
- 8 vCPU / 16GB RAM / 100GB disk (your existing plan is fine)
- Shared memory: 64MB default is fine -- this server doesn't use
  multi-process DataLoader workers. Only raise it if you see shared-memory
  errors in the logs.
- Container Gateway: enabled, port **8000**
- Health probe path: `/health`
- Replicas: 1 to start

## API

```bash
# image -> mesh only (~10s target)
curl -X POST http://<endpoint>/generate \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "<base64 png/jpg>"}'
# -> {"job_id": "...", "queue_position": 1, "message": "Generating..."}

curl http://<endpoint>/status/<job_id>
# -> {"status": "done", "shape_seconds": 8.4, "texture_seconds": 0.0}

curl http://<endpoint>/result/<job_id> -o model.glb
```

Add `"texture": true` to also run the paint pipeline (slower, separate
`texture_seconds` reported). Add `"prompt": "a red toy car"` instead of
`image_b64` to go text -> (1-step SDXL-Turbo image) -> mesh.

## Next steps once this is live

1. Deploy, watch startup logs for which shape checkpoint actually loaded.
2. Hit `/generate` a few times, note real `shape_seconds` on the 5090 --
   that's your actual number against the 10s target, not a guess.
3. If shape gen is still too slow, the first lever is inference steps on the
   shape pipeline (check `hy3dgen/shapegen` for a `num_inference_steps` /
   `steps` kwarg) before considering a smaller/quantized checkpoint.
4. Wire your website to `/generate` + poll `/status` + fetch `/result`.
5. Add more replicas behind a router for real parallelism once one worker
   isn't enough.
