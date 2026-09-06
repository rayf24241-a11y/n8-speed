"""
Runs once at `docker build` time (no GPU available during build, and none
needed here -- this only downloads weight files into the HF cache layer).
Deliberately does NOT import hy3dgen or construct pipeline objects: that would
require CUDA-capable modules to import cleanly with no GPU present, which is
an unnecessary risk during build. Plain file download is enough.
"""
import os
import time

from huggingface_hub import snapshot_download


def download_with_retries(retries=5, **kwargs):
    """
    snapshot_download skips files already sitting in HF_HOME's cache, so
    retrying picks up roughly where a failed attempt left off instead of
    starting the whole repo over -- important here since a dropped connection
    partway through (seen on earlier build attempts: HF's Xet CDN backend
    threw a transient DNS error, then several ~3.8GB checkpoint files
    downloading in parallel all dropped connection at once and crashed the
    whole Docker build) would otherwise be indistinguishable from having to
    redownload everything.

    max_workers=2 deliberately limits how many of those multi-GB files
    download in parallel -- the default (8) was opening that many simultaneous
    huge transfers, which is what took down the connection last time.
    """
    kwargs.setdefault("max_workers", 2)
    for attempt in range(1, retries + 1):
        try:
            return snapshot_download(**kwargs)
        except Exception as e:  # noqa: BLE001
            if attempt == retries:
                raise
            wait = 15 * attempt
            print(f"  download failed ({e}); retry {attempt}/{retries} in {wait}s", flush=True)
            time.sleep(wait)


print("Prefetching tencent/Hunyuan3D-2mini (turbo shape checkpoint only)...", flush=True)
download_with_retries(
    repo_id="tencent/Hunyuan3D-2mini",
    # The full repo carries 3 DiT variants (mini/mini-fast/mini-turbo) each
    # duplicated in both .ckpt and .safetensors (~3.8GB apiece) -- grabbing
    # everything is 20GB+ of formats/variants we never load. server.py's
    # first candidate is hunyuan3d-dit-v2-mini-turbo, so that's all we fetch;
    # its fallback candidates come from the tencent/Hunyuan3D-2 repo below.
    allow_patterns=[
        "hunyuan3d-dit-v2-mini-turbo/*.safetensors",
        "hunyuan3d-dit-v2-mini-turbo/*.json",
        "hunyuan3d-dit-v2-mini-turbo/*.yaml",
        "hunyuan3d-vae-v2-mini-turbo/*",
    ],
)

print("Prefetching tencent/Hunyuan3D-2 (turbo fallback shape + paint checkpoints)...", flush=True)
download_with_retries(
    repo_id="tencent/Hunyuan3D-2",
    # Same reasoning: this repo also carries multiple shape DiT variants we
    # don't use (we only fall back to v2-0-turbo/v2-0, never plain v2-0 mid
    # variants) plus every paint variant. Keep the two shape fallbacks from
    # server.py's candidate list and both paint checkpoints (unconfirmed
    # exact folder name -- keeping both turbo/base is cheap insurance
    # against one being wrong, and still far smaller than the full repo).
    allow_patterns=[
        "hunyuan3d-dit-v2-0-turbo/*.safetensors",
        "hunyuan3d-dit-v2-0-turbo/*.json",
        "hunyuan3d-dit-v2-0-turbo/*.yaml",
        "hunyuan3d-dit-v2-0/*.safetensors",
        "hunyuan3d-dit-v2-0/*.json",
        "hunyuan3d-dit-v2-0/*.yaml",
        "hunyuan3d-vae-v2-0/*",
        "hunyuan3d-paint-v2-0-turbo/*",
        "hunyuan3d-paint-v2-0/*",
    ],
)

print("Prefetching stabilityai/sdxl-turbo (fp16 only, for text prompts)...", flush=True)
download_with_retries(
    repo_id="stabilityai/sdxl-turbo",
    allow_patterns=["*.json", "*.txt", "*fp16*"],
)

cache_root = os.environ["HF_HOME"]
total = 0
for dirpath, _, filenames in os.walk(cache_root):
    for f in filenames:
        total += os.path.getsize(os.path.join(dirpath, f))
print(f"Prefetch complete. HF cache size: {total / 1e9:.1f} GB", flush=True)
