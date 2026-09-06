"""
N8 Speed local test UI.

Stdlib-only local server (no pip installs needed): serves index.html and
proxies /api/* calls to the currently running RunPod pod. Proxying
server-side sidesteps browser CORS entirely without touching or redeploying
N8 Speed's own image -- the browser only ever talks to localhost.

FBX export works the same way: converted locally via headless Blender (the
same approach already proven in Hunyuan3D-Output\\Generate-Model.ps1), not
on the RunPod side -- N8 Speed's own pipeline only ever produces glb, and
bloating that image with a full Blender install just for format conversion
isn't worth it when a working local Blender already does the job.
"""
import json
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Update this every time you spin up a new pod -- RunPod pod IDs change on
# every redeploy, so this is the one thing that needs editing between runs.
N8_SPEED_URL = "https://0pcaq4vahzqzqn-8000.proxy.runpod.net"

BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"

HERE = Path(__file__).parent
GLB_TO_FBX_SCRIPT = HERE / "glb_to_fbx.py"
PORT = 7860


class Handler(BaseHTTPRequestHandler):
    def _proxy(self, method, path, body=None):
        req = urllib.request.Request(N8_SPEED_URL + path, data=body, method=method)
        # RunPod's proxy domain sits behind Cloudflare, which blocks the
        # default "Python-urllib/x.y" User-Agent as a bot signature (error
        # code 1010) -- confirmed by testing this proxy live. curl wasn't
        # affected since it isn't on that blocklist, which is why direct
        # curl testing never surfaced this.
        req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        if body is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.status, resp.read(), resp.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as e:
            return e.code, e.read(), "application/json"
        except urllib.error.URLError as e:
            return 502, json.dumps({"error": f"Can't reach N8 Speed pod: {e.reason}"}).encode(), "application/json"

    def _convert_to_fbx(self, glb_bytes):
        """Round-trips through headless Blender, same as Generate-Model.ps1's
        Save-Result does. Returns (fbx_bytes, error_message)."""
        with tempfile.TemporaryDirectory() as tmp:
            glb_path = Path(tmp) / "model.glb"
            fbx_path = Path(tmp) / "model.fbx"
            glb_path.write_bytes(glb_bytes)
            try:
                result = subprocess.run(
                    [BLENDER_EXE, "--background", "--python", str(GLB_TO_FBX_SCRIPT),
                     "--", str(glb_path), str(fbx_path)],
                    capture_output=True, timeout=120,
                )
            except FileNotFoundError:
                return None, f"Blender not found at {BLENDER_EXE} -- update BLENDER_EXE in server.py"
            except subprocess.TimeoutExpired:
                return None, "Blender conversion timed out"
            if not fbx_path.exists():
                stderr = result.stderr.decode(errors="replace")[-2000:]
                return None, f"Blender conversion failed: {stderr}"
            return fbx_path.read_bytes(), None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._serve_file("index.html", "text/html")
        elif parsed.path == "/api/health":
            self._respond(*self._proxy("GET", "/health"))
        elif parsed.path.startswith("/api/status/"):
            self._respond(*self._proxy("GET", parsed.path[len("/api"):]))
        elif parsed.path.startswith("/api/result/"):
            fmt = urllib.parse.parse_qs(parsed.query).get("format", ["glb"])[0]
            status, body, _ = self._proxy("GET", parsed.path[len("/api"):])
            if status != 200:
                self._respond(status, body, "application/json")
                return
            if fmt == "fbx":
                fbx_bytes, error = self._convert_to_fbx(body)
                if error:
                    self._respond(500, json.dumps({"error": error}).encode(), "application/json")
                else:
                    self._respond(200, fbx_bytes, "application/octet-stream")
            else:
                self._respond(200, body, "model/gltf-binary")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/generate":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            self._respond(*self._proxy("POST", "/generate", body=body))
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_file(self, filename, content_type):
        data = (HERE / filename).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _respond(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"N8 Speed local UI:  http://localhost:{PORT}")
    print(f"Proxying to:        {N8_SPEED_URL}")
    server.serve_forever()
