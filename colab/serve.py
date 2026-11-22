"""Run the My Voice app inside Google Colab and give it a public https link.

In a Colab cell:
    !git clone -q https://github.com/stffinfcti/my-voice /content/app; python /content/app/colab/serve.py

Uses the shared TORGO model from notebooks/01_train_base.ipynb if that notebook ran first
in the same runtime (/content/myvoice_out/base_lora). Keep the cell running while you use the app.
"""
import os, re, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PORT = "7860"

subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-r", os.path.join(ROOT, "app", "requirements-server.txt")], check=True)

cf = "/content/cloudflared"
if not os.path.exists(cf):
    print("Downloading the tunnel (Cloudflare quick tunnel, free, no account)...")
    urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", cf)
    os.chmod(cf, 0o755)

env = dict(os.environ, PORT=PORT)
shared = "/content/myvoice_out/base_lora"
if os.path.exists(os.path.join(shared, "adapter_config.json")):
    env["MYVOICE_SHARED_ADAPTER"] = shared
    print("Using your shared TORGO model from step 1.")
else:
    print("No shared TORGO model found in this runtime; using plain Whisper + your personal layer.")

log = open("/content/myvoice_server.log", "w")
srv = subprocess.Popen([sys.executable, os.path.join(ROOT, "app", "server.py")], env=env, stdout=log, stderr=subprocess.STDOUT)

# Wait for the server and warm the models up so the first tap is fast
for _ in range(180):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/info", timeout=2); break
    except Exception:
        if srv.poll() is not None:
            log.flush(); print(open("/content/myvoice_server.log").read()[-3000:]); sys.exit("Server stopped. Screenshot this and send it to Claude.")
        time.sleep(2)

tun = subprocess.Popen([cf, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
url = None
for line in tun.stdout:
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
    if m:
        url = m.group(0); break
if not url:
    sys.exit("Could not open a public link. Screenshot this and send it to Claude.")

print("\n" + "=" * 60)
print("MY VOICE IS LIVE. Open this on your phone:")
print(url)
print("Keep this cell running while you use it.")
print("=" * 60 + "\n", flush=True)

try:
    while srv.poll() is None and tun.poll() is None:
        time.sleep(5)
finally:
    srv.terminate(); tun.terminate()
