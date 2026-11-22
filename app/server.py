"""My Voice web server: custom phone UI (static/) + JSON API around core.py."""
import io, os, re, glob, time, uuid, difflib, threading, tempfile
import numpy as np, soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import core
from phrases import TRAIN, TEST, MIN_TO_TRAIN

app = FastAPI(title="My Voice")
HERE = os.path.dirname(os.path.abspath(__file__))
VOICES = {}            # id -> adapter dict (kept in memory for this server run)
JOBS = {}              # job id -> {"state","progress","desc","voice","error"}
MAX_AUDIO_BYTES = 3_000_000
for p in sorted(glob.glob(os.path.join(HERE, "voices", "*.safetensors"))):
    try: VOICES["demo-" + os.path.splitext(os.path.basename(p))[0]] = core.load_adapter(p)
    except Exception as e: print("skip demo voice", p, e)

def _wav(upload: UploadFile):
    data = upload.file.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES: raise HTTPException(413, "Recording too long.")
    try: y, sr = sf.read(io.BytesIO(data), dtype="float32")
    except Exception: raise HTTPException(400, "Couldn't read that recording.")
    return (sr, y)

@app.get("/api/info")
def info():
    s = core._load()
    return {"phrases": TRAIN, "test": TEST, "min_to_train": MIN_TO_TRAIN, "shared": s["shared"],
            "demo_voices": [k for k in VOICES if k.startswith("demo-")]}

@app.post("/api/train")
async def train(files: list[UploadFile] = File(...), indices: str = Form(...)):
    idx = [int(x) for x in indices.split(",") if x.strip() != ""]
    if len(idx) != len(files) or len(idx) < MIN_TO_TRAIN: raise HTTPException(400, f"Record at least {MIN_TO_TRAIN} phrases.")
    if any(i < 0 or i >= len(TRAIN) for i in idx): raise HTTPException(400, "Bad phrase index.")
    clips = [_wav(f) for f in files]; texts = [TRAIN[i] for i in idx]
    job = uuid.uuid4().hex[:12]; JOBS[job] = {"state": "running", "progress": 0.0, "desc": "Getting ready", "voice": None}
    def run():
        try:
            def prog(frac, desc=""): JOBS[job].update(progress=float(frac), desc=desc)
            ad = core.train_personal(clips, texts, progress=prog)
            vid = "v-" + uuid.uuid4().hex[:10]; VOICES[vid] = ad
            JOBS[job].update(state="done", progress=1.0, voice=vid, seconds=ad["train_seconds"], n=ad["n_clips"])
        except Exception as e:
            JOBS[job].update(state="error", error=str(e))
    threading.Thread(target=run, daemon=True).start()
    return {"job": job}

@app.get("/api/train/{job}")
def train_status(job: str):
    if job not in JOBS: raise HTTPException(404, "Unknown job")
    return JOBS[job]

def _words(t): return [w for w in re.sub(r"[^\w\s']", " ", t.lower()).split() if w]

def _mark(hyp, ref):
    """Word-level marks for hyp against ref: ok / wrong (substituted or extra) plus missing ref words."""
    h, r = _words(hyp), _words(ref)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=r, b=h, autojunk=False).get_opcodes():
        if op == "equal": out += [{"w": w, "s": "ok"} for w in h[j1:j2]]
        elif op in ("replace", "insert"): out += [{"w": w, "s": "wrong"} for w in h[j1:j2]]
        if op in ("replace", "delete"): out += [{"w": w, "s": "missing"} for w in r[i1:i2]]
    return out

@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...), voice: str = Form(""), expected: str = Form("")):
    clip = _wav(audio)
    ad = VOICES.get(voice) if voice else None
    t0 = time.time()
    a = core.transcribe(clip, "stock"); b = core.transcribe(clip, "mine", ad)
    res = {"regular": a, "myvoice": b, "seconds": round(time.time() - t0, 2)}
    if expected.strip():
        res.update(expected=expected.strip(),
                   regular_marks=_mark(a, expected), myvoice_marks=_mark(b, expected),
                   regular_wer=round(core.word_error_rate(expected, a)), myvoice_wer=round(core.word_error_rate(expected, b)))
    return res

@app.get("/api/voice/{vid}")
def download_voice(vid: str):
    if vid not in VOICES: raise HTTPException(404, "Unknown voice")
    path = os.path.join(tempfile.mkdtemp(), "my-voice.safetensors")
    core.save_adapter(VOICES[vid], path)
    return FileResponse(path, filename="my-voice.safetensors")

@app.post("/api/voice")
async def upload_voice(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 50_000_000: raise HTTPException(413, "File too big.")
    path = os.path.join(tempfile.mkdtemp(), "v.safetensors"); open(path, "wb").write(data)
    try: ad = core.load_adapter(path)
    except Exception as e: raise HTTPException(400, f"Not a My Voice file: {e}")
    vid = "v-" + uuid.uuid4().hex[:10]; VOICES[vid] = ad
    return {"voice": vid, "n": ad.get("n_clips")}

# Frontend: the designed web/ folder if present (sibling of app/), else the fallback static/ page.
WEB = os.environ.get("MYVOICE_WEB") or next((d for d in [os.path.join(HERE, "..", "web"), os.path.join(HERE, "web"), os.path.join(HERE, "static")]
                                            if os.path.exists(os.path.join(d, "index.html"))), os.path.join(HERE, "static"))
app.mount("/", StaticFiles(directory=WEB, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
