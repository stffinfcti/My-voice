# My Voice API (app/server.py, FastAPI)
Run: `python app/server.py` (port 7860, or $PORT). Serves the frontend from `myvoice/web/` (fallback `app/static/`) at `/`. Use relative URLs (`api/...`).
Audio: send **16 kHz mono 16-bit WAV** (encode in the browser with Web Audio; iOS MediaRecorder's mp4/aac is NOT decoded server-side). Max 3 MB per clip (~90 s), keep clips under 15 s.

GET  api/info
  -> {phrases: [30 strings], test: [10 strings], min_to_train: 10, shared: bool, demo_voices: [ids]}

POST api/train   (multipart) files: WAV per recorded phrase (repeat field "files"); indices: "0,1,2,..." phrase index for each file, same order
  -> {job}
GET  api/train/{job}
  -> {state: "running"|"done"|"error", progress: 0..1, desc, voice (when done), seconds, n, error}
  Poll every ~1 s. Expect ~5-7 min on a free 2-CPU host for 30 phrases.

POST api/transcribe   (multipart) audio: WAV; voice: voice id or "" (shared model); expected: what they meant to say, optional
  -> {regular: str, myvoice: str, seconds}
     + if expected: {expected, regular_wer: int %, myvoice_wer: int %,
                     regular_marks: [{w, s}], myvoice_marks: [{w, s}]}  s = "ok" | "wrong" (said/heard wrong word) | "missing" (expected word not heard)
     "Fixed" words for green highlight = words that are "missing" in regular_marks and "ok" in myvoice_marks.

GET  api/voice/{id}        -> my-voice.safetensors download (to keep the voice)
POST api/voice  (multipart) file: .safetensors -> {voice, n}   (load a saved voice)
Voice ids live in server memory; a restart forgets them (frontend can keep the id in localStorage and re-upload the file if 404).
