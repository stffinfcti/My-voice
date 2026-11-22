"""My Voice core: speech recognition that adapts to one person's voice.

Two models:
  stock    = plain Whisper (what everyday voice tools hear)
  myvoice  = Whisper + shared adapter trained on TORGO dysarthric speech (merged in),
             plus an optional small personal LoRA adapter trained on ~30 of the user's phrases.
"""
import os, io, time, copy, threading
import numpy as np
import torch

SR = 16000
BASE_MODEL = os.environ.get("MYVOICE_BASE", "openai/whisper-base")
SHARED_ADAPTER = os.environ.get("MYVOICE_SHARED_ADAPTER", "")  # HF repo id or local dir of the TORGO LoRA
torch.set_num_threads(max(1, os.cpu_count() or 1))

_lock = threading.Lock()
_state = {}

def load_audio(x):
    """Accepts (sr, np.array) from Gradio, a file path, or raw float array at 16 kHz."""
    import librosa
    if isinstance(x, tuple):
        sr, y = x
        y = np.asarray(y)
        if y.dtype.kind == "i": y = y.astype(np.float32) / np.iinfo(y.dtype).max
        y = y.astype(np.float32)
        if y.ndim > 1: y = y.mean(axis=1)
        if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        return y
    if isinstance(x, str):
        y, _ = librosa.load(x, sr=SR, mono=True)
        return y.astype(np.float32)
    return np.asarray(x, dtype=np.float32)

def trim(y, top_db=35):
    import librosa
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    return yt if len(yt) > SR * 0.2 else y

def _load():
    if _state: return _state
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    proc = WhisperProcessor.from_pretrained(BASE_MODEL)
    proc.tokenizer.set_prefix_tokens(language="english", task="transcribe")
    stock = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL).eval()
    mine = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    if SHARED_ADAPTER:
        from peft import PeftModel
        mine = PeftModel.from_pretrained(mine, SHARED_ADAPTER).merge_and_unload()
    for m in (stock, mine):
        m.generation_config.forced_decoder_ids = None
        m.config.forced_decoder_ids = None
    _state.update(proc=proc, stock=stock, mine=mine.eval(), shared=bool(SHARED_ADAPTER))
    return _state

def _features(audios):
    fe = _load()["proc"].feature_extractor
    return fe(audios, sampling_rate=SR, return_tensors="pt").input_features

@torch.no_grad()
def transcribe(audio, which="mine", adapter=None):
    """which: 'stock' or 'mine'. adapter: personal LoRA state (dict from train_personal) or None."""
    s = _load()
    y = trim(load_audio(audio))
    feats = _features([y])
    with _lock:
        if which == "stock":
            m = s["stock"]
        else:
            m = _with_adapter(adapter)
        ids = m.generate(input_features=feats, language="en", task="transcribe", max_new_tokens=96)
    return s["proc"].tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()

_adapter_cache = {"key": None, "model": None}
def _with_adapter(adapter):
    s = _load()
    if not adapter: return s["mine"]
    key = adapter["id"]
    if _adapter_cache["key"] != key:
        from peft import PeftModel, LoraConfig, set_peft_model_state_dict, get_peft_model
        m = get_peft_model(copy.deepcopy(s["mine"]), LoraConfig(**adapter["config"]))
        set_peft_model_state_dict(m, adapter["weights"])
        _adapter_cache.update(key=key, model=m.eval())
    return _adapter_cache["model"]

LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])

def train_personal(clips, texts, epochs=8, lr=1e-3, batch_size=8, progress=None, seed=0):
    """Train a personal LoRA on (audio, text) pairs. Returns a small dict you can save with torch.save."""
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    s = _load(); tok = s["proc"].tokenizer
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    audios = [trim(load_audio(c)) for c in clips]
    feats = _features(audios)
    sot = s["mine"].config.decoder_start_token_id
    labels = []
    for t in texts:
        ids = tok(t).input_ids
        labels.append(ids[1:] if ids[0] == sot else ids)
    model = get_peft_model(copy.deepcopy(s["mine"]), LoraConfig(**LORA))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    n = len(audios); steps_per_epoch = max(1, int(np.ceil(n / batch_size))); total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / 4) * max(0.1, 1 - k / total))
    model.train(); step = 0; t0 = time.time(); losses = []
    for ep in range(epochs):
        order = rng.permutation(n)
        for b in range(steps_per_epoch):
            idx = order[b * batch_size:(b + 1) * batch_size]
            L = max(len(labels[i]) for i in idx)
            lab = torch.full((len(idx), L), -100, dtype=torch.long)
            for r, i in enumerate(idx): lab[r, :len(labels[i])] = torch.tensor(labels[i])
            loss = model(input_features=feats[idx], labels=lab).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            losses.append(loss.item()); step += 1
            if progress: progress(step / total, desc=f"Learning your voice: step {step}/{total}")
    model.eval()
    return {"id": f"voice-{int(time.time()*1000)}", "config": LORA,
            "weights": {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()},
            "base": BASE_MODEL, "shared": SHARED_ADAPTER, "n_clips": n,
            "train_seconds": round(time.time() - t0, 1), "final_loss": float(np.mean(losses[-steps_per_epoch:]))}

def save_adapter(adapter, path):
    """Save as safetensors (no pickle), metadata as JSON in the header."""
    import json
    from safetensors.torch import save_file
    meta = {k: v for k, v in adapter.items() if k != "weights"}
    save_file({k: v.contiguous() for k, v in adapter["weights"].items()}, path,
              metadata={"myvoice": json.dumps(meta)})
    return path

def load_adapter(path):
    import json
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        meta = json.loads(f.metadata()["myvoice"])
        meta["weights"] = {k: f.get_tensor(k) for k in f.keys()}
    if meta.get("base") != BASE_MODEL:
        raise ValueError(f"This voice file was made for {meta.get('base')}, the app runs {BASE_MODEL}.")
    return meta

def word_error_rate(ref, hyp):
    import jiwer
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    n = BasicTextNormalizer()
    return 100 * jiwer.wer(n(ref).strip() or "<empty>", n(hyp).strip())
