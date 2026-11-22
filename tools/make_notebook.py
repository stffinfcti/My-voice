import json, sys
cells = []
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n")})

md("""
# My Voice: train the shared base model

Fine-tunes Whisper on TORGO (speech from people with dysarthria) with a LoRA adapter, then compares word error rate before and after on speakers the model never trained on.

**How to run:** Runtime > Change runtime type > **T4 GPU**, then Runtime > **Run all**. Keep this tab open and the screen on. Takes about 30 to 45 minutes. The results print in the last cells.
""")

code(r'''
# 1. Setup
!pip -q install peft jiwer soundfile
!pip -q uninstall -y torchao   # Colab ships an old torchao that new peft refuses to load; we do not need it
import torch, sys
assert torch.cuda.is_available(), "No GPU. Go to Runtime > Change runtime type > T4 GPU, then Run all again."
print("GPU:", torch.cuda.get_device_name(0))
''')

code(r'''
# 2. Settings (safe defaults, no need to change)
MODEL_NAME   = "openai/whisper-base"    # shared base model (small enough to run and personalize on a free CPU)
DATASET      = "abnerh/TORGO-database"  # TORGO, free for academic / non-profit use
TEST_SPEAKERS = ["F03", "M05", "M01", "MC01"]  # held out: moderate, moderate-severe, severe, typical control
MAX_STEPS    = 1500     # training steps (batch 16)
TIME_LIMIT_MIN = 40     # stop training early if it runs longer than this
EVAL_CAP     = 600      # test clips used for the before/after score
LR           = 1e-3
SEED         = 42
''')

code(r'''
# 3. Load TORGO (about 1.6 GB) and split by speaker
import io, os, re, random, time, json
import numpy as np, soundfile as sf, librosa
from datasets import load_dataset, Audio

ds = load_dataset(DATASET, split="train")
ds = ds.cast_column("audio", Audio(decode=False))   # we decode WAV bytes ourselves
print(ds)

def speaker_of(path):
    base = os.path.basename(path or "")
    return base.split("_")[0] if "_" in base else None

meta = ds.remove_columns(["audio"]).to_pandas()
import pyarrow.compute as pc
meta["speaker"] = [speaker_of(x) for x in pc.struct_field(ds.data.column("audio"), "path").to_pylist()]
assert meta["speaker"].notna().mean() > 0.95, f"Could not read speaker ids from file names: {meta['speaker'].head()}"
meta["text"] = meta["transcription"].astype(str).str.strip()
ok = (meta["text"].str.len() > 0) & (~meta["text"].str.contains(r"[\[\]/]|\.wav|\.jpg", regex=True)) & (meta["duration"] <= 29.5)
meta = meta[ok]
print(meta.groupby(["speaker","speech_status","severity"]).size().to_string())

missing = [s for s in TEST_SPEAKERS if s not in set(meta["speaker"])]
assert not missing, f"Test speakers not found: {missing}. Found: {sorted(set(meta['speaker']))}"
train_idx = meta.index[~meta["speaker"].isin(TEST_SPEAKERS)].tolist()
test_all  = meta[meta["speaker"].isin(TEST_SPEAKERS)]
train_texts = set(meta.loc[train_idx, "text"].str.lower())
print(f"\nTrain clips: {len(train_idx)} | Test clips: {len(test_all)} from {TEST_SPEAKERS}")
''')

code(r'''
# 4. Pick the test set: dysarthric held-out speakers + one control
random.seed(SEED)
dys = test_all[test_all["speech_status"] != "healthy"]
ctl = test_all[test_all["speech_status"] == "healthy"]
n_dys = min(len(dys), int(EVAL_CAP * 0.8)); n_ctl = min(len(ctl), EVAL_CAP - n_dys)
test_idx = random.sample(dys.index.tolist(), n_dys) + random.sample(ctl.index.tolist(), n_ctl)
test = meta.loc[test_idx].copy()
test["unseen_text"] = ~test["text"].str.lower().isin(train_texts)
print(f"Test: {n_dys} dysarthric + {n_ctl} control clips. {test['unseen_text'].sum()} have sentences never seen in training.")
''')

code(r'''
# 5. Model, processor, audio helpers, scoring
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import jiwer

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
tok, fe = processor.tokenizer, processor.feature_extractor
tok.set_prefix_tokens(language="english", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).cuda()
model.config.forced_decoder_ids = None
model.generation_config.forced_decoder_ids = None
norm = BasicTextNormalizer()

def load_wav(i):
    a = ds[int(i)]["audio"]
    y, sr = sf.read(io.BytesIO(a["bytes"])) if a.get("bytes") else sf.read(a["path"])
    if y.ndim > 1: y = y.mean(1)
    if sr != 16000: y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000)
    return y.astype(np.float32)

@torch.no_grad()
def transcribe(indices, m, bs=32):
    m.eval(); outs = []
    for k in range(0, len(indices), bs):
        audio = [load_wav(i) for i in indices[k:k+bs]]
        feats = fe(audio, sampling_rate=16000, return_tensors="pt").input_features.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            ids = m.generate(input_features=feats, language="en", task="transcribe", max_new_tokens=64)
        outs += tok.batch_decode(ids, skip_special_tokens=True)
    return outs

def wer(refs, hyps):
    r = [norm(x).strip() or "<empty>" for x in refs]; h = [norm(x).strip() for x in hyps]
    return 100 * jiwer.wer(r, h)

def report(df, hyp_col):
    rows = {}
    for name, part in [("Dysarthric (held-out speakers)", df[df.speech_status!="healthy"]),
                       ("  ...severe only", df[df.severity=="severe"]),
                       ("  ...sentences never seen in training", df[(df.speech_status!="healthy") & df.unseen_text]),
                       ("Typical speech (control)", df[df.speech_status=="healthy"])]:
        if len(part): rows[name] = (round(wer(part.text.tolist(), part[hyp_col].tolist()), 1), len(part))
    return rows
''')

code(r'''
# 6. BEFORE: plain Whisper on the held-out speakers
t = time.time()
test["before"] = transcribe(test.index.tolist(), model)
before = report(test, "before")
for k, (w, n) in before.items(): print(f"{k:45s} WER {w:5.1f}%  ({n} clips)")
print(f"[{time.time()-t:.0f}s]")
''')

code(r'''
# 7. Train the LoRA adapter on the other speakers
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader

model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","out_proj","fc1","fc2"]))
model.print_trainable_parameters()
sot = model.config.decoder_start_token_id

class Clips(Dataset):
    def __init__(self, idx): self.idx = idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, j):
        i = self.idx[j]
        f = fe(load_wav(i), sampling_rate=16000).input_features[0]
        lab = tok(meta.at[i, "text"]).input_ids
        if lab[0] == sot: lab = lab[1:]
        return f, lab

def collate(b):
    feats = torch.tensor(np.stack([x[0] for x in b]))
    L = max(len(x[1]) for x in b)
    labels = torch.full((len(b), L), -100, dtype=torch.long)
    for r, (_, lab) in enumerate(b): labels[r, :len(lab)] = torch.tensor(lab)
    return feats, labels

g = torch.Generator().manual_seed(SEED)
dl = DataLoader(Clips(train_idx), batch_size=16, shuffle=True, collate_fn=collate, num_workers=2, drop_last=True, generator=g)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s+1)/100) * max(0.05, 1 - s/MAX_STEPS))
scaler = torch.amp.GradScaler("cuda")

model.train(); step = 0; t0 = time.time(); run = []
while step < MAX_STEPS and (time.time()-t0) < TIME_LIMIT_MIN*60:
    for feats, labels in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_features=feats.cuda(), labels=labels.cuda()).loss
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        run.append(loss.item()); step += 1
        if step % 50 == 0:
            print(f"step {step}/{MAX_STEPS}  loss {np.mean(run[-50:]):.3f}  {(time.time()-t0)/60:.1f} min", flush=True)
        if step >= MAX_STEPS or (time.time()-t0) > TIME_LIMIT_MIN*60: break
print(f"Trained {step} steps in {(time.time()-t0)/60:.1f} min")
''')

code(r'''
# 8. AFTER: same test clips, same speakers the model never trained on
test["after"] = transcribe(test.index.tolist(), model)
after = report(test, "after")
print(f"{'':45s} {'BEFORE':>8s} {'AFTER':>8s}")
for k in before:
    print(f"{k:45s} {before[k][0]:7.1f}% {after.get(k,(float('nan'),))[0]:7.1f}%   ({before[k][1]} clips)")
''')

code(r'''
# 9. Examples for the demo video + save everything
ex = test[(test.speech_status!="healthy")].copy()
ex["fixed"] = [norm(a) != norm(r) and norm(b) == norm(r) for r, a, b in zip(ex.text, ex.before, ex.after)]
show = ex[ex.fixed].head(15)
for _, r in show.iterrows():
    print(f"[{r.speaker} {r.severity}]  SAID: {r.text}\n    Whisper: {r.before}\n    My Voice: {r.after}\n")

os.makedirs("/content/myvoice_out", exist_ok=True)
model.save_pretrained("/content/myvoice_out/base_lora")
test.to_csv("/content/myvoice_out/test_results.csv")
json.dump({"model": MODEL_NAME, "test_speakers": TEST_SPEAKERS, "steps": step,
           "before": before, "after": after}, open("/content/myvoice_out/results.json","w"), indent=1)
for _, r in show.head(5).iterrows():
    sf.write(f"/content/myvoice_out/example_{r.name}.wav", load_wav(r.name), 16000)
!cd /content && zip -qr myvoice_out.zip myvoice_out && ls -la myvoice_out.zip
print("Saved. Continuing to the personal layer test...")
''')

code(r'''
# 10. Personal layer proof: for each held-out dysarthric speaker, learn their voice from 30 clips
#     (same recipe as the app), then test on their OTHER clips and sentences.
import copy
from peft import get_peft_model_state_dict
shared = model.merge_and_unload().eval()            # shared TORGO knowledge baked in
stock = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).cuda().eval()
stock.generation_config.forced_decoder_ids = None
PERSONAL_LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","out_proj","fc1","fc2"])

def train_personal(idx, base, epochs=8, lr=1e-3, bs=8):
    m = get_peft_model(copy.deepcopy(base), LoraConfig(**PERSONAL_LORA)); m.train()
    feats = torch.tensor(np.stack([fe(load_wav(i), sampling_rate=16000).input_features[0] for i in idx]))
    labs = []
    for i in idx:
        ids = tok(meta.at[i, "text"]).input_ids; labs.append(ids[1:] if ids[0] == sot else ids)
    ps = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=0.01)
    per = int(np.ceil(len(idx)/bs)); total = epochs*per
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0,(k+1)/4)*max(0.1,1-k/total))
    rng = np.random.default_rng(SEED)
    for ep in range(epochs):
        order = rng.permutation(len(idx))
        for b in range(per):
            sel = order[b*bs:(b+1)*bs]; L = max(len(labs[j]) for j in sel)
            lab = torch.full((len(sel), L), -100, dtype=torch.long)
            for r, j in enumerate(sel): lab[r,:len(labs[j])] = torch.tensor(labs[j])
            with torch.autocast("cuda", dtype=torch.float16):
                loss = m(input_features=feats[sel].cuda(), labels=lab.cuda()).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(ps, 1.0); opt.step(); sch.step(); opt.zero_grad()
    return m.eval()

personal_rows = []
for spk in [s for s in TEST_SPEAKERS if (meta.loc[meta.speaker==s, "speech_status"] != "healthy").any()]:
    clips = meta[meta.speaker == spk]
    uniq = clips.drop_duplicates("text").sample(frac=1, random_state=SEED)
    train_p = uniq.index[:30].tolist()
    seen = set(meta.loc[train_p, "text"].str.lower())
    rest = clips[~clips.text.str.lower().isin(seen)]
    eval_p = rest.sample(min(len(rest), 150), random_state=SEED).index.tolist()
    refs = meta.loc[eval_p, "text"].tolist()
    t = time.time(); pm = train_personal(train_p, shared); tt = time.time() - t
    row = {"speaker": spk, "severity": clips.severity.iloc[0], "eval_clips": len(eval_p),
           "plain_whisper": round(wer(refs, transcribe(eval_p, stock)), 1),
           "shared_model": round(wer(refs, transcribe(eval_p, shared)), 1),
           "shared_plus_personal": round(wer(refs, transcribe(eval_p, pm)), 1),
           "personal_train_sec": round(tt, 1)}
    personal_rows.append(row); print(row, flush=True)
    del pm; torch.cuda.empty_cache()

import pandas as pd
print("\nWORD ERROR RATE on each person's held-out clips (lower is better)")
print(pd.DataFrame(personal_rows).to_string(index=False))
json.dump({"model": MODEL_NAME, "test_speakers": TEST_SPEAKERS, "steps": step, "before": before,
           "after": after, "personal": personal_rows}, open("/content/myvoice_out/results.json","w"), indent=1)
''')

code(r'''
# 11. Save the shared model for the app
shared.save_pretrained("/content/myvoice_out/shared_merged")   # not needed by the app, kept for reference
!cd /content && rm -f myvoice_out.zip && zip -qr myvoice_out.zip myvoice_out/base_lora myvoice_out/results.json myvoice_out/test_results.csv myvoice_out/example_*.wav && ls -la myvoice_out.zip
print("DONE. Screenshot the tables from step 8 and step 10 and send them to Claude.")
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1], len(cells), "cells")
