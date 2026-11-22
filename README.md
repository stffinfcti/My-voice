# My Voice

Speech recognition that learns how *you* talk. Record about 30 short phrases, and in a few minutes a small personal model makes Whisper understand you better than regular voice tools.

- `app/` – API server (FastAPI) and model code: shared Whisper model + per-person LoRA adapter trained on CPU.
- `web/` – phone web app (plain HTML/CSS/JS).
- `notebooks/01_train_base.ipynb` – Colab notebook: fine-tunes Whisper on TORGO (dysarthric speech) and measures word error rate before/after on speakers it never trained on, plus the 30-phrase personal layer per held-out speaker.

Built during EvoHacks II. Uses Whisper (OpenAI, MIT), TORGO (Rudzicz et al. 2012, academic/non-profit use), Hugging Face Transformers + PEFT.

## Run it
- **Hugging Face Space:** create a Space (SDK: Docker, blank) and add `space/Dockerfile` as its `Dockerfile`. It clones this repo and serves the app on port 7860.
- **Locally:** `pip install torch -r app/requirements-server.txt && python app/server.py`, then open http://localhost:7860.

## License
Code: MIT. TORGO data is not included; it is free for academic and non-profit use (Rudzicz et al., 2012).
