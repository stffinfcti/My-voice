# My Voice

Speech recognition that learns how *you* talk. Record about 30 short phrases, and in a few minutes a small personal model makes Whisper understand you better than regular voice tools.

- `app/` – API server (FastAPI) and model code: shared Whisper model + per-person LoRA adapter trained on CPU.
- `web/` – phone web app (plain HTML/CSS/JS).
- `notebooks/01_train_base.ipynb` – Colab notebook: fine-tunes Whisper on TORGO (dysarthric speech) and measures word error rate before/after on speakers it never trained on, plus the 30-phrase personal layer per held-out speaker.

Built during EvoHacks II. Uses Whisper (OpenAI, MIT), TORGO (Rudzicz et al. 2012, academic/non-profit use), Hugging Face Transformers + PEFT.

## Run it
- **Try it on a free Colab GPU:** open `notebooks/02_try_the_app.ipynb` in Google Colab, pick a T4 GPU, Run all, and open the public link it prints. To also use the shared TORGO model, run `notebooks/01_train_base.ipynb` first and then this line in a new cell: `!git clone -q --depth 1 https://github.com/stffinfcti/my-voice /content/app; python /content/app/colab/serve.py`
- **Locally or any Docker host:** `pip install torch -r app/requirements-server.txt && python app/server.py`, then open http://localhost:7860. `Dockerfile` and `space/Dockerfile` build the same server.

## License
Code: MIT. TORGO data is not included; it is free for academic and non-profit use (Rudzicz et al., 2012).
