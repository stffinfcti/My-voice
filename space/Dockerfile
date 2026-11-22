# Paste this as the only file in a Hugging Face Space (SDK: Docker).
# It builds My Voice straight from the public GitHub repo.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
ENV HOME=/home/user HF_HOME=/home/user/hf OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
WORKDIR /home/user
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# Re-download the code whenever the repo changes
ADD https://api.github.com/repos/stffinfcti/my-voice/commits/main /tmp/version.json
RUN git clone --depth 1 https://github.com/stffinfcti/my-voice.git src && pip install --no-cache-dir -r src/app/requirements-server.txt
RUN python -c "from transformers import WhisperProcessor as P, WhisperForConditionalGeneration as W; P.from_pretrained('openai/whisper-base'); W.from_pretrained('openai/whisper-base')"
RUN chown -R user /home/user
USER user
EXPOSE 7860
CMD ["python", "src/app/server.py"]
