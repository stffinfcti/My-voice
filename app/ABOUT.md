## Why
Voice assistants and dictation fail people whose speech doesn't sound "typical": dysarthria after a stroke or with cerebral palsy, ALS, Parkinson's, or a lasting speech sound difference like trouble with R. The one big tool built for this (Google's Project Relate) asks for 500 recorded phrases, takes up to a day, works only in English and isn't taking new sign-ups.

## How it works
1. **Shared model.** We fine-tuned OpenAI's Whisper (base) with a LoRA adapter on TORGO, a public research set of speech from people with dysarthria. This makes it better at non-typical speech in general.
2. **Your layer.** You record about 30 short phrases. My Voice trains a tiny add-on (a few MB) just for your voice, in a few minutes, on an ordinary CPU. Your recordings never leave this app session; you keep your voice file.
3. **Honest scoring.** Test phrases are never used in training, and the model is scored on speakers it never trained on.

## Built during EvoHacks II
All code, training and this app were written during the event. Uses: Whisper (OpenAI, MIT license), TORGO (Rudzicz et al., 2012, free for academic and non-profit use), Hugging Face Transformers and PEFT, Gradio.
