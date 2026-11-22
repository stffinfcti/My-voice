# My Voice

**Name:** My Voice
**Tagline:** Speech recognition that learns how you talk, from 30 phrases, in minutes.

## Inspiration
I grew up around people with speech impediments, and I watched them get misheard over and over, by people and by the technology that was supposed to make life easier. I wanted to build something that listens to them instead of making them repeat themselves.

Voice assistants and dictation work great until your speech doesn't sound "typical." People with dysarthria (after a stroke, or with cerebral palsy, ALS or Parkinson's) and people with lasting speech sound differences, like trouble with R, get misheard all the time. The best-known fix asks for hundreds of recorded phrases and a long wait. We wanted something that works after a few minutes and stays free and open.

## What it does
1. **Teach:** you read about 30 short phrases out loud on your phone.
2. **Training:** My Voice trains a small personal add-on for your voice, a few megabytes, in about 11 seconds on a free Colab GPU. No waiting a day.
3. **Talk:** you speak, and you see what a regular speech model hears next to what My Voice hears, with the wrong words struck through. Copy, speak aloud or share the result.

You keep your voice file and can load it again later.

## How we built it
- **Shared model:** we fine-tuned OpenAI's Whisper (base) with a LoRA adapter on TORGO, a public research set of speech from people with dysarthria, so it gets better at non-typical speech in general.
- **Personal layer:** a second, tiny LoRA adapter trained per person from about 30 clips, merged on top of the shared model.
- **Honest scoring:** we held out four TORGO speakers the model never trained on and measured word error rate (WER) before and after. Test phrases are never used for training.
- **App:** FastAPI backend and a plain HTML/CSS/JS phone web app (live waveform, training progress ring). It runs on a free Colab GPU and opens on any phone through a public link, so it costs nothing to run.

## Results (speakers the model never trained on)
Word error rate, lower is better.

| Test set | Regular Whisper | My Voice shared model |
|---|---|---|
| Dysarthric speakers (480 clips) | 82.6% | 36.8% |
| Severe dysarthria only (156 clips) | 117.1% | 45.1% |
| Typical speech control (120 clips) | 17.8% | 5.7% |

Per speaker, adding the 30-phrase personal layer (trained in about 11 seconds each):

| Speaker | Regular Whisper | Shared model | Shared + personal |
|---|---|---|---|
| F03, moderate | 49.0% | 21.6% | 19.6% |
| M05, moderate-severe | 100.5% | 39.3% | 39.6% |
| M01, severe | 104.4% | 35.8% | 36.9% |

On speakers it never saw, word errors dropped from 83% to 37%. Example: regular Whisper heard "The quick brown fight jumps over the baby dog." My Voice heard "The quick brown fox jumps over the lazy dog."

Honest limits: most of the gain comes from the shared model, and TORGO repeats the same prompts across speakers, so this mostly shows the model learning dysarthric voices on known vocabulary. On a small set of sentences it never saw in training (9 clips) it did not improve (63.3% to 70.0%). The 30-phrase personal layer helped one speaker a little and was flat for the other two. Open-vocabulary speech is the next thing to fix.

## Challenges we ran into
- Training fast enough to feel instant: we picked Whisper base and a small LoRA so a personal layer trains in seconds.
- Free hosting: Hugging Face's GPU-capable Spaces turned out to be paid, so the app runs in a free Colab notebook and opens on a phone through a Cloudflare tunnel.
- A dependency clash in Colab (an old torchao that new PEFT refuses) broke training mid-run; we pinned it out.
- Making iPhone recordings work in the browser: we encode 16 kHz WAV in the browser so the server never has to decode phone audio formats.

## Accomplishments we're proud of
- A real before/after number on speakers the model never trained on (83% to 37% word errors), not a cherry-picked demo, and an honest note on where it doesn't work yet.
- A personal model in about 30 phrases.

## What we learned
How much a small adapter can fix, and how to measure it honestly so the number means something.

## What's next
- Open-vocabulary speech: more varied training sentences so it generalizes beyond known prompts.
- Spanish and English, starting with Spanish phrase sets.
- Let people run the personal layer on their own device so recordings never leave it.
- Test with real users and speech therapists.

## Built with
python, pytorch, whisper, hugging-face-transformers, peft, lora, fastapi, javascript, html, css, google-colab, cloudflare-tunnel

## Links
- Try it: open notebooks/02_try_the_app.ipynb from the repo in Google Colab, Run all, and open the link it prints.
- Code: https://github.com/stffinfcti/my-voice

## Built during the event
All code, model training and the app were built during EvoHacks II. We used OpenAI's Whisper (MIT license), the TORGO dataset (Rudzicz et al., 2012, free for academic and non-profit use), and the open-source Hugging Face Transformers and PEFT libraries.
