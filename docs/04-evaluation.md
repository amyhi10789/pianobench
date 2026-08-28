# Part 4 — Evaluate Videos with Metrics

**Goal:** Score each generated video on audio accuracy, video accuracy, and audio–video alignment, then compare models.

**Time:** about 25–40 minutes  
**Prerequisites:** [Parts 1–3](01-vscode-setup.md)

---

## What “evaluation” means here

We do **not** ask “which video looks prettier?”  
We ask measurable questions:

1. **Audio accuracy** — Is the heard note the one the prompt requested?  
2. **Video accuracy** — Does the correct key visually press near the requested time?  
3. **AV alignment** — Does the sound start when the key starts moving?

Each metric is a number from **0.0 (bad)** to **1.0 (perfect)**.

An **overall** score is a weighted average:

| Metric | Weight |
|--------|--------|
| Audio accuracy | 0.35 |
| Video accuracy | 0.35 |
| AV alignment | 0.30 |

You can change weights later in `evaluation/evaluate.py` (`DEFAULT_WEIGHTS`).

---

## Code layout

```text
evaluation/
├── evaluate.py                 # main script (shared metrics, per-level expectations)
├── metrics/
│   ├── audio_accuracy.py       # used for every level
│   ├── video_accuracy.py
│   └── av_alignment.py
└── results/                    # created/updated when you run evaluate.py
    ├── scores.json
    └── scores.csv
```

Tutorial text for this step is this file: `docs/04-evaluation.md`.

---

## Important: placeholder demo mode

Real note detection and key tracking need extra libraries and more advanced code.

- **Audio accuracy** is implemented with `librosa` (onset + pYIN) and is shared across levels. Install
  `requirements.txt` and have `ffmpeg` available for real scores.
- **Video accuracy** samples six frames around the expected press and sends them
  to an image-capable model. Set `OPENAI_API_KEY` before running it. The model can
  be overridden with `PIANOBENCH_VISION_MODEL` (default: `gpt-5.4-mini`).
- **AV alignment** reuses the audio note onsets and VLM visual press times from
  the same evaluation. It pairs detections one-to-one by nearest time and scores
  synchronization independently of pitch correctness. Missing events receive
  zero; lag credit tapers to zero at 0.5 seconds.

On Windows PowerShell, set the API key for the current terminal with:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
python evaluation/evaluate.py --no-demo
```

Do not paste an API key into source code or commit it to the repository. Video
accuracy sends sampled JPEG frames—not the complete MP4—to the configured API.

Demo mode is **ON** by default for metrics that cannot produce a real score.
Audio uses real detection when dependencies work; it only falls back to a demo
score if loading/detection fails (or finds nothing) while demo mode is on.

To run **without** demo scores (stubs → zeros; audio still real if deps work):

```bash
python3 evaluation/evaluate.py --no-demo
```

---

## Run the full manifest

From the **project root**:

```bash
python3 evaluation/evaluate.py
```

Example table (demo video/AV placeholders; audio is real if deps are installed):

```text
prompt  lvl  model                    audio   video   align   overall
---------------------------------------------------------------------
1A      1    Gemini                   0.589   0.650   0.833     0.689
1A      1    Gemini                   0.502   0.650   0.833     0.658
```

Numbers will differ on your machine. Audio also reports component scores
(`count`, `pitch`, `order`, `timing`, `duration`) in `scores.json`.

Outputs:

- `evaluation/results/scores.json` — full details  
- `evaluation/results/scores.csv` — easy to open in Excel / Google Sheets

---

## Score one video

```bash
python3 evaluation/evaluate.py \
  --video data/videos/gemini/level1/piano_vid_new_1.mp4 \
  --prompt-id 1A
```

---

## How to read the scores

| Overall (rough guide) | Meaning |
|-----------------------|---------|
| 0.85 – 1.00 | Strong match to the prompt |
| 0.60 – 0.84 | Partial match; check which metric is weak |
| below 0.60 | Likely wrong note, wrong timing, or poor sync |

Always look at the **three** top-level metrics, not only overall. For audio,
also check the component scores in `scores.json` when a clip fails.

- Low **audio**, high **video** → looks right but sounds wrong  
- High **audio**, low **video** → sounds right but wrong/unclear key motion  
- High audio & video, low **alignment** → right events, but out of sync  

---

## Comparing multiple models

1. Put each model’s videos under `data/videos/{model}/{level}/` (e.g. `data/videos/gemini/level1/`)
2. Register them in `data/manifest.json` (see [Part 3](03-prepare-data.md))
3. Run `evaluation/evaluate.py`
4. Sort `scores.csv` by `prompt_id`, then compare `overall` across `model_name`

Suggested follow-up:

- Fix one prompt (e.g. 1A)
- Compare models on the same prompt
- Note which model scored higher, which metric drove the difference, and what you might change in the prompt

---

## Where to extend the code (advanced)

| File | Function to replace | Future idea |
|------|---------------------|-------------|
| `evaluation/metrics/audio_accuracy.py` | already implemented (librosa onsets + pYIN) | refine thresholds / multi-pitch for chords |
| `evaluation/metrics/video_accuracy.py` | VLM frame analysis implemented | calibrate against human-labeled videos |
| `evaluation/metrics/av_alignment.py` | real event matching implemented | calibrate lag tolerance against labeled videos |

**Audio accuracy** combines count, pitch, order, timing, and duration
component scores (see `DEFAULT_WEIGHTS` in that file). The same functions
score every level; only `data/expectations/levelN.json` changes.

Keep the scoring functions’ return format the same so `evaluate.py` keeps working.

---

## Checkpoint

- [ ] You ran `python3 evaluation/evaluate.py`
- [ ] You opened `evaluation/results/scores.csv`
- [ ] You can explain the three metrics in one sentence each
- [ ] You know video/AV demo mode is temporary until those detectors are added

---

## You’re done with the tutorial path

Return to the main [README](../README.md) or continue by:

1. Adding Prompt 1B / 1C videos to the manifest  
2. Implementing real video or AV detectors  
3. Changing metric weights and seeing how rankings change  
