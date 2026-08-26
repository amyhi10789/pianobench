# PianoBench

This project walks through how to:

1. Open a coding project in **VS Code**
2. Set up a simple **Python environment**
3. Organize **video + prompt data** for AI video generation
4. **Score** generated videos with clear metrics

We study AI models that generate short piano videos from text prompts.
Each prompt asks for a specific note (for example, middle C / C4) with matching audio and video.

## What we measure

| Metric | Question it answers |
|--------|---------------------|
| **Audio accuracy** | Did we hear the correct piano note(s)? |
| **Video accuracy** | Did the correct key move at the right time? |
| **AV alignment** | Did the sound start when the key pressed? |

Each (prompt × model × video) gets metric scores and one **overall** score.

## Project layout

```
pianobench/
├── README.md                 ← you are here
├── requirements.txt
├── docs/                     ← tutorial (read in order)
│   ├── 01-vscode-setup.md
│   ├── 02-environment.md
│   ├── 03-prepare-data.md
│   └── 04-evaluation.md
├── data/                     ← prompts, model videos, expectations
│   ├── prompts/              ← shared + level1…level5
│   ├── videos/               ← videos/{model}/{level}/
│   ├── expectations/         ← level1.json … level5.json (no GT videos)
│   └── manifest.json
├── evaluation/               ← scoring code
│   ├── evaluate.py
│   ├── metrics/              ← shared audio / video / AV scores
│   └── results/
├── scripts/
│   └── build_full_prompt.py
└── video+prompt/             ← original PDF (reference only)
```

## Learning path

| Step | Doc | Time (approx.) |
|------|-----|----------------|
| 1 | [docs/01-vscode-setup.md](docs/01-vscode-setup.md) | 15–20 min |
| 2 | [docs/02-environment.md](docs/02-environment.md) | 15–25 min |
| 3 | [docs/03-prepare-data.md](docs/03-prepare-data.md) | 20–30 min |
| 4 | [docs/04-evaluation.md](docs/04-evaluation.md) | 25–40 min |

## Quick start (after docs 01–02)

From the project root:

```bash
python3 evaluation/evaluate.py
```

Scores are written to `evaluation/results/`.

## Example data included

- Prompts from the original PDF → `.txt` files under `data/prompts/level1/`
- Two Gemini example videos in `data/videos/gemini/level1/`
- Scoring expectations for Level 1 notes: **1A (C4)**, **1B (E4)**, **1C (G4)** in `data/expectations/level1.json`
- There are **no ground-truth videos** — only generated model outputs plus per-level expectations

Original PDF (reference only): `video+prompt/research prompt.pdf`

## Notes

You do **not** need to train models — the focus is preparing data and comparing outputs.

- **Audio accuracy** is implemented once and used for every level (librosa onsets + pYIN): count, pitch, order, timing, and duration components.
- **Video accuracy** and **AV alignment** still use placeholder demo scores until real detectors are added (see [docs/04-evaluation.md](docs/04-evaluation.md)).
- Install `requirements.txt` (and system `ffmpeg`) for real audio scoring.
