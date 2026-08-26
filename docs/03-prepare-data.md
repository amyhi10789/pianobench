# Part 3 — Prepare Video & Prompt Data

**Goal:** Understand how prompts, model videos, and scoring expectations fit together — and how to add your own.

**Time:** about 20–30 minutes  
**Prerequisites:** [Parts 1–2](01-vscode-setup.md)

---

## Big picture

For every test we need three things:

1. **Prompt text** — what we asked the AI model to generate  
2. **Video file** — what the model produced (`.mp4`)  
3. **Expectations** — the “correct answer” we score against (note name, timing, …)

There are **no ground-truth videos**. `data/videos/` only stores model outputs.

Then a **manifest** connects them: *this video came from model X for prompt Y*.

```text
Prompt 1A (C4)  +  Gemini  →  videos/gemini/level1/...mp4  →  scores
Prompt 1B (E4)  +  Gemini  →  ...
```

### Prompt levels

| Level | Theme |
|-------|--------|
| 1 | Singular notes |
| 2 | Ascending sequences |
| 3 | Descending sequences |
| 4 | Repeated notes |
| 5 | Chords |

---

## Folder layout

All example data lives in `data/`:

```text
data/
├── prompts/
│   ├── shared_instructions.txt      # rules shared by every prompt
│   ├── research_prompt_full.txt     # full text converted from the PDF
│   ├── level1/                      # Singular notes
│   │   ├── prompt_1a_middle_c.txt
│   │   ├── prompt_1b_e4.txt
│   │   └── prompt_1c_g4.txt
│   ├── level2/                      # Ascending sequences (placeholder)
│   ├── level3/                      # Descending sequences (placeholder)
│   ├── level4/                      # Repeated notes (placeholder)
│   └── level5/                      # Chords (placeholder)
├── videos/
│   └── gemini/
│       └── level1/
│           ├── piano_vid_new_1.mp4
│           └── piano_vid_new_2.mp4
├── expectations/
│   ├── level1.json                  # singular notes
│   ├── level2.json … level5.json    # placeholders
└── manifest.json                    # prompt × model × video list
```

Video layout: `videos/{model}/{level}/*.mp4` (e.g. `videos/gemini/level1/`).

The original PDF is kept at `video+prompt/research prompt.pdf` (reference only).
Day-to-day work uses the `.txt` files — easier to open and edit in VS Code.

---

## How a full prompt is built

Each generation request is usually:

```text
[shared_instructions.txt]
+
[one specific prompt file, e.g. level1/prompt_1a_middle_c.txt]
```

### Level 1 (singular notes) summary

| Prompt ID | Target note | MIDI | Press time | Hold |
|-----------|-------------|------|------------|------|
| 1A | C4 (middle C) | 60 | 3.0 s | 0.5 s |
| 1B | E4 | 64 | 3.0 s | 0.5 s |
| 1C | G4 | 67 | 3.0 s | 0.5 s |

Shared rules (short version):

- 8-second video, fixed overhead camera
- One right hand, exaggerated clear key presses
- Matching acoustic piano audio, no extra sounds
- White keys only for these Level 1 prompts

Open the `.txt` files to read the full wording.

---

## Expectations (`expectations/levelN.json`)

Each level has its own expectations file. These tell the **shared** scorers
(`evaluation/metrics/`) what “correct” means. They are **not** videos.
There is one metrics package for every level; only the expectation JSON changes.

Example for Prompt 1A in `expectations/level1.json`:

- note = `C4`
- press at `3.0` seconds
- hold `0.5` seconds
- release at `3.5` seconds
- exactly one note

If you add Prompt 1D later, add a matching block in `data/expectations/level1.json`.

---

## Manifest (`manifest.json`)

The manifest lists every evaluation you want to run:

```json
{
  "id": "eval_1a_gemini_1",
  "prompt_id": "1A",
  "level": 1,
  "model_id": "gemini",
  "video_path": "videos/gemini/level1/piano_vid_new_1.mp4"
}
```

Included examples map both sample videos to **Prompt 1A** under **Gemini**.

---

## Practice: add a new evaluation row

Suppose Gemini generated a new file for Prompt 1B.

1. Copy the `.mp4` into `data/videos/gemini/level1/`  
   Example name: `prompt_1b.mp4`
2. Open `data/manifest.json`
3. Add the model (if new) under `"models"`
4. Add an entry under `"evaluations"`:

```json
{
  "id": "eval_1b_gemini",
  "prompt_id": "1B",
  "level": 1,
  "model_id": "gemini",
  "video_path": "videos/gemini/level1/prompt_1b.mp4",
  "notes": "My new sample"
}
```

5. Save the file
6. Run the evaluator (Part 4)

---

## Practice: create a new prompt file

1. Copy `data/prompts/level1/prompt_1a_middle_c.txt` to `data/prompts/level1/prompt_1d_d4.txt`
2. Edit the text so it asks for **D4** instead of C4
3. Add a `"1D"` section in `data/expectations/level1.json`
4. Generate videos with your models into `data/videos/{model}/level1/`
5. Register them in `data/manifest.json`

For a new difficulty tier, put files under `prompts/level2/` (etc.) and videos under `videos/{model}/level2/`.

---

## Naming tips (keep things tidy)

| Item | Suggested pattern |
|------|-------------------|
| Prompt file | `prompts/levelN/prompt_<id>_<note>.txt` |
| Video file | `videos/<model>/levelN/<prompt_id>.mp4` |
| Eval id | `eval_<prompt_id>_<model>` |

Avoid spaces in filenames when you can (`piano_vid_new_1.mp4` is fine; `my video.mp4` is harder to use in terminals).

---

## Optional helper script

From the project root:

```bash
python3 scripts/build_full_prompt.py --prompt-id 1A
```

This prints (or saves) shared instructions + the specific prompt combined — useful when you paste into a video model UI.

---

## Checkpoint

- [ ] You can open and read the `.txt` prompts under `prompts/level1/`
- [ ] You know videos live under `videos/{model}/{level}/` (no GT videos)
- [ ] You understand that `manifest.json` links prompt + model + video
- [ ] (Optional) You added one new manifest row

**Next:** [04-evaluation.md](04-evaluation.md)
