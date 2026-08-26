# Part 2 — Set Up Your Environment

**Goal:** Install Python and run a tiny test so the evaluation script can run on your machine.

**Time:** about 15–25 minutes  
**Prerequisites:** [Part 1](01-vscode-setup.md) finished (project open in VS Code)

---

## What is an “environment”?

Your **environment** is the set of tools Python uses to run code:

- The Python program itself (`python3`)
- Packages listed in `requirements.txt` (needed for real audio scoring)
- Sometimes a **virtual environment** (a private toolbox for this project only)

For this tutorial, **Python 3.10+** is enough to run the evaluator.
**Video / AV** metrics can still use demo placeholders from the standard library, but
**audio accuracy** needs the packages in `requirements.txt` plus system **ffmpeg**.

---

## 1. Check whether Python is already installed

In the VS Code terminal:

```bash
python3 --version
```

Good examples:

```text
Python 3.11.6
Python 3.12.1
```

If the command is not found, install Python next.

---

## 2. Install Python (if needed)

### macOS

1. Install from [https://www.python.org/downloads/](https://www.python.org/downloads/), **or**
2. If you use Homebrew:

```bash
brew install python
```

### Windows

1. Download the installer from [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. During setup, check **“Add python.exe to PATH”**
3. Finish install, then **close and reopen** VS Code
4. In a new terminal try:

```bash
python --version
python3 --version
```

On Windows, `python` often works even when `python3` does not. Use whichever prints a version.

### Linux (Ubuntu/Debian example)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

---

## 3. (Recommended) Create a virtual environment

This keeps project packages separate from the rest of your computer.

From the **project root** (`pianobench`):

```bash
python3 -m venv .venv
```

Activate it:

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```bat
.\.venv\Scripts\activate.bat
```

When it works, your prompt often starts with `(.venv)`.

To leave the virtual environment later:

```bash
deactivate
```

---

## 4. Install packages (needed for real audio scores)

The evaluator runs without extra packages only for video/AV
stubs, but **audio accuracy** needs packages listed in `requirements.txt`
(`librosa`, `soundfile`, `numpy`) plus system **ffmpeg**.

When you are ready (recommended for real audio scoring):

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Also ensure `ffmpeg` is on your PATH (`brew install ffmpeg` on macOS).

---

## 5. Smoke test

Still in the project root:

```bash
python3 -c "print('Environment OK')"
python3 evaluation/evaluate.py
```

You should see a small score table and messages like:

```text
Wrote .../evaluation/results/scores.json
Wrote .../evaluation/results/scores.csv
```

---

## Common problems

| Problem | What to try |
|---------|-------------|
| `python3: command not found` | Install Python; reopen terminal; on Windows try `python` |
| `No module named ...` | Activate `.venv`, then `pip install -r requirements.txt` |
| Script can’t find videos | Make sure your terminal is in `pianobench` (run `pwd` / `cd` into the root) |
| VS Code uses the wrong Python | Command Palette → “Python: Select Interpreter” → pick `.venv` |

---

## Checkpoint

- [ ] `python3 --version` (or `python --version`) works
- [ ] (Optional) `.venv` created and activated
- [ ] `python3 evaluation/evaluate.py` runs without crashing
- [ ] `evaluation/results/scores.csv` exists

**Next:** [03-prepare-data.md](03-prepare-data.md)
