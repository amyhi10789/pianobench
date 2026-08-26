# Part 1 — Open the Project in VS Code

**Goal:** Get this project onto your computer and open it correctly in Visual Studio Code (VS Code).

**Time:** about 15–20 minutes  
**You will need:** a computer (Mac, Windows, or Linux) and an internet connection

---

## 1. Install VS Code

1. Go to [https://code.visualstudio.com/](https://code.visualstudio.com/)
2. Download VS Code for your operating system
3. Install it like a normal app
4. Open VS Code once to make sure it starts

---

## 2. Get the project folder

Pick the option that matches how you received the project.

### Option A — Folder already on your computer (USB / shared drive)

1. Copy the whole `pianobench` folder to a place you can find easily  
   Example: `Desktop/pianobench`
2. Keep the inner folder names as they are

### Option B — Download a ZIP

1. Download the project ZIP
2. Unzip it
3. You should see a folder that contains `README.md`, `docs/`, `data/`, and `evaluation/`

### Option C — Clone with Git (optional)

If the project is on GitHub/Git:

```bash
git clone <REPO_URL>
cd pianobench
```

---

## 3. Open the project in VS Code (important)

Always open the **project root** (`pianobench`), not a single file.

1. In VS Code: **File → Open Folder…**
2. Select the `pianobench` folder
3. Click **Open**

### Check that it worked

In the left **Explorer** sidebar you should see:

- `README.md`
- `docs`
- `data`
- `evaluation`
- `scripts`

If you only see one file tab and no folder tree, you probably opened a file instead of a folder. Close VS Code and open the folder again.

---

## 4. Helpful VS Code basics

| Action | Mac | Windows / Linux |
|--------|-----|-----------------|
| Open terminal | `` Ctrl+` `` or **Terminal → New Terminal** | same |
| Save file | `Cmd+S` | `Ctrl+S` |
| Command Palette | `Cmd+Shift+P` | `Ctrl+Shift+P` |

Recommended extensions (optional but useful):

1. Open Extensions (`Cmd+Shift+X` / `Ctrl+Shift+X`)
2. Search and install:
   - **Python** (Microsoft)
   - **Python Debugger** (Microsoft)

---

## 5. Open the built-in terminal

1. **Terminal → New Terminal**
2. Confirm you are in the project root. You can run:

```bash
pwd
ls
```

You should see `docs`, `data`, and `evaluation`.

---

## Checkpoint

You are done with Part 1 when all of these are true:

- [ ] VS Code is installed
- [ ] `pianobench` is open as a **folder**
- [ ] You can see `docs/`, `data/`, and `evaluation/` in Explorer
- [ ] The integrated terminal opens at the project root

**Next:** [02-environment.md](02-environment.md)
