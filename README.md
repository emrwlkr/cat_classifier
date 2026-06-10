# Research software engineering class: cat-classifier

A minimal sklearn classifier that distinguishes between two cats from photos.
For the VS Code + Git + GitHub workshop.

> **Note on data:** The `data/` folder is included in this repository for the workshop. In a real project, data belongs in a dedicated data repository or object store (e.g. Zenodo) and would be listed in `.gitignore`.

---

## Workshop walkthrough

### 1. Fork the repository on GitHub

Go to the repository page on GitHub and click **Fork** (top-right corner).  
This creates your own copy of the repo under your GitHub account — you can push to it freely without affecting the original.

---

### 2. Copy your fork's SSH clone URL

On your fork's GitHub page click the green **Code** button → **SSH** tab → copy the URL.  
It will look like:

```
git@github.com:<your-username>/cat_classifier.git
```

> If you have not set up an SSH key with GitHub yet, see the troubleshooting section at the bottom.

---

### 3. Clone the fork in your terminal

cd into your group user directory on bmrc:
```bash
cd /well/{GROUP_NAME}/users/{USER_NAME}
```

Then clone your fork using the SSH URL you copied:

```bash
git clone git@github.com:<your-username>/cat_classifier.git
cd cat_classifier
```

---

### 4. Create a virtual environment and install dependencies

> **BMRC note:** The BMRC cluster runs Linux. Even if you are connecting from a Windows machine, use the Linux/Mac commands below.

**Linux / Mac / WSL**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (cmd)**

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

### 5. Select the Python interpreter in VS Code

You can also activate the correct environment through VS Code:

1. Open the Command Palette: `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac).
2. Type **Python: Select Interpreter**.
3. Choose the `.venv` entry — it should show something like `./.venv/bin/python`.

VS Code will then use this interpreter for the integrated terminal, IntelliSense, and the debugger.

Note: can also call the venv something else (e.g. cat-env) so clear on the interpreter.

---

### 6. The startup script (BMRC / HPC)

On the BMRC cluster (and other module-based HPC systems) you need to load the right Python module before activating your venv. Building a `startup_script.sh` does this all in one place, so you don't have to remember the module commands every time:

```bash
#!/usr/bin/bash
module purge
module load Python/3.11.3-GCCcore-12.3.0

source /gpfs3/well/{GROUP_NAME}/users/{USER_NAME}/cat_classifier/.venv/bin/activate
```

**Update the path** on the last line to match your own group and username before using it.

Source it at the beginning of any interactive session or put it at the top of every batch script:

```bash
source startup_script.sh
```

This means all your environment setup lives in one file — change it once and every script picks up the change automatically.

---

### 7. Run training and inference

#### Interactive session with `srun`

Start an interactive node on the `short` partition:

```bash
srun -p short --pty bash
```

Then, inside that session:

```bash
source startup_script.sh

# Train
python cat_classifier_train.py --data-dir data/ --model-out models/cat_classifier.joblib

# Infer
python cat_classifier_inf.py --model models/cat_classifier.joblib --image data/test/fluff/IMG_6411.jpeg
```

> **Tip:** You cannot submit `sbatch` jobs from inside an `srun` session. If you need to do both, open a second terminal and connect to a login node there.

#### Batch jobs with `sbatch`

```bash
sbatch slurm/train.sh
sbatch slurm/infer.sh
```

Each script already calls `source startup_script.sh` at the top. Logs are written to `slurm/output/`.

Example `slurm/train.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=cat-train
#SBATCH --partition=short
#SBATCH --output=slurm/output/train_%j.out

source startup_script.sh

python cat_classifier_train.py \
    --data-dir data/ \
    --model-out models/cat_classifier.joblib
```

Example scripts are given in slurm/ folder. Can edit these. Normally have these in gitignore as well so not pushed. (have hard coded file paths etc)

Check job status with `squeue --me` and cancel with `scancel <job-id>`.

---

### 8. Add a new feature, retrain, and push

edit → commit → push cycle.

#### Create and switch to a new branch

Name branch something descriptive of its purpose, e.g. v2-development, add-brightness-feature

```bash
git checkout -b add-brightness-feature
```

Checkout makes a new branch.
Can also do this in VS Code — click the branch name in the bottom-left corner and select **Create new branch**.

#### Edit `cat_classifier/features.py`

Open [cat_classifier/features.py](cat_classifier/features.py) and add a brightness feature.

e.g.
In `extract_features`, after the line that builds `arr`, add:

```python
brightness = arr.mean()
```

Then include `brightness` in the returned array:

```python
return np.array(
    [mean_r, mean_g, mean_b, width_orig, height_orig, aspect_ratio, brightness],
    dtype=np.float64,
)
```

#### Retrain the model

```bash
python cat_classifier_train.py --data-dir data/ --model-out models/cat_classifier_bright.joblib
```

#### View your changes in VS Code

1. Open the **Source Control** sidebar (`Ctrl+Shift+G` / `Cmd+Shift+G`).
2. You will see `features.py` listed under **Changes**.
3. Click the file to open a **diff view** showing exactly what you changed.

#### Commit and push — terminal

```bash
git add cat_classifier/features.py
git commit -m "Add brightness feature"
git push -u origin add-brightness-feature
```

#### Commit and push — VS Code Source Control sidebar

1. In the Source Control sidebar, click **+** next to `features.py` to stage it.
2. Type a commit message in the box at the top (e.g. `Add brightness feature`).
3. Click **Commit**.
4. Click **Sync Changes** / **Publish Branch** to push to GitHub.

---

### 9. Open a Pull Request back to the original repo

1. Go to your fork on GitHub.
2. GitHub will show a banner: **"Compare & pull request"** — click it.
3. Make sure the **base repository** is the original repo and **base branch** is `master`.
4. Add a title and description, then click **Create pull request**.

I can now view your changes :) .

---

## Code layout

```
cat_classifier/          # importable Python package
    features.py          # extract_features(path) → np.ndarray
    classify.py          # CatClassifier: fit / predict / save / load
cat_classifier_train.py  # entry point: reads data/, trains, saves model
cat_classifier_inf.py    # entry point: loads model, predicts one image
slurm/
    train.sh             # sbatch script for training
    infer.sh             # sbatch script for inference
    output/              # log files (gitignored)
data/                    # cat images (included here for teaching; normally gitignored)
startup_script.sh        # loads HPC modules + activates venv (also normally gitignored)
```

The **entry points** (top-level `*.py` scripts) are thin — they handle CLI arguments and call into the **package** (`cat_classifier/`). The package contains all reusable logic and can be imported by notebooks or other scripts without pulling in CLI machinery.

---

## Troubleshooting

### Permission denied (publickey)

Trying to push or clone over SSH but GitHub doesn't recognise your key.

```bash
ssh -T git@github.com
```

If this also fails, the key isn't set up. Follow the GitHub SSH setup guide (see the workshop PDF).

---

### fatal: not a git repository

You're running git commands in a folder that isn't a git repo.

```bash
pwd          # check where you are
ls -a        # you should see a .git folder inside the repo root
cd cat_classifier
```

---

### Please tell me who you are

Git wants a name and email before it will commit.

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

Use the same email as your GitHub account.

---

### Merge conflict

Git can't auto-merge two sets of changes to the same lines.  
open the file and look for the conflict markers:

```
<<<<<<< HEAD
your version
=======
their version
>>>>>>> branch-name
```

Edit the file to keep what you want and delete the markers, then:

```bash
git add <file>
git commit
```

---

### python: command not found

Some systems only have `python3`.

```bash
python3 -m venv .venv
```

---

### source: command not found / activate doesn't work on Windows

On Windows the activation command is different depending on your shell:

```bat
.venv\Scripts\activate        # Windows cmd
```

```powershell
.venv\Scripts\Activate.ps1    # Windows PowerShell
```

```bash
source .venv/bin/activate     # Mac / Linux / WSL
```

> **BMRC:** Always use the Linux form — even if you are connecting from Windows, the cluster is Linux.

---

### ModuleNotFoundError: No module named 'cat_classifier'

The venv isn't activated, or you're running from the wrong directory.

- Check the terminal prompt — if it doesn't start with `(.venv)`, run `source .venv/bin/activate` (or `source startup_script.sh` on BMRC).
- Check you're in the repo root: `pwd` should end in `cat_classifier`.

---

### Can't submit sbatch scripts from inside an srun session

`sbatch` must be run from a login node, not from inside an interactive `srun` job.  
Open a second terminal, SSH to a login node, and submit from there.
