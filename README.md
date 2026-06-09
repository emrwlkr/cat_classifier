# Research software engineering class: cat-classifier

A minimal sklearn classifier that distinguishes between two cats from photos
Built as a teaching example for a VS Code + Git + GitHub workshop

---

## Install

```bash
git clone <your-repo-url> cat-classifier
cd cat-classifier
python -m venv .venv
pip install -r requirements.txt
```

## Environment setup (Oxford BMRC / any module-based HPC)

Source `startup_script.sh` instead of activating the venv directly:
Allows for venv activation and building required modules

```bash
source startup_script.sh
```

This runs `module purge`, loads `Python/3.11.3-GCCcore-12.3.0`, then activates
`.venv`. 

Run at the start of srun or put at the start of any bash script

## Train

```bash
python cat_classifier_train.py
# or override defaults:
python cat_classifier_train.py --data-dir data/
```

## Predict

```bash
python cat_classifier_inf.py --image data/fluff/photo1.jpeg
# → Predicted: cat_a (87% confident)
```

## Run on SLURM

```bash
sbatch slurm/train.sh
sbatch slurm/infer.sh --image data/fluff/photo1.jpeg
```

Logs land in `slurm/output/` (gitignored).

---

## Code layout

```
cat_classifier/          # importable Python package
│   features.py          # extract_features(path) → np.ndarray
│   classify.py          # CatClassifier: fit / predict / save / load
cat_classifier_train.py  # entry point: reads data/, trains, saves model
cat_classifier_inf.py    # entry point: loads model, predicts one image
```

The **entry points** (top-level `*.py` scripts) are thin — they handle CLI
arguments and orchestrate calls into the **package** (`cat_classifier/`).
The package contains all reusable logic and can be imported by notebooks or
other scripts without pulling in any CLI machinery.
