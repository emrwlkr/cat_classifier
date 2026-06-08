# cat-classifier

A minimal sklearn classifier that distinguishes between two cats from photos.
Built as a teaching example for a VS Code + Git + GitHub workshop — the focus is
on clean repo structure, not ML sophistication.

---

## Install

```bash
git clone <your-repo-url> cat-classifier
cd cat-classifier
python -m venv .venv
pip install -r requirements.txt
```

## Environment setup (Oxford BMRC / any module-based HPC)

The venv's Python requires `libffi` from the module system. Source
`startup_script.sh` instead of activating the venv directly:

```bash
source startup_script.sh
```

This runs `module purge`, loads `Python/3.11.3-GCCcore-12.3.0`, then activates
`.venv`. The SLURM scripts already do this for you.

## Add your data

Place JPEG photos under:

```
data/cat_a/   ← photos of cat A
data/cat_b/   ← photos of cat B
```

The `data/` directory is gitignored — photos are never committed.

## Train

```bash
python cat_classifier_train.py
# or override defaults:
python cat_classifier_train.py --data-dir data/ --model-out models/cat_classifier.joblib
```

## Predict

```bash
python cat_classifier_inf.py --image data/cat_a/photo1.jpeg
# → Predicted: cat_a (87% confident)
```

## Run on SLURM

```bash
sbatch slurm/train.sh
sbatch slurm/infer.sh --image data/cat_a/photo1.jpeg
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
