#!/usr/bin/env bash
#SBATCH --job-name=cat-train
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm/output/train_%j.out
#SBATCH --error=slurm/output/train_%j.err

set -euo pipefail

source startup_script.sh

python cat_classifier_train.py \
    --data-dir data/ \
    --model-out models/cat_classifier.joblib
