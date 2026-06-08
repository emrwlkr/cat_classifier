#!/usr/bin/env bash
#SBATCH --job-name=cat-infer
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm/output/infer_%j.out
#SBATCH --error=slurm/output/infer_%j.err

# Usage: sbatch slurm/infer.sh --image data/cat_a/photo1.jpg
set -euo pipefail

source startup_script.sh

python cat_classifier_inf.py \
    --model models/cat_classifier.joblib \
    "$@"
