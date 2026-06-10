#!/bin/bash
#SBATCH --job-name=cat-train
#SBATCH --partition=short
#SBATCH --output=slurm/output/train_%j.out

source startup_script.sh

python cat_classifier_train.py \
    --data-dir data/ \
    --model-out models/cat_classifier.joblib
