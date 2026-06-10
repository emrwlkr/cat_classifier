#!/bin/bash
#SBATCH --job-name=cat-infer
#SBATCH --partition=short
#SBATCH --output=slurm/output/infer_%j.out


source startup_script.sh

python cat_classifier_inf.py \
    --model models/cat_classifier.joblib
