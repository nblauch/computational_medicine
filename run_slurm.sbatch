#!/bin/bash -l

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=blauch@cmu.edu
#SBATCH --exclude=mind-1-1,mind-1-3,mind-1-5,mind-1-7

cd ~/git/computational_medicine
conda activate pycortex

echo "$COMMAND"
$COMMAND
