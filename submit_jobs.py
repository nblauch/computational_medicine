
import os
from helpers import BIDS_DIR, SUBJECTS
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true')
parser.add_argument('--no-slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--workers', type=int, default=8)
args = parser.parse_args()

resources=f'-p plaut --cpus-per-task={args.workers} --mem=16GB'
debug = '-m pdb ' if args.debug else ''

for ii, sub in enumerate(SUBJECTS):
    COMMAND=f"python {debug}do_sub_ce.py --sub {sub} --workers {args.workers}"
    print(COMMAND)
    if args.dry_run:
        continue
    if args.no_slurm:
        os.system(COMMAND)
    else:
        os.system(f"sbatch --export=\"COMMAND={COMMAND}\" --job-name CNP --time 1:00:00 {resources} --output=log/%j.log run_slurm.sbatch")
    
    # when we are debugging, typically don't want to have to loop through all the jobs
    if args.debug and ii > 0:
        sys.exit()