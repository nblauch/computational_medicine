
import os
import sys
import glob
from helpers import BIDS_DIR

# here we want the directory above the normal BIDS_DIR
bids_dir = os.path.dirname(BIDS_DIR)

# initial data acq
os.system(f"aws s3 sync s3://openneuro/ds000030/ds000030_R1.0.5/uncompressed/derivatives/fmriprep/ {bids_dir}/ds000030/derivatives/fmriprep/ --exclude='*' --include='*fsaverage5*'")

for sub in [os.path.basename(folder) for folder in glob.glob('/mnt/d/bids/ds000030/derivatives/fmriprep/*')]:
    os.system(f"aws s3 sync s3://openneuro/ds000030/ds000030_R1.0.5/uncompressed/derivatives/fmriprep/{sub}/func/ {bids_dir}/ds000030/derivatives/fmriprep/{sub}/func/ --exclude='*' --include='*confounds*'")

os.system(f"aws s3 sync s3://openneuro/ds000030/ds000030_R1.0.5/uncompressed/phenotype {bids_dir}/ds000030/")