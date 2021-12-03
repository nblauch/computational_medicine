
import os
import sys
import glob

# configure as needed
bids_dir='/mnt/d/bids'

# initial data acq
os.system(f"aws s3 sync s3://openneuro/ds000030/ds000030_R1.0.5/uncompressed/derivatives/fmriprep/ {bids_dir}/ds000030/derivatives/fmriprep/ --exclude='*' --include='*fsaverage5*'")

for sub in [os.path.basename(folder) for folder in glob.glob('/mnt/d/bids/ds000030/derivatives/fmriprep/*')]:
    os.system(f"aws s3 sync s3://openneuro/ds000030/ds000030_R1.0.5/uncompressed/derivatives/fmriprep/{sub}/func/ {bids_dir}/ds000030/derivatives/fmriprep/{sub}/func/ --exclude='*' --include='*confounds*'")

os.system(f"aws s3 sync s3://openneuro/ds000030/ds000030_R1.0.5/uncompressed/phenotype {bids_dir}/ds000030/")