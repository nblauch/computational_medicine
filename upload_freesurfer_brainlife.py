import os
import glob
import numpy as np
from helpers import SUBJECTS

for sub in SUBJECTS[1:]:
    subnum = sub.replace('sub-','')
    os.system(f'aws s3 sync s3://openneuro/ds000030/ds000030_R1.0.5/uncompressed/derivatives/freesurfer/{sub} $BIDS/ds000030/derivatives/freesurfer/{sub}')
    os.system(f"\
    bl data upload \
    --project 617f5909b0856ce5c6f0f142   \
    --datatype neuro/freesurfer \
    --desc 'freesurfer from openneuro R1.0.5' \
    --subject {subnum} \
    --output $BIDS/ds000030/derivatives/freesurfer/{sub} \
    "
)
    # os.system(f'aws s3 sync s3://openneuro/ds000030/ds000030_R1.0.5/uncompressed/derivatives/freesurfer/{sub} $BIDS/ds000030/derivatives/freesurfer/')

