
import argparse
from helpers import BIDS_DIR, SUBJECTS 
import numpy as np
import cepy
import os

parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=str)
parser.add_argument('--task', type=str, default='rest')
parser.add_argument('--sparsity', type=float, default=0.3)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--overwrite',action='store_true')
args = parser.parse_args()

if args.task != 'rest':
    raise NotImplementedError()

if not os.path.exists(f'{BIDS_DIR}/derivatives/python/cepy/{args.sub}_ce_fc-{args.task}_sparsity-{args.sparsity}_aligned.json') or args.overwrite:

    connectome = np.load(f'{BIDS_DIR}/derivatives/connectomes/{args.sub}/{args.task}_hcp.npy')
    connectome = np.abs(connectome)
    connectome[np.isposinf(connectome)] = 0
    connectome[connectome < np.percentile(connectome, 1-args.sparsity)] = 0
    connectome = np.maximum(connectome, connectome.T)

    ce_group = cepy.load_model(f'{BIDS_DIR}/derivatives/python/cepy/group_ce_fc-{args.task}_sparsity-{args.sparsity}.json')
    cosine_sim = ce_group.similarity()

    ce_sub = cepy.CE(permutations=1,seed=1, workers=args.workers)
    ce_sub.fit(connectome)
    ce_sub.save_model(f'{BIDS_DIR}/derivatives/python/cepy/{args.sub}_ce_fc-{args.task}_sparsity-{args.sparsity}.json')
    ce_sub_aligned = cepy.align(ce_group, ce_sub)
    ce_sub_aligned.save_model(f'{BIDS_DIR}/derivatives/python/cepy/{args.sub}_ce_fc-{args.task}_sparsity-{args.sparsity}_aligned.json')