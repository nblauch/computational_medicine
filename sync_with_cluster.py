
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--from-cluster', action='store_true')
args = parser.parse_args()

if args.from_cluster:
    os.system(
        f'rsync -rvae ssh $mindid:~/git/computational_medicine/figures/ ~/git/computational_medicine/figures/'
    )
    os.system(
        f'rsync -rvae ssh $mindid:~/git/computational_medicine/data/ ~/git/computational_medicine/data/'
    )
else:
    os.system(
        f'rsync -rvae ssh ~/git/computational_medicine/figures/ $mindid:~/git/computational_medicine/figures/'
    )
    os.system(
        f'rsync -rvae ssh ~/git/computational_medicine/data/ $mindid:~/git/computational_medicine/data/'
    )