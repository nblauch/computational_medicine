# Classifying mental illness from brain data
Project for computational medicine course

To run the code, first configure directories and acquire the data
1) configure `BIDS_DIR` in `helpers.py`. this should be the directory into which `ds000030` will be downloaded.
2) run `python get_data.py`
3) download additional freesurfer derivatives from https://cmu.box.com/s/px6n005ru51lscn5m9b4jz2ckg2gsohu
4) place the `freesurfer` folder in `{BIDS_DIR}/derivatives/`
5) if access to a SLURM-based HPC is available, configure `run_slurm.sbatch` and `submit_jobs.py`

Next, run the following scripts
1) `python get_connectomes.py`
2) `python submit_jobs.py` ; if no HPC is available, run with `--no-slurm`. note this may take a very long time to run, depending on how many CPUs are available per job.

Finally, perform classification analyses using the `project.ipynb` notebook

questions: raise an issue or email blauch@cmu.edu


