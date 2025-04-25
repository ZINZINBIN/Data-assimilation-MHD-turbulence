#!/bin/bash
#SBATCH --job-name=DA-MHD        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=4               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)

#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=jk9075@princeton.edu
#SBATCH --error=job.err
#SBATCH --output=job.out

srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=8G python3 simulate_enkf_rho.py &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=8G python3 simulate_enkf_v.py   &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=8G python3 simulate_enkf_P.py   &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=8G python3 simulate_enkf_B.py   &
wait