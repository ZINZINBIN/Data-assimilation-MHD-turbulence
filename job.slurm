#!/bin/bash
#SBATCH --job-name=DA-MHD        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=4               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)

#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=jk9075@princeton.edu
#SBATCH --error=job.err
#SBATCH --output=job.out

# num ensemble
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_n.py --num_data_point 100 --num_ensemble 50 --sigma_x 0.01 --sigma_z 0.01 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_v.py --num_data_point 100 --num_ensemble 50 --sigma_x 0.01 --sigma_z 0.01 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_P.py --num_data_point 100 --num_ensemble 50 --sigma_x 0.01 --sigma_z 0.01 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_B.py --num_data_point 100 --num_ensemble 50 --sigma_x 0.01 --sigma_z 0.01 &

srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_n.py --num_data_point 100 --num_ensemble 100 --sigma_x 0.01 --sigma_z 0.01 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_v.py --num_data_point 100 --num_ensemble 100 --sigma_x 0.01 --sigma_z 0.01 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_P.py --num_data_point 100 --num_ensemble 100 --sigma_x 0.01 --sigma_z 0.01 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_B.py --num_data_point 100 --num_ensemble 100 --sigma_x 0.01 --sigma_z 0.01 &

srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_n.py --num_data_point 100 --num_ensemble 200 --sigma_x 0.01 --sigma_z 0.01 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_v.py --num_data_point 100 --num_ensemble 200 --sigma_x 0.01 --sigma_z 0.01 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_P.py --num_data_point 100 --num_ensemble 200 --sigma_x 0.01 --sigma_z 0.01 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_B.py --num_data_point 100 --num_ensemble 200 --sigma_x 0.01 --sigma_z 0.01 &

srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_n.py --num_data_point 100 --num_ensemble 400 --sigma_x 0.01 --sigma_z 0.01 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_v.py --num_data_point 100 --num_ensemble 400 --sigma_x 0.01 --sigma_z 0.01 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_P.py --num_data_point 100 --num_ensemble 400 --sigma_x 0.01 --sigma_z 0.01 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_B.py --num_data_point 100 --num_ensemble 400 --sigma_x 0.01 --sigma_z 0.01 &

# num datapoint
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_n.py --num_data_point 50 --num_ensemble 100 --sigma_x 0.01 --sigma_z 0.01 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_v.py --num_data_point 50 --num_ensemble 100 --sigma_x 0.01 --sigma_z 0.01 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_P.py --num_data_point 50 --num_ensemble 100 --sigma_x 0.01 --sigma_z 0.01 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_B.py --num_data_point 50 --num_ensemble 100 --sigma_x 0.01 --sigma_z 0.01 &

srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_n.py --num_data_point 200 --num_ensemble 100 --sigma_x 0.01 --sigma_z 0.01 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_v.py --num_data_point 200 --num_ensemble 100 --sigma_x 0.01 --sigma_z 0.01 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_P.py --num_data_point 200 --num_ensemble 100 --sigma_x 0.01 --sigma_z 0.01 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_B.py --num_data_point 200 --num_ensemble 100 --sigma_x 0.01 --sigma_z 0.01 &

srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_n.py --num_data_point 400 --num_ensemble 100 --sigma_x 0.01 --sigma_z 0.01 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_v.py --num_data_point 400 --num_ensemble 100 --sigma_x 0.01 --sigma_z 0.01 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_P.py --num_data_point 400 --num_ensemble 100 --sigma_x 0.01 --sigma_z 0.01 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_B.py --num_data_point 400 --num_ensemble 100 --sigma_x 0.01 --sigma_z 0.01 &

# noise parameter
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_n.py --num_data_point 100 --num_ensemble 100 --sigma_x 0.1 --sigma_z 0.1 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_v.py --num_data_point 100 --num_ensemble 100 --sigma_x 0.1 --sigma_z 0.1 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_P.py --num_data_point 100 --num_ensemble 100 --sigma_x 0.1 --sigma_z 0.1 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_B.py --num_data_point 100 --num_ensemble 100 --sigma_x 0.1 --sigma_z 0.1 &

srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_n.py --num_data_point 100 --num_ensemble 100 --sigma_x 0.001 --sigma_z 0.001 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_v.py --num_data_point 100 --num_ensemble 100 --sigma_x 0.001 --sigma_z 0.001 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_P.py --num_data_point 100 --num_ensemble 100 --sigma_x 0.001 --sigma_z 0.001 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_B.py --num_data_point 100 --num_ensemble 100 --sigma_x 0.001 --sigma_z 0.001 &

srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_n.py --num_data_point 100 --num_ensemble 100 --sigma_x 0.0001 --sigma_z 0.0001 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_v.py --num_data_point 100 --num_ensemble 100 --sigma_x 0.0001 --sigma_z 0.0001 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_P.py --num_data_point 100 --num_ensemble 100 --sigma_x 0.0001 --sigma_z 0.0001 &
srun --ntasks=1 --exclusive --cpus-per-task=4 --mem-per-cpu=4G python3 simulate_enkf_B.py --num_data_point 100 --num_ensemble 100 --sigma_x 0.0001 --sigma_z 0.0001 &
wait