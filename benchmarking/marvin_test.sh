#!/bin/bash

#SBATCH --partition=haswell
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu 8G # mem to user for each core
#SBATCH -t 0-1:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --mail-type=START,END,FAIL # notifications for job start, done & fail
#SBATCH --mail-user=david.ampudia@upf.edu # send-to address
 
# run the application
module load CUDA
module load Python
DIR="/homes/users/dampudia"

source $DIR/trieste/.trieste/bin/activate
python $DIR/trieste/benchmarking/minitest_parallel_benchmarking.py