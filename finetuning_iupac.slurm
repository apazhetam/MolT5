#!/bin/bash
#SBATCH --job-name=finetuning_t5_iupac   # create a short name for your job
#SBATCH --output=/home/jomeike/projects/cos484/slurms/slurm-%A.%a.out # stdout file
#SBATCH --error=/home/jomeike/projects/cos484/slurms/slurm-%A.%a.err  # stderr file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fault
#SBATCH --mem-per-cpu=36G
#SBATCH --mail-user=jomeike@princeton.edu
echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)
module purge
module load anaconda3/2024.2 cudatoolkit/11.7 cudnn/cuda-11.x/8.2.0
conda activate cuda-torch-env # this is cuda 11.7

python3 -u /home/jomeike/projects/cos484/slurms/finetuning_t5_iupac.py