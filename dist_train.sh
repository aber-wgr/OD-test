#!/bin/bash

#partition
#SBATCH --partition=gpu
#gpus
#SBATCH --gres=gpu:2
#job name
#SBATCH --job-name=OD-test2
# job stdout file
#SBATCH --output=odtest2.out.%J
# job stderr file
#SBATCH --error=odtest2.err.%J
# maximum job time in D-HH:MM
#SBATCH --time=2-00:00
# number of tasks you are requesting
# memory per process in MB
#SBATCH --mem=8192
# number of nodes needed
#SBATCH --nodes=1
#number of tasks per each node
#SBATCH --ntasks-per-node=1
# specify our current project
#SBATCH --account=scw1744

module load CUDA/11.7
module load singularity/3.8.5

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# zoom zoom - recommended from lightning
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32

echo "Run Started at:- "
date

srun singularity exec --nv ~/Containers/Singularity_ODTest2.sif /bin/bash ./sing_setup.sh

echo "Run Finished at:- "
date
