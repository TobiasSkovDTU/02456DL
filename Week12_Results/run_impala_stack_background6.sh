#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J run_impala_stack_background6
#BSUB -n 1
#BSUB -W 20:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o run_impala_stack_background6.out
#BSUB -e run_impala_stack_background6.err

module load python3/3.8.0 
module load cuda/8.0 
module load cudnn/v7.0-prod-cuda8 
module load ffmpeg/4.2.2 

pip3 install --user torch torchvision matplotlib seaborn 
pip3 install --user procgen 
pip3 install --user gym 

lscpu 
nvidia-smi 

echo "Running script..."
python3 train_and_save_checkpoint_impala_stack_background6.py > out_impala_stack_background6.txt




