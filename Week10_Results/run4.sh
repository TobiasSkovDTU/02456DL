#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J myJob
#BSUB -n 1
#BSUB -W 23:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

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
python3 getting_started_ppo4.py > out4.txt




