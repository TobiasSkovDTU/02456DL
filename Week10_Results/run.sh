#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J myJob
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load python3/3.8.0 > out.txt
module load cuda/8.0 > out.txt
module load cudnn/v7.0-prod-cuda8 > out.txt
module load ffmpeg/4.2.2 > out.txt

pip3 install --user torch torchvision matplotlib seaborn > out.txt
pip3 install --user procgen > out.txt
pip3 install --user gym > out.txt

lscpu > out.txt
nvidia-smi > out.txt

echo "Running script..."
python3 getting_started_ppo.py > out.txt




