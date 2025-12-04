#!/bin/bash

wget https://github.com/uwfiberlab/sz4d_das_workshop/archive/refs/heads/main.zip
unzip main.zip
rm main.zip
cd sz4d_das_workshop-main

conda create --name workshop python=3.10 h5py scipy numpy pandas matplotlib pytorch-cpu -y
conda activate workshop
pip install noisepy-seis ipykernel tqdm
python -m ipykernel install --user --name=workshop

# copying DAS data from our S3 bucket. Should take about 2-4 minutes to complete.
aws s3 cp --recursive --no-sign-request s3://2025-sz4d-das-workshop/ ./