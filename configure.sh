#!/bin/bash

git clone --depth=1 https://github.com/uwfiberlab/sz4d_das_workshop.git
cd sz4d_das_workshop

conda create --name workshop python=3.10 h5py scipy numpy pandas matplotlib pytorch-cpu -y
conda activate workshop
pip install noisepy-seis ipykernel tqdm
python -m ipykernel install --user --name=workshop

# copying DAS data from our S3 bucket. Should take about 2-4 minutes to complete.
aws s3 cp --recursive --no-sign-request s3://2025-sz4d-das-workshop/ ./
