#!/bin/bash

git clone --depth=1 https://github.com/uwfiberlab/sz4d_das_workshop.git
cd sz4d_das_workshop

source /srv/conda/etc/profile.d/conda.sh
conda create --name workshop python=3.10 h5py scipy numpy pandas matplotlib pytorch-cpu libcomcat -y
conda activate workshop
pip install noisepy-seis ipykernel tqdm ELEP joblib seisbench geopy
python -m ipykernel install --user --name=workshop

# copying DAS data from our S3 bucket.
aws s3 cp --recursive --no-sign-request s3://shared/niyiyu/sz4d-das-workshop/ ./ --endpoint https://dasway.ess.washington.edu
