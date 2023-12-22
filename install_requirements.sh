#!/bin/bash

# Install packages
conda install -y pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pandas==2.1.3 matplotlib==3.8.2 pyyaml==6.0.1 dotmap==1.3.30 tqdm==4.66.1 wandb==0.15.0 einops==0.7.0 openpyxl==3.1.2 scikit-learn==1.3.2 scipy==1.11.4 kornia==0.7.0 plotly==5.18.0 umap-learn==0.5.5
