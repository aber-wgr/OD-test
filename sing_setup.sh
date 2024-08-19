#!/bin/bash

echo "running script"
source /.venv/bin/activate
python3.11 setup/model_setup.py --batch-size 64 --exp model_ref --workers 1 --save --no-wandb 
