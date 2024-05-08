#!/bin/bash

echo "running script"
source /.venv/bin/activate
python3.11 setup/model_setup.py --exp model_ref --save --no-wandb
