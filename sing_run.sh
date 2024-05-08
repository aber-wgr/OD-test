#!/bin/bash

echo "running script"
source /.venv/bin/activate
python3.11 eval3d.py -exp master --batch-size 32 --seed 27 --workers 1 --save --wandb-project ODTest2-Run 
