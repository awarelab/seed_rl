#!/usr/bin/env bash

module load plgrid/tools/python/3.6.5
ENV_DIR=~/mrunner_example_env
rm -rf $ENV_DIR
python3 -m venv $ENV_DIR
source $ENV_DIR/bin/activate
pip install -r requirements.txt
