#!/usr/bin/env bash

#run as basic_setup.sh PROMETHEUS_LOGIN GRANT_NAME

set -e
export PROMETHEUS_LOGIN=$1
export GRANT_NAME=$2

function prepare_local_venv {
    #ENV_DIR=/tmp/example_venv
    #echo "=================================================================="
    #echo "Setting up local virtual env in $ENV_DIR"
    #echo "We assume python3"
    #echo "=================================================================="

    #rm -rf $ENV_DIR
    #python3 -m venv $ENV_DIR
    #source $ENV_DIR/bin/activate
    #pip install -r resources/requirements.txt
    sleep 0
}

function prepare_mrunner_config {
    #echo "=================================================================="
    #echo "Preparing mrunner config in /tmp/mrunner_config.yaml"
    #echo "=================================================================="

    sed "s/<username>/$PROMETHEUS_LOGIN/g" mrunner_resources/prometheus_config_template.yaml > /tmp/mrunner_config_1.yaml
    sed "s/<grantname>/$GRANT_NAME/g" /tmp/mrunner_config_1.yaml > /tmp/mrunner_config.yaml

    rm /tmp/mrunner_config_1.yaml

    #cat /tmp/mrunner_config.yaml
    echo "basic_setup done."
}

function prepare_envs_and_mrunner_config {
    if [ -z "$PROMETHEUS_LOGIN" ]; then echo "PROMETHEUS_LOGIN must be set. exiting";exit; fi
    if [ -z "$GRANT_NAME" ]; then echo "GRANT_NAME must be set. exiting";exit; fi
       prepare_local_venv
      prepare_mrunner_config
}

prepare_envs_and_mrunner_config;
