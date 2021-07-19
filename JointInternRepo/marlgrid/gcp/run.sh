#!/bin/bash

set -e

if [ "$#" -ne "7" ]; then
    echo "JOB_NAME_PREFIX BUCKET_NAME AI_PLATFORM_CONFIG_FILE ENV_CONFIG_FILE ALGORITHM WORKERS ACT_PER_W"
    exit 1
fi

export JOB_NAME_PREFIX="$1"
BUCKET_NAME="$2"
AI_PLATFORM_CONFIG_FILE="$3"
ENV_CONFIG_FILE="$4"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

CONFIG_OUT_FILE="/tmp/${JOB_NAME_PREFIX}.yaml"

export CONFIG=marlgrid
export ENVIRONMENT=marlgrid
export AGENT="$5"
export WORKERS="$6"
export ACTORS_PER_WORKER="$7"

PROJECT_ID=$(gcloud config get-value project)
export IMAGE_URI=gcr.io/$PROJECT_ID/seed

start_training () {
  DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
  $DIR/../docker/build.sh
  $DIR/../docker/push.sh
  # Create bucket if doesn't exist.
  gsutil ls "gs://${BUCKET_NAME}" || gsutil mb "gs://${BUCKET_NAME}"
  JOB_NAME="${JOB_NAME_PREFIX}"
  # Start training on AI platform.
  gcloud beta ai-platform jobs submit training ${JOB_NAME} \
    --project=${PROJECT_ID} \
    --job-dir "gs://${BUCKET_NAME}/${JOB_NAME}" \
    --region "us-central1" \
    --config "$CONFIG_OUT_FILE" \
    --stream-logs -- --environment=${ENVIRONMENT} --agent=${AGENT} \
    --actors_per_worker=${ACTORS_PER_WORKER} --workers=${WORKERS} --
}


#ENV_CONFIG=$(python3 $DIR/../l/config_encoder.py --f "$ENV_CONFIG_FILE")


sed "s|ENV_CONFIG_HERE|${ENV_CONFIG}|g" "$AI_PLATFORM_CONFIG_FILE" > "$CONFIG_OUT_FILE"
sed -i="" "s|\${IMAGE_URI}|${IMAGE_URI}|g" "$CONFIG_OUT_FILE"
sed -i="" "s|\${CONFIG}|${CONFIG}_${JOB_NAME_PREFIX}|g" "$CONFIG_OUT_FILE"
sed -i="" "s|\${WORKERS}|${WORKERS}|g" "$CONFIG_OUT_FILE"

start_training
