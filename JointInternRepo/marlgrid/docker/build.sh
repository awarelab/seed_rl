#!/usr/bin/env bash

set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/..
GRPC_IMAGE=gcr.io/seedimages/seed:grpc

LABEL="${CONFIG}_${JOB_NAME_PREFIX}"

set -x

"$DIR/build_base.sh"

docker build --build-arg grpc_image=${GRPC_IMAGE} --build-arg seed_path=${SEED_PATH} -t seed_rl:${LABEL}  -f $DIR/Dockerfile.marlgrid_seed ..
