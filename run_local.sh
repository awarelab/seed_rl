#!/bin/bash
# Copyright 2021 Anonymous authors of "Off-Policy Correction For Multi-Agent Reinforcement Learning"
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -e
die () {
    echo >&2 "$@"
    exit 1
}

ENVIRONMENTS="atari|dmlab|football|marlgrid|particles|starcraft"
AGENTS="r2d2|vtrace|sac"
[ "$#" -ne 0 ] || die "Usage: run_local.sh [$ENVIRONMENTS] [$AGENTS] [Num. actors] [configuration flags]"
echo $1 | grep -E -q $ENVIRONMENTS || die "Supported games: $ENVIRONMENTS"
echo $2 | grep -E -q $AGENTS || die "Supported agents: $AGENTS"
echo $3 | grep -E -q "^((0|([1-9][0-9]*))|(0x[0-9a-fA-F]+))$" || die "Number of actors should be a non-negative integer without leading zeros"
export ENVIRONMENT=$1
export AGENT=$2
export NUM_ACTORS=$3
export CONFIG=$ENVIRONMENT
export CONFIG_FLAGS=$4

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
docker/build.sh
docker_version=$(docker version --format '{{.Server.Version}}')
if [[ "19.03" > $docker_version ]]; then
  docker run --entrypoint ./docker/run.sh -ti -it --name seed --rm seed_rl:$ENVIRONMENT $ENVIRONMENT $AGENT $NUM_ACTORS "$CONFIG_FLAGS"
else
  docker run --entrypoint ./docker/run.sh -ti -it -e HOST_PERMS="$(id -u):$(id -g)" --name seed --rm seed_rl:$ENVIRONMENT $ENVIRONMENT $AGENT $NUM_ACTORS "$CONFIG_FLAGS"
fi
