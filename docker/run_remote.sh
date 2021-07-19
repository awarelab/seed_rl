#!/bin/bash
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


die () {
    echo >&2 "$@"
    exit 1
}

echo "run begins"
pwd

EXPDIR=`pwd`

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

echo "in DIR"

ENVIRONMENT=$1
AGENT=$2
NUM_ACTORS=$3
MRUNER_CONFIG=$5
shift 5

echo "args read"

cd ../..
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd $DIR

echo "PYTHONPATH updated"

python3 -m pip install git+https://gitlab.com/awarelab/mrunner

#source /home/mizaw/venvs/py38_seedrl/bin/activate

export PYTHONPATH=$PYTHONPATH:/

NONCE=id$RANDOM$RANDOM$RANDOM

echo "nonce created"

export NEPTUNE_PROJECT_NAME="do-not-be-hasty/tmp"

SERVER_ADDRESS="localhost:$((49152 + RANDOM % (65535 - 49152)))"

ACTOR_BINARY="python3 ../${ENVIRONMENT}/${AGENT}_main.py --run_mode=actor --nonce=${NONCE} --server_address=$SERVER_ADDRESS --mrunner_config=$EXPDIR/$MRUNER_CONFIG";
LEARNER_BINARY="python3 ../${ENVIRONMENT}/${AGENT}_main.py --run_mode=learner --nonce=${NONCE} --server_address=$SERVER_ADDRESS --mrunner_config=$EXPDIR/$MRUNER_CONFIG";
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "running..."

rm /tmp/agent -Rf
${LEARNER_BINARY} --logtostderr --pdb_post_mortem --num_envs=${NUM_ACTORS} &
LEARNER_PID=$!

for ((id=0; id<$NUM_ACTORS; id++)); do
    echo "run actor"
    ${ACTOR_BINARY} --logtostderr --pdb_post_mortem --num_envs=${NUM_ACTORS} --task=${id} &
done

while kill -0 $LEARNER_PID
do
    sleep 60
done
