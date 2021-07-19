#!/bin/bash
echo "run begins at $(pwd)"

ENVIRONMENT=$1
AGENT=$2
MRUNER_CONFIG=$4

export PYTHONPATH=$PYTHONPATH:$(pwd):/:$(pwd)/seed_rl/JointInternRepo/marlgrid/

RANDOM_PORT=$((49152 + RANDOM % (65535 - 49152)))
NONCE=id$RANDOM$RANDOM
SERVER_HOST=`python3 seed_rl/get_learner_node.py`
SERVER_ADDRESS="$SERVER_HOST:$RANDOM_PORT"
NUM_ACTORS=$((MRUNNER_NTASKS - 1))

BINARY="python3 seed_rl/${ENVIRONMENT}/${AGENT}_main.py --nonce=${NONCE} --server_address=$SERVER_ADDRESS --logtostderr --num_envs=${NUM_ACTORS} --mrunner_config=$MRUNER_CONFIG";

echo "---------------"
echo "SINGULARITY_PREFIX $SINGULARITY_PREFIX"
echo "BINARY $BINARY"
echo "SLURM_JOB_NODELIST $SLURM_JOB_NODELIST"
echo "NUM_ACTORS $NUM_ACTORS"
echo "---------------"

echo "Running the learner"
srun --ntasks 1 --mem-per-cpu 600M --nodes 1 $SINGULARITY_PREFIX $BINARY --run_mode=learner &

for ((id=0; id<$NUM_ACTORS; id++)); do
  sleep 3 # To avoid racing condition in singularity
  echo "Running an actor $id"
  srun --ntasks 1 --mem-per-cpu 600M --nodes 1 $SINGULARITY_PREFIX $BINARY --run_mode=actor --task=$id &
done

wait
