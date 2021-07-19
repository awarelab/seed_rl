#!/bin/bash

set -e
die () {
    echo >&2 "$@"
    exit 1
}

ENVIRONMENTS="atari|dmlab|football|marlgrid|particles"
AGENTS="r2d2|vtrace|sac"

echo $1 | grep -E -q $ENVIRONMENTS || die "Supported games: $ENVIRONMENTS"
echo $2 | grep -E -q $AGENTS || die "Supported agents: $AGENTS"

ENVIRONMENT=$1
AGENT=$2

cd ..
export PYTHONPATH=$(pwd):$PYTHONPATH; python3 seed_rl/$ENVIRONMENT/${AGENT}_main.py --run_mode=visualize 
