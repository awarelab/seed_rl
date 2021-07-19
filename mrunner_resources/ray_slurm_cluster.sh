#!/usr/bin/env bash
export LC_ALL=en_GB.utf8

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
node1=${nodes_array[0]}
worker_num=${#nodes_array[@]}

echo "Starting ray cluster with $worker_num nodes"
echo "Nodes are ======= $nodes_array ====="

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
echo $ip_prefix

suffix=':6379'
ip_head=$ip_prefix$suffix
export redis_password=$(uuidgen)

export ip_head # Exporting for latter access by trainer.py
echo "Starting head"
srun --nodes=1 --ntasks=1 -w $node1 $SINGULARITY_PREFIX ray start --block --head --redis-port=6379 --redis-password=$redis_password & # Starting the head
sleep 10

echo "Starting workers"
for ((  i=1; i<$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 $SINGULARITY_PREFIX ray start --block --address=$ip_head --redis-password=$redis_password & # Starting the workers
  sleep 5
done

echo "Starting ray cluster has succeeded"