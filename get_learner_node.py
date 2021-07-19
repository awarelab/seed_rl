import os

nodelist = os.environ['SLURM_JOB_NODELIST']

if "[" in nodelist:
  x = nodelist.find("[")
  nodes_nums = nodelist[x+1:-1]
  print(nodelist[:x] + nodes_nums.split(",")[0].split("-")[0])
else:
  print(nodelist)
