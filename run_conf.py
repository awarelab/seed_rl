from mrunner.helpers.specification_helper import create_experiments_helper
import os

tags = os.environ["PROJECT_TAG"].split(' ') if "PROJECT_TAG" in os.environ.keys() else []

experiments_list = create_experiments_helper(
    experiment_name='Test',
    base_config={
        # Parameters of dataset:
    },
    params_grid={},
    #project=os.environ["NEPTUNE_PROJECT_NAME"],
    script='./seed_rl/docker/run_remote.sh starcraft vtrace 3',
    exclude=['.pytest_cache', '.git', 'docs', 'data', 'data_out', 'assets', 'out', '.vagrant'],
    python_path=':ncc',
    with_mpi=False,
    tags=tags,
    callbacks=[],
    with_neptune=True
)
