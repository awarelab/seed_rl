from mrunner.helpers.specification_helper import create_experiments_helper
import os

tags = os.environ["PROJECT_TAG"].split(' ') if "PROJECT_TAG" in os.environ.keys() else []

smac_tasks = ['2s_vs_1sc', '2s3z', '3s5z', '1c3s5z', '10m_vs_11m', '2c_vs_64zg',
              'bane_vs_bane', '5m_vs_6m', '3s_vs_5z', '3s5z_vs_3s6z',
              '6h_vs_8z', '27m_vs_30m', 'MMM2', 'corridor']

dev_tasks = ['10m_vs_11m', '2c_vs_64zg', '3s5z_vs_3s6z', 'MMM2', 'corridor']
new_dev_tasks = ['5m_vs_6m', '6h_vs_8z', '3s5z_vs_3s6z', 'MMM2', 'corridor']

experiments_list = create_experiments_helper(
    experiment_name='MA-Trace',
    base_config={
        'task_name': None,
        'learning_rate': 0.001,
        'learning_rate_warmup': 100,
        'learning_rate_decay_scale': -1,
        'entropy_cost': 1.,
        'target_entropy': 0.00001,
        'entropy_cost_adjustment_speed': 10.,
        'full_state_critic': False,
        'is_centralized': True,
        'frames_stacked': 1,
    },
    # base_config={},
    #params_grid={},
    # params_grid={'task_name': new_dev_tasks * 5, 'learning_rate': [0.001, 0.002, 0.005], 'learning_rate_decay_scale': [0.001, 0.0005, 0.0001]},
    # params_grid={'task_name': ['3s_vs_5z'] * 10, 'target_entropy': [0.01, 0.001, 0.0001, 0.00001], 'entropy_cost_adjustment_speed': [1, 2.5, 5, 10, 15]},
    params_grid={'task_name': smac_tasks},
    # params_grid={'learning_rate': [0.001] * 10},
    #project=os.environ["NEPTUNE_PROJECT_NAME"],
    # script='./seed_rl/docker/run_remote.sh starcraft vtrace_better 30',
    script='./seed_rl/docker/run_learner_actor.sh starcraft vtrace_better',
    # script='./seed_rl/docker/run_learner_actor.sh marlgrid vtrace_better',
    exclude=['.pytest_cache', '.git', 'docs', 'data', 'data_out', 'assets', 'out', '.vagrant', 'seed_rl/.git'],
    python_path=':ncc',
    with_mpi=False,
    tags=tags,
    callbacks=[],
    with_neptune=True,
    with_srun=True,
    project_name='pmtest/marl-vtrace'
)
