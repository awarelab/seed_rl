
import ray
from ray import tune
from envs import RllibOneHead, get_player_name_from_id
from ray.tune.registry import register_env
from gfootball_model.ray_api import MarlGridRayGFootballNoPackBits


#args = parser.parse_args()
ray.init(num_gpus=1)

# Simple environment with `num_agents` independent players
#register_env('gfootball', lambda _: RllibGFootball(args.num_agents))
single_env = RllibOneHead("MarlGrid-3AgentCluttered15x15-v0")
obs_space = single_env.observation_space
act_space = single_env.action_space

def gen_policy(i):
    player_name = get_player_name_from_id(i)
    return (None, obs_space[player_name], act_space[player_name], {})


num_policies = 1

# Setup PPO with an ensemble of `num_policies` different policies
policies = {
    'policy_{}'.format(i): gen_policy(i) for i in range(num_policies)
}
policy_names = list(policies.keys())

register_env('ray_marl_grid', lambda _: RllibOneHead("MarlGrid-3AgentCluttered15x15-v0"))

tune.run(
    'PPO',
    stop={'training_iteration': 100000},
    checkpoint_freq=50,
    config={
        "model": {
        "custom_model": MarlGridRayGFootballNoPackBits,
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {},
         },
        'env': 'ray_marl_grid',
        'lambda': 0.95,
        'kl_coeff': 0.2,
        'clip_rewards': False,
        'vf_clip_param': 10.0,
        'entropy_coeff': 0.01,
        'train_batch_size': 32,
        'sample_batch_size': 32,
        'sgd_minibatch_size': 32,
        'num_sgd_iter': 1,
        'num_workers': 2,
        'num_envs_per_worker': 8,
        'batch_mode': 'truncate_episodes',
        'observation_filter': 'NoFilter',
        'vf_share_layers': 'true',
        'num_gpus': 0,
        'lr': 2.5e-4,
        'log_level': 'DEBUG',
        #'simple_optimizer': args.simple,
        'multiagent': {
            'policies': policies,
            'policy_mapping_fn': tune.function(
                lambda _: policy_names[0]),
        },
    },
)