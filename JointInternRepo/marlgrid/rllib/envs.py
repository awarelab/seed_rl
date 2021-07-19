from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gym

from marlgrid.custom.wrappers import ListOfDiscreteToMultiDiscrete, RewardForMultiHead, ObservationForMultiHead


def generate_agent_names(num_agents):
    result = []
    for i in range(num_agents):
        result.append(f"agent_{i}")
    return result

def get_player_name_from_id(id):
    return f"agent_{id}"

def get_agent_id_from_name(player_name):
    return int(player_name.split('_')[-1])



class RllibMarlGridBase(MultiAgentEnv):
  def __init__(self, env, num_players, player_names):
    self.env = env
    self.num_players = num_players
    self.player_names = player_names

  def reset(self):
    original_obs = self.env.reset()
    return dict(zip(self.player_names, original_obs))

  def step(self, action_dict):
    actions = []
    for k, a in sorted(action_dict.items(), key=lambda x: get_agent_id_from_name(x[0])):
      actions.append(a)
    obs, reward, done, info = self.env.step(actions)

    new_obs = dict(zip(self.player_names, obs))
    new_rew = dict(zip(self.player_names, reward))
    new_done = {'__all__': done}
    new_info = dict(zip(self.player_names, info))

    return new_obs, new_rew, new_done, new_info

class RllibOneHead(RllibMarlGridBase):
  def __init__(self, env_name):
    env = gym.make(env_name)
    num_players = len(env.observation_space)
    player_names = generate_agent_names(num_players)
    super(RllibOneHead, self).__init__(env=env,
                                        num_players=num_players,
                                        player_names=player_names)

    

    self.action_space = dict(zip(self.player_names, self.env.action_space))

    self.observation_space = dict(zip(self.player_names, self.env.observation_space))


class RayMultiHeadMinor(gym.Wrapper):
  def __init__(self, env):
    env = ListOfDiscreteToMultiDiscrete(gym.make(env_name))
    env = RewardForMultiHead(env)
    env = ObservationForMultiHead(env)

    gym.Wrapper.__init__(self, env)
    self.action_space = [self.env.action_space]
    self.observation_space = [self.env.observation_space]

  def convert_obs(self, org_obs):
    return [np.squeeze(org_obs, axis=0)]

  def reset(self):
    org_obs = self.env.reset()
    return self.conveconvert_obs(org_obs)

  def step(self, action):
    obs, rew, done, info = self.env.step(action)

    return self.conveconvert_obs(obs), rew, done, info

class RllibMultiHead(RllibMarlGridBase):

  def __init__(self, env_name):
    env = RayMultiHeadMinor(env)
    num_players = 1
    player_names = generate_agent_names(self.num_players)
    super(RllibMultiHead, self).__init__(env=env,
                                          num_players=num_players,
                                          player_names=player_names)
    

    self.action_space = dict(zip(self.player_names, self.env.action_space))

    self.observation_space = dict(zip(self.player_names, self.env.observation_space))