# coding=utf-8
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


"""Environment wrappers."""
import sys

from absl import flags
import gym
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS


def spec_to_box(spec):
  minimum, maximum = -np.inf, np.inf
  if hasattr(spec, 'minimum'):
    if not hasattr(spec, 'maximum'):
      raise ValueError('spec has minimum but no maximum: {}'.format(spec))
    minimum = np.array(spec.minimum, np.float32)
    maximum = np.array(spec.maximum, np.float32)
    return gym.spaces.Box(minimum, maximum)

  return gym.spaces.Box(-np.inf, np.inf, shape=spec.shape)


def flatten_and_concatenate_obs(obs_dict):
  return np.concatenate(
      [obs.astype(np.float32).flatten() for obs in obs_dict.values()])




class UniformBoundActionSpaceWrapper(gym.Wrapper):
  """Rescale actions so that action space bounds are [-1, 1]."""

  def __init__(self, env):
    """Initialize the wrapper.
    Args:
      env: Environment to be wrapped. It must have an action space of type
        gym.spaces.Box.
    """
    super().__init__(env)
    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.action_space.dtype == np.float32
    n_action_dim = env.action_space.shape[0]
    self.half_range = (env.action_space.high - env.action_space.low) / 2.
    self.center = env.action_space.low + self.half_range
    self.action_space = gym.spaces.Box(low=-np.ones(n_action_dim),
                                       high=np.ones(n_action_dim),
                                       dtype=np.float32)

  def step(self, action):
    assert np.abs(action).max() < 1.00001, 'Action: %s' % action
    action = np.clip(action, -1, 1)
    action = self.center + action * self.half_range
    return self.env.step(action)


class DiscretizeEnvWrapper(gym.Env):
  """Wrapper for discretizing actions."""

  def __init__(self, env, n_actions_per_dim, discretization='lin',
               action_ratio=None):
    """"Discretize actions.
    Args:
      env: Environment to be wrapped.
      n_actions_per_dim: The number of buckets per action dimension.
      discretization: Discretization mode, can be 'lin' or 'log',
        'lin' spaces buckets linearly between low and high while 'log'
        spaces them logarithmically.
      action_ratio: The ratio of the highest and lowest positive action
        for logarithim discretization.
    """

    self.env = env
    assert len(env.action_space.shape) == 1
    dim_action = env.action_space.shape[0]
    self.action_space = gym.spaces.MultiDiscrete([n_actions_per_dim] *
                                                 dim_action)
    self.observation_space = env.observation_space
    high = env.action_space.high
    if isinstance(high, float):
      assert env.action_space.low == -high
    else:
      high = high[0]
      assert (env.action_space.high == [high] * dim_action).all()
      assert (env.action_space.low == -env.action_space.high).all()
    if discretization == 'log':
      assert n_actions_per_dim % 2 == 1, (
          'The number of actions per dimension '
          'has to be odd for logarithmic discretization.')
      assert action_ratio is not None
      log_range = np.linspace(np.log(high / action_ratio),
                              np.log(high),
                              n_actions_per_dim // 2)
      self.action_set = np.concatenate([-np.exp(np.flip(log_range)),
                                        [0.],
                                        np.exp(log_range)])
    elif discretization == 'lin':
      self.action_set = np.linspace(-high, high, n_actions_per_dim)

  def step(self, action):
    assert self.action_space.contains(action)
    action = np.take(self.action_set, action)
    assert self.env.action_space.contains(action)
    obs, rew, done, info = self.env.step(action)
    return obs, rew, done, info

  def reset(self):
    return self.env.reset()

  def render(self, *args, **kwargs):
    return self.env.render(*args, **kwargs)


class BatchedEnvironment:
  """A wrapper that batches several environment instances."""

  def __init__(self, create_env_fn, batch_size, id_offset):
    """Initialize the wrapper.
    Args:
      create_env_fn: A function to create environment instances.
      batch_size: The number of environment instances to create.
      id_offset: The offset for environment ids. Environments receive sequential
        ids starting from this offset.
    """
    self._batch_size = batch_size
    # Note: some environments require an argument to be of a native Python
    # numeric type. If we create env_ids as a numpy array, its elements will
    # be of type np.int32. So we create it as a plain Python array first.
    env_ids = [id_offset + i for i in range(batch_size)]
    self._envs = [create_env_fn(id) for id in env_ids]
    self._env_ids = np.array(env_ids, np.int32)
    self._obs = None

  @property
  def env_ids(self):
    return self._env_ids

  @property
  def envs(self):
    return self._envs

  @property
  def _mapped_obs(self):
    """Maps observations to preserve the original structure.
    This is needed to support environments that return structured observations.
    For example, gym.GoalEnv has `observation`, `desired_goal`, and
    `achieved_goal` elements in its observations. In this case the batched
    observations would contain the same three elements batched by element.
    Returns:
      Mapped observations.
    """
    return tf.nest.map_structure(lambda *args: np.array(args), *self._obs)

  def step(self, action_batch):
    """Does one step for all batched environments sequentially."""
    num_envs = self._batch_size
    rewards = np.zeros(num_envs, np.float32)
    dones = np.zeros(num_envs, np.bool)
    infos = [None] * num_envs
    for i in range(num_envs):
      self._obs[i], rewards[i], dones[i], infos[i] = self._envs[i].step(
          action_batch[i])
    return self._mapped_obs, rewards, dones, infos

  def reset(self):
    """Reset all environments."""
    observations = [env.reset() for env in self._envs]
    self._obs = observations
    return self._mapped_obs

  def reset_if_done(self, done):
    """Reset the environments for which 'done' is True.
    Args:
      done: An array that specifies which environments are 'done', meaning their
        episode is terminated.
    Returns:
      Observations for all environments.
    """
    assert self._obs is not None, 'reset_if_done() called before reset()'
    for i in range(len(self._envs)):
      if done[i]:
        self._obs[i] = self.envs[i].reset()

    return self._mapped_obs

  def render(self, mode='human', **kwargs):
    # Render only the first one
    self._envs[0].render(mode, **kwargs)

  def close(self):
    for env in self._envs:
      env.close()


class FloatWrapper(gym.Env):
  def __init__(self, env):
    self.env = env
    self.action_space = env.action_space
    self.observation_space = env.observation_space

  def reset(self):
    obs = self.env.reset()
    return obs.astype(np.float32)

  def step(self, action):
    obs, rew, done, info = self.env.step(action)
    return obs.astype(np.float32), rew, done, info

  def render(self, mode='human'):
    return self.env.render()


class MultiWrapper(gym.Env):
  def __init__(self, env, normalization=None):
    self.env = env
    self.num_agents = len(env.action_space)
    if normalization is None:
      self.normalization = (lambda x: x)
    else:
      self.normalization = normalization

    self.action_space = gym.spaces.MultiDiscrete([space.n for space in env.action_space])

    self.observation_space = env.observation_space[0]
    self.observation_space.shape = (len(env.observation_space),) + env.observation_space[0].shape
    self.observation_space.dtype = np.float32

    print(env.action_space, env.observation_space)
    print(self.action_space, self.observation_space)

  def reset(self):
    obs = self.env.reset()
    return self._convert_observation(obs)

  def step(self, action):
    action = self._convert_action(action)
    obs, rew, done, info = self.env.step(action)
    return self._convert_observation(obs), self._convert_reward(rew), done, info

  def render(self, mode='human'):
    return self.env.render(mode)

  def _convert_observation(self, obs):
    return np.stack([self.normalization(obs[i]) for i in range(self.num_agents)], axis=0).astype(np.float32)

  def _convert_action(self, action):
    return action

  def _convert_reward(self, reward):
    return np.sum(reward)


class ParticleWrapper(gym.Env):
  def __init__(self, env, normalization=None):
    self.env = env
    self.num_agents = len(env.action_space)
    if normalization is None:
      self.normalization = (lambda x: x)
    else:
      self.normalization = normalization

    self.n_step = 0
    self.step_limit = 25

    # The discrete action space in fact mean N binary actions
    self.action_dim = env.action_space[0].n
    self.action_dims = [space.n for space in env.action_space]
    self.action_space = gym.spaces.MultiDiscrete([2 for _ in range(sum(self.action_dims))])

    self.observation_space = env.observation_space[0]
    self.observation_space.shape = (len(env.observation_space),) + env.observation_space[0].shape
    self.observation_space.dtype = np.float32

    print(env.action_space, env.observation_space)
    print(self.action_space, self.observation_space)

  def reset(self):
    self.n_step = 0
    obs = self.env.reset()
    return self._convert_observation(obs)

  def step(self, action):
    self.n_step += 1
    action = self._convert_action(action)
    obs, rew, done, info = self.env.step(action)
    done = np.all(done) or self.n_step >= self.step_limit
    return self._convert_observation(obs), self._convert_reward(rew), done, info

  def render(self, mode='human'):
    return self.env.render(mode=mode)

  def _convert_observation(self, obs):
    return np.stack([self.normalization(obs[i]) for i in range(self.num_agents)], axis=0).astype(np.float32)

  def _convert_action(self, action):
    # print(action)
    act = [action[self.action_dim * i:self.action_dim * (i + 1)] for i in range(self.num_agents)]
    # print(act)
    return act
    # return action
    # return [np.eye(5)[a] for a in action]

  def _convert_reward(self, reward):
    return reward[0] / 100


class SCWrapper(gym.Env):
  def __init__(self, env, normalization=None):
    self.env = env
    self.stacked_obs = []

    if normalization is None:
      self.normalization = (lambda x: x)
    else:
      self.normalization = normalization

    info = self.env.get_env_info()
    self.num_agents = info['n_agents']

    self.action_space = gym.spaces.MultiDiscrete(
                            [info['n_actions'] for _ in range(self.num_agents)])

    self.observation_space = gym.spaces.Box(
      low=-np.inf,
      high=np.inf,
      shape=(info['n_agents'], info['obs_shape']*FLAGS.frames_stacked + info['n_actions'] + info['state_shape']))
    self.observation_space.dtype = np.float32

    self.state_dim = info['state_shape']

    print(self.action_space, self.observation_space, 'state size:', self.state_dim)

  def reset(self):
    self.stacked_obs = []
    self.env.reset()
    obs = self.env.get_obs()
    return self._convert_observation(obs)

  def step(self, action):
    action = self._convert_action(action)
    rew, done, info = self.env.step(action)
    obs = self.env.get_obs()
    return self._convert_observation(obs), self._convert_reward(rew), done, info

  def render(self, mode='human'):
    return self.env.render(mode)

  def _add_new_obs(self, obs):
    if self.stacked_obs == []:
      self.stacked_obs = [np.zeros_like(obs)] * (FLAGS.frames_stacked - 1) + [obs]
    else:
      self.stacked_obs = self.stacked_obs[1:] + [obs]

  def _convert_observation(self, obs):
    # Prepare raw observation for agent network.
    # Add information about available actions.
    self._add_new_obs(obs)

    return np.stack([
      np.concatenate([
        self.normalization(self.stacked_obs[stack_i][agent_i])
                       for stack_i in range(FLAGS.frames_stacked)] + [
        # np.eye(self.num_agents)[agent_i],
        self.env.get_avail_agent_actions(agent_i),
        self.env.get_state(),
      ]) for agent_i in range(self.num_agents)
    ], axis=0).astype(np.float32)

  def _convert_action(self, action):
    # Ensure the executed action is available.
    # Currently all the actions are chosen from availables only and thus no
    # invalidity should be detected here.
    true_actions = []

    for i in range(self.num_agents):
      a = action[i]
      avail_actions = self.env.get_avail_agent_actions(i)
      if avail_actions[a] == 0:
        # Should never be true.
        print('Invalid action. This is undesired that you see this message.',
              'Likely a bug in action choice logic.')
        avail_actions_ind = np.nonzero(avail_actions)[0]
        a = np.random.choice(avail_actions_ind)
      true_actions.append(a)

    return true_actions

  def save_replay(self):
    return self.env.save_replay()

  def _convert_reward(self, reward):
    return np.sum(reward)
