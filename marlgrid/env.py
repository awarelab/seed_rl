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

"""Football env factory."""
import sys

from absl import flags
from absl import logging

import gym
from marlgrid.agents import GridAgentInterface
from seed_rl.common import common_flags
from seed_rl.common import env_wrappers
from seed_rl.marlgrid import observation
import marlgrid
from marlgrid import envs
from gym.envs.registration import register as gym_register

from marlgrid.envs import CollectorMultiGrid

this_module = sys.modules[__name__]


def register_marl_env(
        env_name,
        env_class,
        n_agents,
        grid_size,
        view_size,
        view_tile_size=1,
        view_offset=0,
        agent_color=None,
        env_kwargs={},
):
  colors = ["red", "blue", "purple", "orange", "olive", "pink"]
  assert n_agents <= len(colors)
  print('n_agents', n_agents)

  class RegEnv(env_class):
    def __new__(cls):
      instance = super(env_class, RegEnv).__new__(env_class)
      instance.__init__(
        agents=[
          GridAgentInterface(
            color=c if agent_color is None else agent_color,
            view_size=view_size,
            view_tile_size=view_tile_size,
            view_offset=view_offset,
          )
          for c in colors[:n_agents]
        ],
        grid_size=grid_size,
        **env_kwargs,
      )
      return instance

  env_class_name = f"env_0"
  setattr(marlgrid.envs, env_class_name, RegEnv)
  # registered_envs.append(env_name)
  gym_register(env_name, entry_point=f"marlgrid.envs:{env_class_name}")




def create_environment(_):
  """Returns a gym Football environment."""
  task = 'MarlGrid-3FixedCollectorMultiGrid15x15-v1'

  register_marl_env(
    task,
    CollectorMultiGrid,
    n_agents=3,
    grid_size=15,
    view_size=7,
    view_tile_size=1,
    env_kwargs={'clutter_density': 0.15, 'n_random_bonuses': 6, 'fixed_players': True}
  )

  # task = 'LunarLander-v2'
  logging.info('Creating environment: %s', task)
  env = gym.make(task)
  normalization = lambda x: x / 255.
  return env_wrappers.MultiWrapper(env, normalization=normalization)
