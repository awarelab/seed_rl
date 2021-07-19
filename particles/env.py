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
from seed_rl.particles import observation

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios


def make_particle_env(scenario_name, benchmark=False):
  scenario = scenarios.load(scenario_name + ".py").Scenario()
  world = scenario.make_world()
  if benchmark:
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
  else:
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
  return env


def create_environment(_):
  """Returns a particle environment."""
  task = 'simple_spread'

  logging.info('Creating environment: %s', task)
  env = make_particle_env(task)
  return env_wrappers.ParticleWrapper(env)
