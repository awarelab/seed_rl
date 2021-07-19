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


"""V-trace (IMPALA) learner for Google Research Football."""
import neptune_tensorboard
from absl import app
from absl import flags

from seed_rl.agents.vtrace import learner
from seed_rl.common import actor
from seed_rl.common import common_flags
from seed_rl.common import utils
from seed_rl.starcraft import env
from seed_rl.starcraft import networks
from seed_rl.starcraft import visualize
import tensorflow as tf
import neptune

# from mrunner.helpers.client_helper import get_configuration
from seed_rl.common.utils import get_configuration


FLAGS = flags.FLAGS

# Optimizer settings.
flags.DEFINE_string('checkpoint', None, 'Checkpoint to visualize')
flags.DEFINE_string('mrunner_config', None, 'Mrunner config file.')
flags.DEFINE_float('learning_rate', 0.001, 'Maximal learning rate.')
flags.DEFINE_integer('learning_rate_warmup', 100, 'Number of warmup iterations.')
flags.DEFINE_float('learning_rate_decay_scale', -1, 'Learning rate decay scale.'
                                              'Set negative to disable.')


def create_agent(unused_action_space, unused_env_observation_space,
                 parametric_action_distribution):
  environment = env.create_environment(0)
  return networks.StarcraftAgentNetwork(parametric_action_distribution,
                                        {'state_dim': environment.state_dim})


def learning_rate_warmup_decay(iteration):
    if iteration <= FLAGS.learning_rate_warmup:
        return FLAGS.learning_rate * (iteration / FLAGS.learning_rate_warmup)
    if FLAGS.learning_rate_decay_scale < 0:
        return FLAGS.learning_rate
    return FLAGS.learning_rate / ((1 + (tf.cast(iteration, tf.float32) - FLAGS.learning_rate_warmup) * FLAGS.learning_rate_decay_scale) ** 0.5)


def create_optimizer(unused_final_iteration):
  learning_rate_fn = learning_rate_warmup_decay
  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
  return optimizer, learning_rate_fn


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if FLAGS.run_mode == 'actor':
    if not FLAGS.is_local:
      get_configuration(config_file=FLAGS.mrunner_config,
                        inject_parameters_to_FLAGS=True)
    actor.actor_loop(env.create_environment)
  elif FLAGS.run_mode == 'learner':
    if not FLAGS.is_local:
      get_configuration(config_file=FLAGS.mrunner_config,
                        print_diagnostics=True, with_neptune=True,
                        inject_parameters_to_FLAGS=True)
      experiment = neptune.get_experiment()
      experiment.append_tag(tag=FLAGS.nonce)
      # experiment.append_tag(tag='paper_fullstate_stacking{0}'.format(FLAGS.frames_stacked))
      neptune_tensorboard.integrate_with_tensorflow()
    learner.learner_loop(env.create_environment,
                         create_agent,
                         create_optimizer)
  elif FLAGS.run_mode == 'visualize':
    visualize.visualize(env.create_environment, create_agent, create_optimizer)
  else:
    raise ValueError('Unsupported run mode {}'.format(FLAGS.run_mode))


if __name__ == '__main__':
  app.run(main)
