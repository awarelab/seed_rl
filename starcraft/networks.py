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

"""SEED agent using Keras."""

import collections
from seed_rl.common import utils
from seed_rl.starcraft import observation
from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_bool('full_state_critic', False,
                  'Whether the critic network uses hidden state '
                  'of the environment.')

AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')


class _Stack(tf.Module):
  """Stack of pooling and convolutional blocks with residual connections."""

  def __init__(self, num_ch, num_blocks):

    super(_Stack, self).__init__(name='stack')
    self._conv = tf.keras.layers.Conv2D(num_ch, 3, strides=1, padding='same',
                                        kernel_initializer='lecun_normal')
    self._max_pool = tf.keras.layers.MaxPool2D(
        pool_size=3, padding='same', strides=2)

    self._res_convs0 = [
        tf.keras.layers.Conv2D(
            num_ch, 3, strides=1, padding='same', name='res_%d/conv2d_0' % i,
            kernel_initializer='lecun_normal')
        for i in range(num_blocks)
    ]
    self._res_convs1 = [
        tf.keras.layers.Conv2D(
            num_ch, 3, strides=1, padding='same', name='res_%d/conv2d_1' % i,
            kernel_initializer='lecun_normal')
        for i in range(num_blocks)
    ]

  def __call__(self, conv_out):
    # Downscale.
    conv_out = self._conv(conv_out)
    conv_out = self._max_pool(conv_out)

    # Residual block(s).
    for (res_conv0, res_conv1) in zip(self._res_convs0, self._res_convs1):
      block_input = conv_out
      conv_out = tf.nn.relu(conv_out)
      conv_out = res_conv0(conv_out)
      conv_out = tf.nn.relu(conv_out)
      conv_out = res_conv1(conv_out)
      conv_out += block_input

    return conv_out


class SimpleNetwork(tf.Module):
  def __init__(self, use_norm=False):
    super().__init__()
    self.use_norm = use_norm

    self.layer0 = tf.keras.layers.Dense(64, activation=None)
    if self.use_norm:
      self.norm0 = tf.keras.layers.LayerNormalization()
    self.act0 = tf.keras.layers.ReLU()
    self.layer1 = tf.keras.layers.Dense(64, activation=None)
    if self.use_norm:
      self.norm1 = tf.keras.layers.LayerNormalization()
    self.act1 = tf.keras.layers.ReLU()

  def eval(self, input):
    x = input
    x = self.layer0(x)
    if self.use_norm:
      x = self.norm0(x)
    x = self.act0(x)
    x = self.layer1(x)
    if self.use_norm:
      x = self.norm1(x)
    x = self.act1(x)
    return x


class StarcraftAgentNetwork(tf.Module):
  def __init__(self, parametric_action_distribution, info):
    super(StarcraftAgentNetwork, self).__init__(name='starcraft_agent')

    # Parameters and layers for unroll.
    self._parametric_action_distribution = parametric_action_distribution
    self.num_agents = parametric_action_distribution._n_dimensions
    self.num_actions = parametric_action_distribution._n_actions_per_dim

    self.state_dim = info['state_dim']

    self.avail_actions = None

    self.actor = SimpleNetwork()
    self.critic = SimpleNetwork()

    self.final_flatten = tf.keras.layers.Flatten()

    # Layers for _head.
    self._policy_logits = tf.keras.layers.Dense(
        self._parametric_action_distribution._n_actions_per_dim,
        name='policy_logits',
        kernel_initializer='lecun_normal')
    self._baseline = tf.keras.layers.Dense(
        1, name='baseline', kernel_initializer='lecun_normal')

  def initial_state(self, batch_size):
    return ()

  def _torso(self, prev_action, env_output):
    _, _, frame_and_actions_and_state, _, _ = env_output

    # Divide the environment output onto observations and available actions.
    # frame, avail_actions, state = frame_and_actions_and_state
    state = frame_and_actions_and_state[:, 0, -self.state_dim:]
    frame_and_actions = frame_and_actions_and_state[:, :, :-self.state_dim]

    frame = tf.stack([
      frame_and_actions[:, i, :-self.num_actions]
      for i in range(self.num_agents)
    ], axis=1)

    self.avail_actions = tf.concat([
      frame_and_actions[:, i, -self.num_actions:]
      for i in range(self.num_agents)
    ], axis=1)

    policy_outputs = [
      self.actor.eval(frame[:, i]) for i in range(self.num_agents)]

    if FLAGS.is_centralized:
      if FLAGS.full_state_critic:
        # state-based critic
        baseline_outputs = [self.critic.eval(state)]
      else:
        # observation-based critic
        baseline_outputs = [
          self.critic.eval(
            tf.concat([frame[:, i] for i in range(self.num_agents)], axis=-1))]
    elif FLAGS.is_action_aware:
      prev_action_one_hot = tf.one_hot(prev_action, self.num_actions)
      frame_action = tf.concat([frame, prev_action_one_hot], axis=-1)
      mask = tf.concat([tf.ones_like(frame[:, 0, :]),
                        tf.zeros_like(prev_action_one_hot[:, 0, :])], axis=-1)

      baseline_outputs = []
      for agent in range(self.num_agents):
        agent_obs_stacked \
          = [frame_action[:, (agent + i) % self.num_agents, :] for i in
             range(self.num_agents)]
        # remove action of the ego agent
        agent_obs_stacked[0] = agent_obs_stacked[0] * mask
        agent_obs = tf.concat(agent_obs_stacked, axis=-1)
        baseline_outputs.append(self.critic.eval(agent_obs))

    else:
      baseline_outputs = [
        self.critic.eval(frame[:, i]) for i in range(self.num_agents)]

    return tf.stack(policy_outputs + baseline_outputs, axis=1)

  def _head(self, core_output):
    policy_logits = tf.concat([
      self._policy_logits(core_output[:, i]) for i in range(self.num_agents)
    ], axis=-1)
    # Policy can choose only available actions.
    # Set logits for all non-available actions to -100.
    # In order to prevent other logits going too low, clip them to -50.
    policy_logits = tf.maximum(policy_logits, -50) * self.avail_actions - \
                    (1 - self.avail_actions) * 100

    # Output of baseline torso is the last row.
    if FLAGS.is_centralized:
      baseline_input = self.final_flatten(core_output[:, self.num_agents])
      baseline = tf.squeeze(self._baseline(baseline_input), axis=-1)
    else:
      baselines = [self._baseline(self.final_flatten(core_output[:, i])) for i in
                   range(self.num_agents, 2 * self.num_agents)]
      baseline = tf.concat(baselines, axis=-1)

    new_action = self._parametric_action_distribution.sample(policy_logits)

    return AgentOutput(new_action, policy_logits, baseline)

  # Not clear why, but if "@tf.function" declarator is placed directly onto
  # __call__, training fails with "uninitialized variable *baseline".
  # when running on multiple learning tpu cores.


  @tf.function
  def get_action(self, *args, **kwargs):
    return self.__call__(*args, **kwargs)

  def __call__(self, prev_actions, env_outputs, core_state, unroll=False,
               is_training=False, postprocess_action=True):
    if not unroll:
      # Add time dimension.
      prev_actions, env_outputs = tf.nest.map_structure(
          lambda t: tf.expand_dims(t, 0), (prev_actions, env_outputs))

    outputs, core_state = self._unroll(prev_actions, env_outputs, core_state)

    if not unroll:
      # Remove time dimension.
      outputs = tf.nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)

    if postprocess_action:
      outputs = outputs._replace(
          action=self._parametric_action_distribution.postprocess(
              outputs.action))

    return outputs, core_state

  def _unroll(self, prev_actions, env_outputs, core_state):
    torso_outputs = utils.batch_apply(self._torso, (prev_actions, env_outputs))
    return utils.batch_apply(self._head, (torso_outputs,)), core_state
