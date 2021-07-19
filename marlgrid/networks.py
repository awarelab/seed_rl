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
from seed_rl.marlgrid import observation
import tensorflow as tf

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
  def __init__(self):
    super().__init__()
    self.conv0 = tf.keras.layers.Conv2D(16, 2, activation='relu')
    self.conv1 = tf.keras.layers.Conv2D(32, 2, activation='relu')
    self.flatten = tf.keras.layers.Flatten()
    self.layer0 = tf.keras.layers.Dense(128, activation='relu')
    self.layer1 = tf.keras.layers.Dense(64, activation=None)

  def eval(self, input):
    x = self.conv0(input)
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.layer0(x)
    x = self.layer1(x)
    return x


class GFootball(tf.Module):
  """Agent with ResNet, but without LSTM and additional inputs.

  Four blocks instead of three in ImpalaAtariDeep.
  """

  def __init__(self, parametric_action_distribution):
    super(GFootball, self).__init__(name='gfootball')

    # Parameters and layers for unroll.
    self._parametric_action_distribution = parametric_action_distribution
    self.num_agents = parametric_action_distribution._n_dimensions

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

  def _torso(self, unused_prev_action, env_output):
    _, _, frame, _, _ = env_output
    # print('frame', frame.shape)
    # print('single frame', frame[:, 0])

    baseline_output = self.critic.eval(tf.concat([frame[:, i] for i in range(self.num_agents)], axis=-1))
    policy_outputs = [self.actor.eval(frame[:, i]) for i in range(self.num_agents)]

    return tf.stack(policy_outputs + [baseline_output], axis=1)

  def _head(self, core_output):
    # print('core output', core_output.shape, core_output)
    policy_logits = tf.concat([self._policy_logits(core_output[:, i]) for i in range(self.num_agents)], axis=-1)
    # print('policy_logits', policy_logits)

    baseline_input = self.final_flatten(core_output[:, self.num_agents])
    baseline = tf.squeeze(self._baseline(baseline_input), axis=-1)
    # print('baseline', baseline)

    # Sample an action from the policy.
    new_action = self._parametric_action_distribution.sample(policy_logits)
    # print('new_action', new_action)

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
