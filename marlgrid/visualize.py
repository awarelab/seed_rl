import PIL
import math
import os
import sys
import time

from absl import app
from absl import flags

from seed_rl.agents.vtrace import learner
from seed_rl.common import actor
from seed_rl.common import common_flags
from seed_rl.marlgrid import env
from seed_rl.marlgrid import networks
import tensorflow as tf
import numpy as np


from seed_rl.common.parametric_distribution import get_parametric_distribution_for_action_space
from seed_rl.common import utils


FLAGS = flags.FLAGS


def visualize(create_env_fn, create_agent_fn, create_optimizer_fn):
  print('Visualization launched...')

  settings = utils.init_learner_multi_host(1)
  strategy, hosts, training_strategy, encode, decode = settings

  env = create_env_fn(0)
  parametric_action_distribution = get_parametric_distribution_for_action_space(
    env.action_space)
  agent = create_agent_fn(env.action_space, env.observation_space,
                          parametric_action_distribution)
  optimizer, learning_rate_fn = create_optimizer_fn(1e9)

  env_output_specs = utils.EnvOutput(
    tf.TensorSpec([], tf.float32, 'reward'),
    tf.TensorSpec([], tf.bool, 'done'),
    tf.TensorSpec(env.observation_space.shape, env.observation_space.dtype,
                  'observation'),
    tf.TensorSpec([], tf.bool, 'abandoned'),
    tf.TensorSpec([], tf.int32, 'episode_step'),
  )
  action_specs = tf.TensorSpec(env.action_space.shape,
                               env.action_space.dtype, 'action')
  agent_input_specs = (action_specs, env_output_specs)
  # Initialize agent and variables.
  agent = create_agent_fn(env.action_space, env.observation_space,
                          parametric_action_distribution)
  initial_agent_state = agent.initial_state(1)
  agent_state_specs = tf.nest.map_structure(
    lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_state)
  unroll_specs = [None]  # Lazy initialization.
  input_ = tf.nest.map_structure(
    lambda s: tf.zeros([1] + list(s.shape), s.dtype), agent_input_specs)
  input_ = encode(input_)

  with strategy.scope():
    @tf.function
    def create_variables(*args):
      return agent.get_action(*decode(args))

    initial_agent_output, _ = create_variables(*input_, initial_agent_state)

    if not hasattr(agent, 'entropy_cost'):
      mul = FLAGS.entropy_cost_adjustment_speed
      agent.entropy_cost_param = tf.Variable(
          tf.math.log(FLAGS.entropy_cost) / mul,
          # Without the constraint, the param gradient may get rounded to 0
          # for very small values.
          constraint=lambda v: tf.clip_by_value(v, -20 / mul, 20 / mul),
          trainable=True,
          dtype=tf.float32)
      agent.entropy_cost = lambda: tf.exp(mul * agent.entropy_cost_param)
    # Create optimizer.
    iter_frame_ratio = (
        FLAGS.batch_size * FLAGS.unroll_length * FLAGS.num_action_repeats)
    final_iteration = int(
        math.ceil(FLAGS.total_environment_frames / iter_frame_ratio))
    optimizer, learning_rate_fn = create_optimizer_fn(final_iteration)


    iterations = optimizer.iterations
    optimizer._create_hypers()
    optimizer._create_slots(agent.trainable_variables)

    # ON_READ causes the replicated variable to act as independent variables for
    # each replica.
    temp_grads = [
        tf.Variable(tf.zeros_like(v), trainable=False,
                    synchronization=tf.VariableSynchronization.ON_READ)
        for v in agent.trainable_variables
    ]

  agent_output_specs = tf.nest.map_structure(
    lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_output)

  if True:
    ckpt = tf.train.Checkpoint(agent=agent, optimizer=optimizer)
    ckpt.restore('seed_rl/checkpoints/marlgrid_vis/ckpt-3').assert_consumed()

  def get_agent_action(obs):
    initial_agent_state = agent.initial_state(1)
    shaped_obs = tf.reshape(tf.convert_to_tensor(obs), shape=(1,)+env.observation_space.shape)
    initial_env_output = (tf.constant([1.]), tf.constant([False]), shaped_obs,
                          tf.constant([False]), tf.constant([1], dtype=tf.float32),)
    # __call__(self, prev_actions, env_outputs, core_state, unroll=False, is_training=False, postprocess_action=True)
    agent_out = agent(tf.zeros([0], dtype=tf.float32), initial_env_output,
                      initial_agent_state)
    return agent_out

  def run_episode(steps):
    mode = 'human'
    obs = env.reset()
    rewards = []
    env.render(mode=mode)

    for _ in range(steps):
      agent_out, state = get_agent_action(obs)
      action = agent_out.action.numpy()[0]
      obs, rew, done, info = env.step(action)
      rewards.append(rew)
      time.sleep(0.01)
      env.render(mode=mode)
      if done:
        break

    reward = np.sum(rewards)
    print('reward: {0}'.format(reward))
    return reward

  all_rewards = []
  while True:
    all_rewards.append(run_episode(1000))
    if len(all_rewards) > 1000:
      all_rewards = all_rewards[-1000:]
    print('mean cum reward: {0}'.format(np.mean(all_rewards)))

  print('Graceful termination')
  sys.exit(0)