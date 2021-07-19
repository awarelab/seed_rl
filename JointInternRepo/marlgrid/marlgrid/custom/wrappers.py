import gym
import numpy as np

def convert_observation_for_one_head(old_observation):
  return np.stack(old_observation, axis=0).astype(np.uint8)


class ObservationForOneHead(gym.ObservationWrapper):
  def __init__(self, env):
    gym.ObservationWrapper.__init__(self, env)

    self._num_players = len(self.env.observation_space)
    assert len(set(map(lambda x: x.shape, self.env.observation_space))) == 1

    self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._num_players, ) + self.env.observation_space[0].shape,
            dtype="uint8",
        )
  def observation(self, old_obs):
    return convert_observation_for_one_head(old_obs)


def convert_observation_for_multi_head(old_observation):
  return np.expand_dims(np.concatenate(old_observation, axis=-1), axis=0).astype(np.uint8)

class ObservationForMultiHead(gym.ObservationWrapper):
  def __init__(self, env):
    gym.ObservationWrapper.__init__(self, env)

    self._num_players = len(self.env.observation_space)
    assert len(set(map(lambda x: x.shape, self.env.observation_space))) == 1
    obs_shape = (1,) + self.env.observation_space[0].shape[:-1] + (self.env.observation_space[0].shape[-1] * self._num_players,)


    self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype="uint8",
        )
  def observation(self, old_obs):
    return convert_observation_for_multi_head(old_obs)

class RewardForMultiHead(gym.RewardWrapper):
  def __init__(self, env):
    gym.RewardWrapper.__init__(self, env)

  def reward(self, old_reward):
    return [np.sum(old_reward)]

class TeamRewardForOneHead(gym.RewardWrapper):
  def __init__(self, env):
    gym.RewardWrapper.__init__(self, env)

  def reward(self, old_reward):
    return [np.sum(old_reward)] * len(old_reward)
   

def convert_action_space(old_action_space):
  nvec = []
  for disc_aprop in old_action_space:
    assert isinstance(disc_aprop, gym.spaces.Discrete)
    nvec.append(disc_aprop.n)

  assert len(set(nvec)) == 1
  
  return gym.spaces.MultiDiscrete(nvec)

class ListOfDiscreteToMultiDiscrete(gym.Wrapper):
  def __init__(self, env):
    gym.Wrapper.__init__(self, env)
    self.action_space = convert_action_space(env.action_space)



class NoRenderWrapper(gym.Wrapper):
  def __init__(self, env):
    gym.Wrapper.__init__(self, env)
  
  def render(self):
    pass


from datetime import datetime
from marlgrid.utils.video import GridRecorder
import time
import os
import tensorflow as tf
import threading

def make_dirs(logdir):
    tf.io.gfile.makedirs(logdir)


def upload_logs(local_logdir, remote_logdir):
    while True:
        local_files = tf.io.gfile.listdir(local_logdir)
        remote_files = tf.io.gfile.listdir(remote_logdir)
        diff = list(set(local_files) - set(remote_files))
        for f in diff:
            tf.io.gfile.copy(os.path.join(local_logdir, f),
                             os.path.join(remote_logdir, f))
        time.sleep(1)


def fix_remote_logdir(logdir):
    """gs://... directory does not work
    this function fixes that by creating local logdir
    and syncing it with remote one"""

    if logdir is not None and logdir.startswith('gs://'):
        pruned_logdir = logdir.replace('/', '_').replace(':', '')
        local_logdir = '/tmp/env_log/' + pruned_logdir
        remote_logdir = logdir
        make_dirs(local_logdir)
        make_dirs(remote_logdir)
        t = threading.Thread(target=upload_logs,
                             args=(local_logdir, remote_logdir))
        t.start()
        return local_logdir
    else:
        return logdir


class DumpMovies(gym.Wrapper):
  def __init__(self, env, log_dir, dump_always=False):
    log_dir = fix_remote_logdir(log_dir)
    env = GridRecorder(env, render_kwargs={"tile_size": 32 if dump_always else 11}, save_root=log_dir)
    env.recording = False
    gym.Wrapper.__init__(self, env)
    self._counter = 0
    self._dump_always = dump_always
  
  def reset(self):
    if self._dump_always:
      self.env.recording = True
      if self._counter > 0:
        now = datetime.now()
        str_time = now.strftime("%m_%d_%Y_%H_%M_%S_" + str(self._counter))
        self.env.export_video(str_time + '.mp4')
    else:
      if self._counter % 100 == 3:
        self.env.recording = True
      elif self._counter % 100 == 4:
        now = datetime.now()
        str_time = now.strftime("%m_%d_%Y_%H_%M_%S_" + str(self._counter))
        self.env.export_video(str_time + '.mp4')
        self.env.recording = False
    self._counter = self._counter + 1
    
    return self.env.reset()

  def close(self):
    self.reset()
    self.env.close()

import math

def euclidean_distance(x1, y1, x2, y2):

  res = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

  #print('points:', x1, y1, x2, y2, 'fist:', res)

  return res

import tensorflow as tf

class LogPlayerEntropy(gym.Wrapper):
  def __init__(self, env, logdir):
    gym.Wrapper.__init__(self, env)
    self.logdir = logdir
    self.summary_writer = tf.summary.create_file_writer(self.logdir)
    self.summary_step = 0

    self.sum_dist_metrics = []
    self.sum_of_closest_to_edges_metric = []


  def _calculate_sum_dist(self, positions):
    sum = 0.0
    for x1, y1 in positions:
      for x2, y2 in positions:
        sum += euclidean_distance(x1, y1, x2, y2)

    return sum

  def _calculate_sum_of_closest_to_core_points(self, positions, core_points):
    sum = 0.0
    for x1, y1 in core_points:
      best = None
      for x2, y2 in positions:
        dist = euclidean_distance(x1, y1, x2, y2)
        if best is None or best > dist:
          best = dist
      sum += best
    
    return sum


  def step(self, action):
    agents = self.env.unwrapped.agents
    positions = list(map(lambda a: a.pos, agents))

    sum_dist = self._calculate_sum_dist(positions)
    self.sum_dist_metrics.append(sum_dist)
    #print('after_dist')

    edges = [
      (0, 0),
      (self.env.unwrapped.width - 1, 0),
      (self.env.unwrapped.width - 1, self.env.unwrapped.height - 1),
      (0, self.env.unwrapped.height - 1)
    ]
    sum_of_closest = self._calculate_sum_of_closest_to_core_points(positions, edges)
    self.sum_of_closest_to_edges_metric.append(sum_of_closest)

    #print(sum_dist, sum_of_closest)
   
    return self.env.step(action)

  def reset(self):
    if len(self.sum_dist_metrics) != 0:
      sum_dist_mean = np.mean(self.sum_dist_metrics)
      sum_of_closest_mean = np.mean(self.sum_of_closest_to_edges_metric)

      with self.summary_writer.as_default():
        tf.summary.scalar("metrics/sum_of_dist_mean", sum_dist_mean, step=self.summary_step)
        tf.summary.scalar("metrics/sum_of_closest_to_edges", sum_of_closest_mean, step=self.summary_step)
      self.summary_step += 1

    self.sum_dist_metrics = []
    self.sum_of_closest_to_edges_metric = []
    return self.env.reset()

def get_tile_range(x, y, tile_size):
  ymin = y * tile_size
  ymax = (y + 1) * tile_size
  xmin = x * tile_size
  xmax = (x + 1) * tile_size

  return xmin, xmax, ymin, ymax

def rgb2gray(rgb):

    rgb = rgb.astype(np.float32)

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray.astype(np.uint8)

from marlgrid.base import MultiGrid

class FullGridImgObservation(gym.ObservationWrapper):
  def __init__(self, env, tile_size=8):
    gym.ObservationWrapper.__init__(self, env)
    self._num_agents = len(self.env.unwrapped.agents)
    assert self._num_agents == len(self.env.observation_space)

    self._width = self.env.unwrapped.width
    self._height = self.env.unwrapped.height
    self._tile_size = tile_size

    one_obs_shape = (self._height * self._tile_size, self._width * self._tile_size, 4)
    self.observation_space = [gym.spaces.Box(
      low=0,
      high=255,
      shape=one_obs_shape,
      dtype="uint8"
    )] * self._num_agents

  def observation(self, org_obs):
    assert len(org_obs) == self._num_agents

    grid = self.env.unwrapped.grid.render(self._tile_size).astype(np.uint8)
    
    new_obs_list = []
    for i in range(self._num_agents):
      agent = self.env.unwrapped.agents[i]
      ax, ay = agent.pos
      agent_img = MultiGrid.render_tile(self.env.unwrapped.grid.get(ax, ay), tile_size=self._tile_size)
      agent_img = rgb2gray(agent_img)

      agent_layer = np.zeros(shape=(self._height * self._tile_size, self._width * self._tile_size), dtype=np.uint8)
      xmin, xmax, ymin, ymax = get_tile_range(ax, ay, self._tile_size)
      agent_layer[ymin:ymax, xmin:xmax] = agent_img
      
      agent_layer = np.expand_dims(agent_layer, axis=-1).astype(np.uint8)

      agent_obs = np.concatenate([grid, agent_layer], axis=-1).astype(np.uint8)
      assert agent_obs.shape == self.observation_space[i].shape
      new_obs_list.append(agent_obs)
    return new_obs_list


from marlgrid.objects import Wall, BonusTile

def xy_in_dir(dir, x, y):
  if dir == 0:
    return x + 1, y
  elif dir == 1:
    return x, y + 1
  elif dir == 2:
    return x - 1, y
  elif dir == 3:
    return x, y - 1
  else:
    assert False

class FullGridImgObservationCompact(gym.ObservationWrapper):
  def __init__(self, env):
    gym.ObservationWrapper.__init__(self, env)
    self._num_agents = len(self.env.unwrapped.agents)
    assert self._num_agents == len(self.env.observation_space)

    self._width = self.env.unwrapped.width
    self._height = self.env.unwrapped.height
    self._tile_size = 8

    one_obs_shape = (self._height, self._width, 6)
    self.observation_space = [gym.spaces.Box(
      low=0,
      high=255,
      shape=one_obs_shape,
      dtype="uint8"
    )] * self._num_agents

  def observation(self, org_obs):
    assert len(org_obs) == self._num_agents

    wall_layer = np.zeros((self._height, self._width), dtype=np.uint8)
    target_layer = np.zeros((self._height, self._width), dtype=np.uint8)
    players_layer = np.zeros((self._height, self._width), dtype=np.uint8)
    players_dir_layer = np.zeros((self._height, self._width), dtype=np.uint8)

    for x in range(self._width):
      for y in range(self._height):
        obj = self.env.unwrapped.grid.get(x, y)
        if isinstance(obj, Wall):
          wall_layer[y, x] = 255
        elif isinstance(obj, BonusTile) and obj.active:
          target_layer[y, x] = 255

    
    for aid in range(len(self.env.unwrapped.agents)):
      agent = self.env.unwrapped.agents[aid]
      ax, ay = agent.pos
      players_layer[ay, ax] = 255
      adx, ady = xy_in_dir(agent.dir, ax, ay)
      players_dir_layer[ady, adx] = 255
      #print(f'player {aid}, {ax}, {ay} orientation: {agent.dir}')


    new_obs_list = []
    for aid in range(len(self.env.unwrapped.agents)):
      agent = self.env.unwrapped.agents[aid]
      agent_layer = np.zeros((self._height, self._width), dtype=np.uint8)
      agent_dir_layer = np.zeros((self._height, self._width), dtype=np.uint8)
      ax, ay = agent.pos
      agent_layer[ay, ax] = 255
      adx, ady = xy_in_dir(agent.dir, ax, ay)
      agent_dir_layer[ady, adx] = 255

      agent_obs = np.stack([wall_layer, target_layer, players_layer, players_dir_layer, agent_layer, agent_dir_layer], axis=-1)
      assert agent_obs.shape == self.observation_space[aid].shape
      new_obs_list.append(agent_obs)
      #print(f'player {aid}, {ay}, {ax} orientation: {agent.observe_orientation}')

    return new_obs_list


class ObservationForMultiHeadFromFull(gym.ObservationWrapper):
  def __init__(self, env):
    gym.ObservationWrapper.__init__(self, env)

    self._num_players = len(self.env.observation_space)
    assert len(set(map(lambda x: x.shape, self.env.observation_space))) == 1
    obs_shape = (1,) + self.env.observation_space[0].shape[:-1] + (3 + self._num_players,)


    self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype="uint8",
        )
  def observation(self, old_obs):
    full_level_obs = old_obs[0][:, :, 0:3]
    #print(full_level_obs.shape)
    obs_list = [full_level_obs]
    for i in range(self._num_players):
      player_mark_layer = np.expand_dims(old_obs[i][:, :, 3], axis=-1)
      #print(player_mark_layer.shape)
      obs_list.append(player_mark_layer)

    #print(obs_list)
    obs = np.concatenate(obs_list, axis=-1)
    obs = np.expand_dims(obs, axis=0)
    return obs

def ObservationForOneHeadFromFull(env):
  return ObservationForOneHead(env)
