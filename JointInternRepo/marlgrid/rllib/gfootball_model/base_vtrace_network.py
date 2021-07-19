import collections
import tensorflow as tf

AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')


class BaseVTraceNetwork(tf.Module):
  def __init__(self, name):
    super(BaseVTraceNetwork, self).__init__(name=name)

  def initial_state(self, batch_size):
    return NotImplementedError

  def __call__(self, input_, core_state, unroll, is_training):
    return NotImplementedError
