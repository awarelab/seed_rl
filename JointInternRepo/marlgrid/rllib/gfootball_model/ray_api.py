from .gfootball_no_pack_bits import GFootball as GFootballNoPackBits
from .base_vtrace_network import AgentOutput

from ray.rllib.models.tf.tf_modelv2 import TFModelV2


from absl import logging
import tensorflow as tf

import gym

#logging.set_verbosity(logging.DEBUG)

class MarlGridRayGFootballNoPackBits(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):

        super(MarlGridRayGFootballNoPackBits, self).__init__(obs_space, action_space,
                                           num_outputs, model_config, name)
        if isinstance(action_space, gym.spaces.Discrete):
            self._action_specs = [action_space.n]
        else:
            self._action_specs = action_space.nvec

        self.num_outputs = num_outputs
        self.base_model = GFootballNoPackBits(self._action_specs)

        #logging.debug('specs: %s %s %s', obs_space, action_space, num_outputs)
        # create variables
        dummy_input = {
            'obs': tf.zeros(shape=(1,) + obs_space.shape),
            'is_training': False
        }

        self._call_base_model(dummy_input, ())

        self.register_variables(self.base_model.variables)

        self._last_model_out = None

    def _call_base_model(self, input_dict, state):
        #logging.debug('input: %s', input_dict)
        prev_actions = input_dict.get('prev_action', ())
        obs = input_dict['obs']
        #logging.debug('obs shape: %s', obs.shape)
        env_outputs = ((), (), obs) # todo add rest
        unroll = False
        is_training = input_dict['is_training']
        postprocess_action = True

        model_out, state = self.base_model(prev_actions=prev_actions, 
                                            env_outputs=env_outputs, 
                                            core_state=state,
                                            unroll=unroll,
                                            is_training=is_training,
                                            postprocess_action=postprocess_action)
        return model_out, state



    def forward(self, input_dict, state, seq_lens):

        
        
        
        
        model_out, state = self._call_base_model(input_dict, state)
        

        self._last_model_out = model_out

        #logging.debug('Logits: %s', model_out.policy_logits.shape)
        #logging.debug('State: %s', state)

        return model_out.policy_logits, state

    def value_function(self):
        #logging.debug('Baseline: %s', self._last_model_out.baseline)
        return self._last_model_out.baseline