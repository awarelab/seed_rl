from marlgrid.envs.cluttered import ClutteredMultiGrid
from marlgrid.agents import GridAgentInterface
import gym
import numpy as np
from marlgrid.custom.wrappers import FullGridImgObservationCompact, ObservationForOneHeadFromFull, ObservationForMultiHeadFromFull, FullGridImgObservation, LogPlayerEntropy, DumpMovies, TeamRewardForOneHead, ListOfDiscreteToMultiDiscrete, RewardForMultiHead, ObservationForOneHead, ObservationForMultiHead
from marlgrid.base import MultiGrid

import matplotlib.pyplot as plt

#env = ClutteredMultiGrid(agents, grid_size=15, n_clutter=10)
env = gym.make("MarlGrid-6CollectorMultiGrid15x15-v0")

#env = ListOfDiscreteToMultiDiscrete(env)

#env = TeamRewardForOneHead(ObservationForOneHead(env))
#env = ObservationForMultiHead(env)


#env = LogPlayerEntropy(env, './')

#env = DumpMovies(env, './')

env = ObservationForOneHeadFromFull(FullGridImgObservationCompact(env))

for i_episode in range(200):

    obs_array = env.reset()

    print(env.observation_space)
    #exit(0)


    episode_over = False
    plt.ion() # enables interactive mode
    agent_id = 0
    env.render()
    #a = input()
    while not episode_over:
        env.render()
        #sleep(10)

        # Get an array with actions for each agent.
        action_array = env.action_space.sample()#agents.action_step(obs_array)

        print(action_array)

        #action_array[0] = int(input())


        #imgplot = plt.imshow(obs_array[agent_id])

        # x = env.unwrapped.grid.render(8)
        
        # print(x.shape)
        # px, py = env.unwrapped.agents[0].pos
        # print(px, py)
        # print(env.unwrapped.grid.get(px, py))
        # x = MultiGrid.render_tile(env.unwrapped.grid.get(px, py), tile_size=8)

        print(obs_array.shape)
        x = obs_array[-1]
        plt.imshow(x[:, :, 0:3])
        input()
        plt.imshow(x[:, :, 3:])
        input()
        # for i in range(3):
            
        #     #plt.imshow(x[:, :, (3 + i):(4 + i)])
        #     plt.imshow(obs_array[i][:, :, 0:3])
        #     input()
        #     plt.imshow(obs_array[i][:, :, 3])
        #     input()

        #plt.imshow(x)


        #imgplot = plt.imshow(rgb2gray(x))

        # action = input()
        # if int(action) == 3:
        #     agent_id = agent_id - 1

        # if int(action) == 4:
        #     agent_id = agent_id + 1

        # action_array[agent_id] = action

        print(action_array)

        print(env.action_space)
        # Step the multi-agent environment
        next_obs_array, reward_array, done, _ = env.step(action_array)
        
        
        #e#xit(0)

        #assert ((0 <= next_obs_array) & (next_obs_array <= 255)).all()
        


        #print(next_obs_array.shape)
        print(reward_array)

        # Save the transition data to replay buffers, if necessary
        #agents.save_step(obs_array, action_array, next_obs_array, reward_array, done)

        obs_array = next_obs_array

        episode_over = done
        # or if "done" is per-agent:
        #episode_over = all(done) # or any(done)
