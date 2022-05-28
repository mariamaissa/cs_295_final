import gym
from QHD_Model import QHD_Model
from time import time
import copy

#task = 'cartpole' #tau=1, ts=4
task = 'pong' # tau=1, ts=10
#task = 'mountaincar'
#task = 'acrobot' #tau=1, ts=4
# if task == 'cartpole':
#     env = gym.make('CartPole-v0')
#     filename = 'hdc_cartpole_results_intel.csv'
# if task == 'lunarlander':
#     env = gym.make('LunarLander-v2')
#     filename = 'hdc_lunarlander_results_intel_6.csv'
# if task == 'mountaincar':
#     env = gym.make('MountainCar-v0')
#     filename = 'hdc_mountaincar_results_intel.csv'
# if task == 'acrobot':
#     env = gym.make('Acrobot-v1')
#     filename = 'hdc_acrobot_results_intel.csv'
if task == "pong":
    env = gym.make("Pong-v0")
    filename = 'hdc_pong_results_intel.csv'

env = env.unwrapped
#env.seed(0)
print("env.action_space", env.action_space)
print("env.observation_space", env.observation_space)
print("env.observation_space.high", env.observation_space.high)
print("env.observation_space.low", env.observation_space.low)
dimension = 6000 #6000
#dimension_list = [250,500,1000,2000,4000,6000,8000,10000]
#for dimension in dimension_list:
ts_list = [10] #64 10
tau = 1
tau_step = 5 #20 5
for ts in ts_list:
    for i in range(1): # iterations for collecting averaged results
        EPISODES = 501
        if task == 'cartpole':
            epsilon = 0.2
            epsilon_decay = 0.99
            reward_decay = 0.9
            EPISODES = 201
        if task == 'lunarlander':
            epsilon = 0.5 #0.8
            epsilon_decay = 0.99
            reward_decay = 0.99 #0.9
            EPISODES = 1001
        if task == 'mountaincar':
            epsilon = 0.2
            epsilon_decay = 0.99
        if task == 'acrobot':
            epsilon = 0.2
            epsilon_decay = 0.99
            reward_decay = 0.99
        if task == 'pong':
            epsilon = 0.2
            epsilon_decay = 0.99
            reward_decay = 0.99
        minimum_epsilon = 0.01

        n_actions = env.action_space.n
        n_obs = env.observation_space.shape[0]

        model = QHD_Model(dimension, n_actions, n_obs, epsilon, epsilon_decay,
                        minimum_epsilon, reward_decay, train_sample_size=ts, lr=0.05)

        with open(filename,'a') as f:
           f.write("Episode,Reward,Runtime\n")

        total_runtime = 0
        total_step = 0
        for episode in range(EPISODES):
            start = time()
            rewards_sum = 0
            obs = env.reset() # observation is numpy array
            model.n_steps = 0 # reset the episode step counter

            while True:
                action = int(model.choose_action(obs))
                new_obs, reward, done, info = env.step(action)
                if task == 'cartpole':
                    if done:
                        reward = -5
                if task == 'mountaincar':
                    if new_obs[0] >= 0.4:
                        reward += 1
                    if done:
                        reward += 1
                if task == 'acrobot':
                    if done:
                        reward = 0
                model.store_transition(obs, action, reward, new_obs)
                rewards_sum += reward
                if task == 'cartpole':
                    model.new_feedback()
                if task == 'lunarlander':
                    model.new_feedback()
                if task == 'mountaincar':
                    model.new_feedback()
                    if model.n_steps > 4000:
                        done = True
                if task == 'acrobot':
                    model.new_feedback()
                    if model.n_steps > 500:
                        done = True

                # model update for lunarlander
                total_step += 1
                if task == 'lunarlander':
                    if total_step % tau_step == 0:
                        model.delay_model = copy.deepcopy(model.model)
                    if model.n_steps > 1000:
                        done = True

                # if we have a bad ending, learn from it (cartpole)
                # learn when the episode ends (mountaincar)
                if done:
                    if task == 'cartpole':
                        #model.feedback()
                        if episode % tau == 0: # 5
                            model.delay_model = copy.deepcopy(model.model)
                    #if task == 'lunarlander': 
                    #    if episode % tau == 0:
                    #        model.delay_model = copy.deepcopy(model.model)
                    if task == 'mountaincar':
                        if episode % tau == 0:
                            model.delay_model = copy.deepcopy(model.model)
                    if task == 'acrobot':
                        if episode % tau == 0: # 1
                            model.delay_model = copy.deepcopy(model.model)

                if rewards_sum > 1000: # for cartpole & lunarlander
                    done = True
                
                if done: # end this episode
                    end = time()
                    total_runtime += end - start
                    print('Episode: ', episode)
                    print('Episode Rewards: ', rewards_sum)
                    print('Total Runtime: ', total_runtime)
                    with open(filename,'a') as f:
                       f.write(str(episode)+','+str(rewards_sum)+','+str(total_runtime)+'\n')
                    break

                model.n_steps += 1
                obs = new_obs

            model.epsilon = max(model.epsilon * model.epsilon_decay, 
                                    model.minimum_epsilon)