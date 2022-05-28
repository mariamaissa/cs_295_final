import numpy as np
import copy
import random
import torch

class QHD_Model(object):
    def __init__(self,
                dimension=10000,
                n_actions=2,
                n_obs=4,
                epsilon=1.0,
                epsilon_decay=0.995,
                minimum_epsilon=0.01,
                reward_decay=0.9,
                mem=70, #70#50#200 (not in use)
                batch=20, #20#10#50 (not in use)
                lr = 0.05, #0.05 for cartpole and acrobot
                train_sample_size = 5 # for training after each step
                ):
        self.D = dimension
        self.n_actions = n_actions
        self.n_obs = n_obs
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.minimum_epsilon = minimum_epsilon
        self.reward_decay = reward_decay
        self.mem = mem
        self.batch = batch
        self.lr = lr
        self.ts = train_sample_size
        
        self.logs = [] # temp log for current episode
        self.episode_logs = [] # logs for past episodes
        self.model = []
        ''' ## encoding is different for each model/action
        self.s_hdvec = []
        self.bias = []
        for a in range(n_actions):
            tmp_hdvec = []
            for n in range(self.n_obs):
                tmp_hdvec.append(np.random.normal(0, 1, dimension))
            tmp_hdvec = np.array(tmp_hdvec)
            self.s_hdvec.append(tmp_hdvec)
            self.bias.append(np.random.uniform(0,2*np.pi, size=dimension))
            self.model.append(np.zeros(dimension, dtype=complex))
        '''
        for a in range(n_actions):
            self.model.append(np.zeros(dimension, dtype=complex))
        self.s_hdvec = []
        for n in range(self.n_obs):
            self.s_hdvec.append(np.random.normal(0, 1, dimension))
        self.bias = np.random.uniform(0,2*np.pi, size=dimension)

        # self.s_hdvec = np.array(self.s_hdvec)
        # self.bias = np.array(self.bias)
        # self.model = np.array(self.model)
        self.s_hdvec = torch.FloatTensor(self.s_hdvec)
        self.bias = torch.FloatTensor(self.bias)
        self.model = torch.tensor(self.model,dtype=torch.cfloat)
        self.delay_model = copy.deepcopy(self.model)
        
        self.model_update_counter = 0
        self.tau = 5 # for cartpole (not in use)
    
    def store_transition(self, s, a, r, n_s):
        self.logs.append((s,a,r,n_s))
        if len(self.logs) > 102400: # if the mem is full, POP
            self.logs.pop(0)

    def choose_action(self, observation): # observation should be numpy ndarray
        if (random.random() <= self.epsilon):
            self.action = random.randint(0, self.n_actions-1)
        else:
            observation = torch.FloatTensor(observation)
            q_values = torch.zeros(self.n_actions)
            '''
            for action in range(self.n_actions):
                q_values.append(self.value(action, observation))
            '''
            encoded = torch.exp(1j* (torch.matmul(observation, self.s_hdvec)+self.bias))
            for action in range(self.n_actions):
                q_values[action] = torch.real(torch.matmul(torch.conj(encoded), self.model[action])/self.D)

            self.action = torch.argmax(q_values)
        return self.action
    
    def value(self, action, observation, delay=False):
        ## Encoding
        #encoded = np.exp(1j* (np.matmul(observation, self.s_hdvec[action])+self.bias[action]))
        encoded = torch.exp(1j* (torch.matmul(observation, self.s_hdvec)+self.bias))
        if delay == True:
            q_value = torch.real(torch.matmul(torch.conj(encoded), self.delay_model[action])/self.D)
        else:
            q_value = torch.real(torch.matmul(torch.conj(encoded), self.model[action])/self.D)
        return q_value
    
    ## NOT changed to Torch yet
    def feedback(self):
        ## Update the delayed model
        self.model_update_counter += 1
        if self.model_update_counter > self.tau:
            self.delay_model = copy.deepcopy(self.model)
            self.model_update_counter = 0

        self.episode_logs.append(self.logs)
        if len(self.episode_logs) > self.mem: # if the mem is full, POP
            self.episode_logs.pop(0)

        for iter in range(20): #15
            if len(self.episode_logs) < self.batch:
                indexs = list(range(len(self.episode_logs)))
            else:
                indexs = random.sample(list(range(len(self.episode_logs))), self.batch)
            for i in indexs:
                episode_logs = self.episode_logs[i]
                if len(episode_logs) < 3:
                    idx = list(range(len(episode_logs)))
                else:
                    idx = random.sample(list(range(len(episode_logs))), 3) + [len(episode_logs)-1]
                for j in idx:
                    log = episode_logs[j]
                    (obs, action, reward, next_obs) = log
                    y_pred = self.value(action, obs)
                    q_list = []
                    '''
                    for a_ in range(self.n_actions):
                        q_list.append(self.value(a_, next_obs, True))
                    y_true = reward + self.reward_decay*max(q_list)
                    encoded = np.exp(1j* (np.matmul(obs, self.s_hdvec[action])+self.bias[action]))
                    '''
                    encoded = np.exp(1j* (np.matmul(obs, self.s_hdvec)+self.bias))
                    encoded_ = np.exp(1j* (np.matmul(next_obs, self.s_hdvec)+self.bias))
                    for a_ in range(self.n_actions):
                       q_list.append(np.real(np.matmul(np.conjugate(encoded_), self.delay_model[a_])/self.D))
                    y_true = reward + self.reward_decay*max(q_list)

                    # model_size = np.linalg.norm(self.model[action])/self.D
                    # if model_size != 0:
                    #     print(action, model_size)
                    #    self.model[action] += self.lr * model_size * (y_true-y_pred) * encoded
                    #else:
                    self.model[action] += self.lr * (y_true-y_pred) * encoded
                    #print(y_true-y_pred)

    def new_feedback(self): # for stepwise model update
        for iter in range(1):
            if len(self.logs) < self.ts: #10 or 5(unoptimized) for acrobot 5 for cartpole
                logs = self.logs
            else:
                logs = random.sample(self.logs, self.ts)
            for k in range(1):
                for log in logs:
                    (obs, action, reward, next_obs) = log
                    obs = torch.FloatTensor(obs)
                    next_obs = torch.FloatTensor(next_obs)
                    y_pred = self.value(action, obs)
                    q_list = []
                    
                    # for a_ in range(self.n_actions):
                    #     q_list.append(self.value(a_, next_obs, True))
                    # y_true = reward + self.reward_decay*max(q_list)
                    #encoded = np.exp(1j* (np.matmul(obs, self.s_hdvec[action])+self.bias[action]))
                    
                    encoded = torch.exp(1j* (torch.matmul(obs, self.s_hdvec)+self.bias))  # (1x36)*(36xD)=(1xD)
                    encoded_ = torch.exp(1j* (torch.matmul(next_obs, self.s_hdvec)+self.bias))
                    for a_ in range(self.n_actions):
                       q_list.append(torch.real(torch.matmul(torch.conj(encoded_), self.delay_model[a_])/self.D))
                    y_true = reward + self.reward_decay*max(q_list)
                    # model_size = np.linalg.norm(self.model[action])/self.D
                    # if model_size != 0:
                    #     print(action, model_size)
                    #    self.model[action] += self.lr * model_size * (y_true-y_pred) * encoded
                    #else:
                    self.model[action] += self.lr * (y_true-y_pred) * encoded  # 1*(1)*(1xD)=(1xD)
                    #print(y_true-y_pred)












