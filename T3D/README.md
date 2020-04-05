
### Initialization

    import os
    import time
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import pybullet_envs
    import gym
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from gym import wrappers
    from torch.autograd import Variable
    from collections import deque
    
### Step 1

    class ReplayBuffer(object):
    def __init__(self, max_size=1e6):      
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []

        for i in ind: 
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))

        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)

### Step 2

    class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)        
        self.max_action = max_action
        
    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x
### Step 3

    class Critic(nn.Module):
    
    def __init__(self, state_dim, action_dim):
        super(Critic, self().__init__())
        # 1st critic NN
        self.layer_1 = nn.Linear(state_dim+action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        # 2nd critic NN
        self.layer_4 = nn.Linear(state_dim+action_dim, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, action_dim)
   
    def forward(self, x, u):
        xu = torch.cat([x,u],1)
        #Forward for 1st critic NN
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        #Forward for 2nd critic NN
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2
    
    def Q1(self, x, u):
        #concat
        xu = torch.cat([x,u],1)#axis = 1 --> concat vertically
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1
### Step 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	class TD3:
	    def __init__(self, state_dim, action_dim, max_action):
	        self.actor = Actor(state_dim, action_dim, max_action).to(device)
	        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
	        self.actor_target.load_state_dict(self.actor.state_dict())
	        self.actor_optimizer = optim.Adam(self.actor.parameters())
	        self.critic = Critic(state_dim, action_dim).to(device)
	        self.critic_target = Critic(state_dim, action_dim).to(device)
	        self.critic_target.load_state_dict(self.critic.state_dict())
	        self.critic_optimizer = optim.Adam(self.critic.parameters())
	        self.max_action = max_action
	    
	    def select_action(self, state):
	        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
	        return self.actor(state).cpu().data.numpy().flatten()

	    def train(self, replay_buffer, iterations, batch_size=100, discount, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):	    
		    for it in range(iterations):
		        batch_states, batch_next_batch_states, batch_action, batch_reward, batch_done = replay_buffer.sample(batch_size)
		        state = torch.FloatTensor(batch_states).to(device)
		        next_batch_state = torch.FloatTensor(batch_next_batch_states).to(device)
		        action = torch.FloatTensor(batch_action).to(device)
		        reward = torch.FloatTensor(batch_reward).reshape((batch_size,1)).to(device)        
		        done = torch.FloatTensor(batch_done).reshape((batch_size,1)).to(device)

### Step 5

    next_action = self.actor_target.forward(next_state)
### Step 6

    noise = torch.Tensor(batch_actions).data.normal_(0,policy_noise).to(device)
    noise = noise.clamp(-noise_clip, noise_clip)
    next_action = (next_action+noise).clamp(-self.max_batch_action, self.max_batch_action)
### Step 7

    target_Q1, target_Q2 = self.critic_target.forward(next_states, next_action)
    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))

### Step 8

    target_Q = torch.min(target_Q1, target_Q2)
### Step 9

    target_Q = reward + ((1-done) * discount * target_Q).detach()
    
### Step 10

    current_Q1, current_Q2 = self.critic.forward(state, action)

### Step 11

    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

### Step 12

    self.critic_optimizer.zero_grad()
	critic_loss.backward()
	self.critic_optimizer.step()
### Step 13

     if it % policy_freq == 0:
	    actor_loss = -(self.critic.Q1(state, self.actor(state)).mean())
	    self.actor_optimizer.zero_grad()
	    actor_loss.backward()
	    self.actor_optimizer.step()


### Step 14

    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
	    target_param.data.copy_( (tau * param.data) + ((1-tau) * target_param.data))
### Step 15

    for param, target_param in  zip(self.critic.parameters(), self.critic_target.parameters()):
        target_param.data.copy_( tau * param.data + (1-tau) * target_param.data)
