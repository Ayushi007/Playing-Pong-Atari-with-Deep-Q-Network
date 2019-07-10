from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import pandas as pd
import pickle

USE_CUDA = torch.cuda.is_available()
from dqn import QLearner, compute_td_loss, ReplayBuffer

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

# print("action_space---------:" ,env.action_space.n, env.unwrapped.get_action_meanings())
# 6 ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
# print("action_space---------:" ,env.observation_space.shape) #(1, 84, 84)

num_frames = 2000000
batch_size = 32
gamma = 0.98 #Decrease 80-90

replay_initial = 9200 #Decrease 5000
replay_buffer = ReplayBuffer(160000) #Increase 200000
model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
if USE_CUDA:
    model = model.cuda()

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000 #May be decrease - not asked
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

losses = []
all_rewards = []
episode_reward = 0

state = env.reset()

error_plot = []
rewards_plot = []
columns = ["state","qval","action"]
df_qval = pd.DataFrame(columns=columns)
#df_ = df_.fillna(0)

for frame_idx in range(1, num_frames + 1):

    epsilon = epsilon_by_frame(frame_idx)
    #action = model.act(state, epsilon)
    action,q_value_entity = model.act(state, epsilon)
    #if q_value_entity:
    #    df_qval.loc[frame_idx] = q_value_entity

    # if len(df_qval)==2:
    #     print(df_qval)
    #     df_qval.to_csv("QVal_test.csv")
    #     break

    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)

    state = next_state
    episode_reward += reward

    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0

    #if len(replay_buffer) == replay_initial:
    #    print("10k File stored")
    #    df_qval.to_csv("QVal_one.csv")

    if len(replay_buffer) > replay_initial:
        loss = compute_td_loss(model, batch_size, gamma, replay_buffer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.cpu().numpy())

    if frame_idx % 10000 == 0 and len(replay_buffer) <= replay_initial:
        print('#Frame: %d, preparing replay buffer' % frame_idx)

    if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:
        print('#Frame: %d, Loss: %f' % (frame_idx, np.mean(losses)))
        print('Last-10 average reward: %f' % np.mean(all_rewards[-10:]))
        rewards_plot.append(np.mean(all_rewards[-10:]))
        error_plot.append(np.mean(losses))

np.savetxt("rewards_two.csv", rewards_plot, delimiter=",")
np.savetxt("error_two.csv", error_plot, delimiter=",")

torch.save(model,"Model2.pt")

state, action, reward, _, _ = replay_buffer.sample(1000)

state = Variable(torch.FloatTensor(np.float32(state)))
action = Variable(torch.LongTensor(action))
reward = Variable(torch.FloatTensor(reward))
comb = (state,action,reward)
torch.save(comb,"Buffer2.csv")
