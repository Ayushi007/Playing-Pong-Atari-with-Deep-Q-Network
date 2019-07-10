from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math, random
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
USE_CUDA = torch.cuda.is_available()

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)    #Kind of network
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon):
        combined = ''
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)

            '''act_values[0] looks like this: [0.67, 0.2], each numbers
            representing the reward of picking action 0 and 1. And argmax
            function picks the index with the highest value. In the example of [0.67, 0.2],
            argmax returns 0 because the value in the 0th index is the highest.'''

            q_value = self.forward(state)
            _, act_v = torch.max(q_value, dim=1)
            action = int(act_v.item())
            #STORE
            # dict1['state'] = {}
            # for i in range(84):
            stateitem = state.tolist()[0][0]
            #print(stateitem,len(stateitem))#" ".join(str(x) for x in state.tolist())
            valueitem = q_value.tolist()[0]#" ".join(str(x) for x in q_value.tolist())
            actionitem = action
            combined = [stateitem,valueitem,str(actionitem)]
        else:
            action = random.randrange(self.env.action_space.n)
        return action,combined

def compute_td_loss(model, batch_size, gamma, replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    action = [action[i:i+1] for i in range(0, len(action), 1)]
    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), requires_grad=False)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = torch.ByteTensor(done)



    #state_action_values = model.forward(state).max(1)[0]#.gather(1, action.unsqueeze(-1)).squeeze(-1)
    state_action_values = model.forward(state).gather(1, action).squeeze(-1)

    next_state_values = model.forward(next_state).max(1)[0]
    data = torch.zeros(32, dtype=torch.float32)
    #print("DONE", next_state_values)
    for i in range(len(done)):
        if bool(done[i]):
            # print("DONE", next_state_values)
            next_state_values[i] = 0
            # print("me andar hu")
            # print(next_state_values)

    next_state_values[done] = 0
    next_state_values = next_state_values.detach() #Check if extra

    expected_state_action_values = next_state_values * gamma + reward
    loss = nn.MSELoss()(state_action_values, expected_state_action_values)

    return loss


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, next_state, done  = zip(*[self.buffer[idx] for idx in indices])

        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)
