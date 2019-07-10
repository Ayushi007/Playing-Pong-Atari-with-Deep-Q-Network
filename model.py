from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
import pandas as pd
from dqn import QLearner
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib
#matplotlib.use('Agg')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.switch_backend('TkAgg')
plt.switch_backend('agg')

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
USE_CUDA = torch.cuda.is_available()

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

state = env.reset()

model = torch.load("Model2.pt")
final_state=[]
final_action=[]
final_reward=[]
end = 1000
checked=True
for frames in range(1, 1001):
    epsilon = epsilon_by_frame(frames)
    action, q_val = model.act(state, epsilon)
    #state = np.expand_dims(state, 0)
    #final_state.append(state)
    print(frames)
    #print(action)
    #print(state)
    next_state, reward, done, info = env.step(action)
    if done and checked:
        end = frames
        checked=False
        print("DONEEEEEEEEE")
    #if (frames < 100) or (frames>450 and frames<550) or (frames>892 and frames>992):
    final_reward.append(reward)
    final_state.append(state)
    final_action.append(action)
    state = next_state

for frames in range(1, 1001):
    if frames<=5 or (frames >=end/2 and frames<(end/2)+5) or (frames >=end-5 and frames <end):
        x = final_state[frames][0]
        x = x.T
        imgplot = plt.imshow(x)
        pltname = "Plot."+str(frames)+".png"
        print(pltname)
        plt.savefig("Run2/"+pltname)

#print(model)
model_dict = model.state_dict()
model.fc = model.fc[:-2]
#print(final_action)
#print(model)
#print(model_dict)


#data  = torch.load("Buffer.csv") # state, action, reward
finalfea = []
print("final_state print", final_state[0])
#for i in range(len(data[0])): #size of shape tensor
for i in range(len(final_state)):
    #state = data[0][i]
    state   = Variable(torch.FloatTensor(np.float32(final_state[i])).unsqueeze(0), requires_grad=True)
    #print(state)
    #fea = model(state)[0].tolist()

    fea = model(state)[0].tolist()
    #print("features",fea)
    finalfea.append(fea)
print("length of fea", len(finalfea))
print("finalfea[0]",len(finalfea[0]))

X_embedded = TSNE(n_components=2, perplexity=50, n_iter = 3000, early_exaggeration = 10).fit_transform(finalfea)
#perplexity=50, n_iter=4000, early_exaggeration = 8
#perplexity = 45, learning_rate=100, n_iter=50
print(X_embedded.shape)
x_df = pd.DataFrame(X_embedded)
x_df.columns = ["PC1","PC2"]
x_df["Actions"] = final_action
x_df["group"] = x_df.index
x_df["Group of frames"] = 0
x_df['size'] = 1
x_df.loc[:5,'Group of frames']=1
x_df.loc[end/2:(end/2)+5,'Group of frames']=2
x_df.loc[end-5:end,'Group of frames']=3
x_df.loc[:5,'size']=0
x_df.loc[end/2:(end/2)+5,'size']=0
x_df.loc[end-5:end,'size']=0
print(end)
#y_df=x_df[:5]
#y_df=y_df.append(x_df[end/2:(end/2)+5])
#y_df=y_df.append(x_df[end-5:end])
#print(y_df)

print(x_df.head())
plt.figure(figsize=(16,10))

sns.scatterplot(
        x="PC1", y="PC2",
        hue="Actions",
        style="Group of frames",
        size='size',
        sizes=(20,200),
        palette=sns.color_palette("Set1", 6),
        data=x_df,
        legend="full",
        alpha=1
    )
# ax.text()
plt.savefig('T-sne2')
#plt.show()
