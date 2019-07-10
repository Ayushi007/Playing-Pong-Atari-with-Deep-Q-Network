# Playing Atari (Pong) with Deep Q learning
I have implemented the [Playing Atari with Deep Reinforcement Learning](https://github.com/Ayushi007/Playing-Pong-Atari-with-Deep-Q-Network/blob/master/DeepMind-dqn.pdf) paper.

The model needs to be trained to optimize two values: loss and reward of the game. Loss needs to be minimized and rewards maximized.
There are various parameters that could be trained to get the optimum result (that is loss and reward of game). Our ultimate goal of Deep Q-learning to train the Atari game to maximize the reward with respect to one player.

The following parameters gave me best results:

num_frames | 2000000
|--------|-------|
Batch_size | 32
Gamma | 0.98
Replay_initial |	9200
Replay buffer length |	160000

The best result that my model could reach is as follows:
Loss | 0.00343
Reward |19.9

## Visualizing the training of the model
I took 1000 frames sequentially on which model is trained. Then, I took my trained model (trained on above parameter), and removed the last layer of Neural network. Then, I got a tensor of dimension 1000 X 512. In order to plot this and realize how the training happens in model, dimensionality reduction needs to be done. I first tried PCA for that and then ultimately used t-sne (as results from PCA were not good and I didn’t get meaningful clusters).
The side information used by me is action taken and the state (that is state of board at that time). So, in the plot shown, points are basically 2-dimensional reduction of 1000 X 512 features. And color is based on the action chosen by model (for that particular state in the model). Then, three groups of frames are selected. First few frames from start of training, then second set consists of some frames just before the end of game (that is when done is true) and third set is some frames in the middle of training. These frames are chosen from the above 1000 frames only. As evident from plot, each set is clustered together. Also, I have plotted state of each of the frames selected (in 3 sets) and frames of same set, that is , frames which are close by are similar in their state of board.
The manifold are not very smooth in the visualization. This is due to the fact that the model diverge a lot in Q-learning to explore more and doesn’t follow a fixed path. Also, maintaining a memory buffer, ensure that the next state are chosen at random and thus, due to this randomized pattern, manifold is not smooth. But, we can see that same type of board state and same type of actions are clustered together (as evident from side information).
Color: Based on action chosen (from 6 possible action set)
Board drawn for each action chosen


Visualization of Trained Model
Side Information: Action and State (of Board)
