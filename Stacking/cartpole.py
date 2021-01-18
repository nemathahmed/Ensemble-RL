import numpy as np
import gym
# %tensorflow_version 1.14
from keras.layers import Dense, Flatten, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import Model
from keras.utils.vis_utils import plot_model
from rl.agents import SARSAAgent
from rl.policy import EpsGreedyQPolicy
from rl.callbacks import TrainEpisodeLogger
from keras.layers.merge import concatenate
#Importing the necessary plotting libraries
import matplotlib.pyplot as plt
from keras.layers import Input
import seaborn as sns
sns.set()
#Setting up the environment
env = gym.make('CartPole-v1')
seed_val = 456
env.seed(seed_val)
np.random.seed(seed_val)

#Getting the state and action space
states = env.observation_space.shape[0]
actions = env.action_space.n


#Defining a Neural Network function for our Cartpole agent 
def agent(states, actions):
    """Creating a simple Deep Neural Network."""

    in_1=Input(shape = (1, states))
    fin_1=Flatten()(in_1)
    D_1=Dense(8, activation='relu')(fin_1)
    D_2=(Dense(8, activation='relu')(D_1))
    D_3=(Dense(8, activation='relu')(D_2))
    #out1=(Dense(actions, activation='linear')(D_3))
    out1=D_3
    
    D_11=(Dense(8, activation='relu')(fin_1))
    D_22=(Dense(8, activation='relu')(D_11))
    D_33=(Dense(8, activation='relu')(D_22))
    out2=D_33

    D_111=(Dense(8, activation='relu')(fin_1))
    D_222=(Dense(8, activation='relu')(D_111))
    D_333=(Dense(8, activation='relu')(D_222))
    out3=D_333
    #out2=(Dense(actions, activation='linear')(D_33))

    merged=concatenate([out1,out2,out3])
    dense2=Dense(4,activation='relu')(merged)
    outputs=Dense(actions,activation='linear')(merged)
    
    model=Model(inputs=in_1,outputs=outputs)
    plot_model(model,show_shapes=True, to_file='Model Arch_4.png')
    return model

#Getting our neural network
model = agent(states, actions)
#Defining SARSA Keras-RL agent: inputing the policy and the model
sarsa = SARSAAgent(model=model, nb_actions=actions, policy=EpsGreedyQPolicy())
#Compiling SARSA with mean squared error loss
sarsa.compile('adam', metrics=["mse"])

#Training the agent for 50000 steps
x=sarsa.fit(env, nb_steps=50000, visualize=True, verbose=1)

scores = sarsa.test(env, nb_episodes = 500, visualize= True)


#Visualizing our resulted rewards
plt.plot(x.history['episode_reward'])
plt.xlabel('Episode')
plt.ylabel('Training total reward')
plt.title('Total rewards over all episodes in training') 
plt.show()
plt.close()


#Visualizing our resulted rewards
plt.plot(scores.history['episode_reward'])
plt.xlabel('Episode')
plt.ylabel('Testing total reward')
plt.title('Total rewards over all episodes in testing') 
plt.show()