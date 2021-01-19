#Importing the necessary plotting libraries
import numpy as np
import gym
from keras.layers import Dense, Flatten, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import Model
from keras.utils.vis_utils import plot_model
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from rl.callbacks import TrainEpisodeLogger
from keras.layers.merge import concatenate
from keras.layers import Input
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
import json


#Setting up the environment
env = gym.make('CartPole-v1')
seed_val = 456
env.seed(seed_val)
np.random.seed(seed_val)
log_dir='log/'
algo='DQN'


#Getting the state and action space
states = env.observation_space.shape[0]
actions = env.action_space.n

#Callback for storing training data
def build_callbacks(filename):
    #checkpoint_weights_filename = 'dqn_' + env_name + '_weights_{step}.h5f'
    #callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=5000)]
    
    log_filename = filename+'_train.json'
    callbacks = [FileLogger(log_filename, interval=100)]
    return callbacks

#Defining a Neural Network function for our Cartpole agent 
def agent0(states, actions):
    """Creating a simple Deep Neural Network."""

    in_1=Input(shape = (1, states))
    fin_1=Flatten()(in_1)
    D_1=Dense(24, activation='relu')(fin_1)
    D_2=(Dense(24, activation='relu')(D_1))
    D_3=(Dense(24, activation='relu')(D_2))    
    outputs=Dense(actions,activation='linear')(D_3)
    
    model=Model(inputs=in_1,outputs=outputs)
    plot_model(model,show_shapes=True, to_file='models/agent0.png')
    return model

def agent1(states, actions):
    """Creating a simple Deep Neural Network."""

    in_1=Input(shape = (1, states))
    fin_1=Flatten()(in_1)
    D_1=Dense(24, activation='relu')(fin_1)
    D_2=(Dense(24, activation='relu')(D_1))
    D_3=(Dense(24, activation='relu')(D_2))
    #out1=(Dense(actions, activation='linear')(D_3))
    out1=D_3
    
    D_11=(Dense(24, activation='relu')(fin_1))
    D_22=(Dense(24, activation='relu')(D_11))
    D_33=(Dense(24, activation='relu')(D_22))
    out2=D_33


    #out2=(Dense(actions, activation='linear')(D_33))

    merged=concatenate([out1,out2])
    outputs=Dense(actions,activation='linear')(merged)
    model=Model(inputs=in_1,outputs=outputs)
    plot_model(model,show_shapes=True, to_file='models/agent1.png')
    return model

def agent2(states, actions):
    """Creating a simple Deep Neural Network."""

    in_1=Input(shape = (1, states))
    fin_1=Flatten()(in_1)
    D_1=Dense(12, activation='relu')(fin_1)
    D_2=(Dense(12, activation='relu')(D_1))
    D_3=(Dense(12, activation='relu')(D_2))
    #out1=(Dense(actions, activation='linear')(D_3))
    out1=D_3
    
    D_11=(Dense(12, activation='relu')(fin_1))
    D_22=(Dense(12, activation='relu')(D_11))
    D_33=(Dense(12, activation='relu')(D_22))
    out2=D_33


    #out2=(Dense(actions, activation='linear')(D_33))

    merged=concatenate([out1,out2])
    outputs=Dense(actions,activation='linear')(merged)
    model=Model(inputs=in_1,outputs=outputs)
    plot_model(model,show_shapes=True, to_file='models/agent2.png')
    return model

def agent3(states, actions):
    """Creating a simple Deep Neural Network."""

    in_1=Input(shape = (1, states))
    fin_1=Flatten()(in_1)
    D_1=Dense(24, activation='relu')(fin_1)
    D_2=(Dense(24, activation='relu')(D_1))
    D_3=(Dense(24, activation='relu')(D_2))
    
    out1=D_3
    
    D_11=(Dense(24, activation='relu')(fin_1))
    D_22=(Dense(24, activation='relu')(D_11))
    D_33=(Dense(24, activation='relu')(D_22))
    out2=D_33

    D_111=(Dense(24, activation='relu')(fin_1))
    D_222=(Dense(24, activation='relu')(D_111))
    D_333=(Dense(24, activation='relu')(D_222))
    out3=D_333
   

    merged=concatenate([out1,out2,out3])
    outputs=Dense(actions,activation='linear')(merged)
    model=Model(inputs=in_1,outputs=outputs)
    plot_model(model,show_shapes=True, to_file='models/agent3.png')
    return model

def agent4(states, actions):
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
    plot_model(model,show_shapes=True, to_file='models/agent4.png')
    return model

def agent5(states, actions):
    """Creating a simple Deep Neural Network."""

    in_1=Input(shape = (1, states))
    fin_1=Flatten()(in_1)
    D_1=Dense(4, activation='relu')(fin_1)
    D_2=(Dense(4, activation='relu')(D_1))
    D_3=(Dense(4, activation='relu')(D_2))
    out1=D_3
    
    D_11=(Dense(4, activation='relu')(fin_1))
    D_22=(Dense(4, activation='relu')(D_11))
    D_33=(Dense(4, activation='relu')(D_22))
    out2=D_33

    D_111=(Dense(4, activation='relu')(fin_1))
    D_222=(Dense(4, activation='relu')(D_111))
    D_333=(Dense(4, activation='relu')(D_222))
    out3=D_333
   

    merged=concatenate([out1,out2,out3])
    outputs=Dense(actions,activation='linear')(merged)
    model=Model(inputs=in_1,outputs=outputs)
    plot_model(model,show_shapes=True, to_file='models/agent5.png')
    return model
    


archs=[ agent0, agent1,agent2,agent3,agent4,agent5]
model_names=['agent0','agent1','agent2','agent3','agent4','agent5']
for i in range(len(archs)):
    arch=archs[i]
    model_name=model_names[i]
    filename=log_dir+algo+'_'+model_name
    
    memory = SequentialMemory(limit=50000, window_length=1)
    #Getting our neural network
    model = arch(states, actions)
    #Defining DQN Keras-RL agent: inputing the policy and the model
    model = DQNAgent(model=model, memory=memory,nb_actions=actions, policy=EpsGreedyQPolicy())
    #Compiling DQN with mean squared error loss
    model.compile('adam', metrics=["mse"])
    callbacks = build_callbacks(filename)
    #Training the agent for 50000 steps
    x=model.fit(env, nb_steps=50000, visualize=False, verbose=1,callbacks=callbacks)
    #Testing
    test_scores = model.test(env, nb_episodes = 1000, visualize= False)

    with open(log_dir+filename+'_test.json', 'w') as fp:
        json.dump(test_scores.history, fp)

