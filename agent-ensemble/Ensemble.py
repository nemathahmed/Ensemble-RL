#Ensemble of SARSA, REINFORCE, Q-Learning

import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats as sts
import seaborn as sns
import math
from rl.agents import SARSAAgent
from REINFORCE import REINFORCE
from Q_Learning import CartPoleQAgent
import collections
import numpy as np
import gym
from datetime import datetime
import json
import os
# %tensorflow_version 1.14
from keras.layers import Dense, Flatten, Activation
from keras.models import Sequential
from keras.optimizers import Adam

from rl.agents import SARSAAgent
from rl.policy import EpsGreedyQPolicy
from rl.callbacks import TrainEpisodeLogger
from keras.callbacks import History
from keras.models import Model
from keras.layers import Input, Lambda
import keras.backend as K
from rl.core import Agent
from rl.agents.dqn import mean_q
from rl.util import huber_loss
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
from rl.util import get_object_config


def create_log_directory():
    now = datetime.now()
    dt_string = now.strftime("%d_%m-%H_%M")
    
    log_dir="log/"+dt_string
    os.makedirs(log_dir, exist_ok=True)
    plot_dir="graphs/"+dt_string
    os.makedirs(plot_dir, exist_ok=True)
    return log_dir,plot_dir



plot_save=True
save_logs=True

if save_logs==True:
    log_dir,plot_dir=create_log_directory()

def load_q_learning():
    agent = CartPoleQAgent()
    steps = agent.train()

    return agent 

def load_REINFORCE_agent():
    ## Config ##
    import tensorflow as tf
    ENV="CartPole-v1"
    RANDOM_SEED=1
    N_EPISODES=500

    # random seed (reproduciblity)
    np.random.seed(RANDOM_SEED)
    # tf.random.set_seed(RANDOM_SEED)

    # set the env
    env=gym.make(ENV) # env to import
    env.seed(RANDOM_SEED)
    env.reset() # reset to env 
    reinforce_agent=REINFORCE(env)

    reinforce_agent.load_model("REINFORCE_model.h5") 

    return reinforce_agent

def load_SARSA_agent():
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
        model = Sequential()
        model.add(Flatten(input_shape = (1, states)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(actions, activation='linear'))
        return model

    #Getting our neural network
    model = agent(states, actions)
    #Defining SARSA Keras-RL agent: inputing the policy and the model
    sarsa = SARSAAgent(model=model, nb_actions=actions, policy=EpsGreedyQPolicy())
    #Compiling SARSA with mean squared error loss
    sarsa.compile('adam', metrics=["mse"])

    #Training the agent for 50000 steps
    sarsa.fit(env, nb_steps=50000, visualize=False, verbose=1)

    return sarsa,env


q_agent = load_q_learning()
reinforce_agent=load_REINFORCE_agent()
sarsa,env=load_SARSA_agent()
n_action = 2
print("SARSA, Q-Learning and REINFORCE Agents Loaded")
agents = [reinforce_agent, q_agent, sarsa]
agent_name=['REINFORCE',"Q-Learning","SARSA"]
votes=['majority_vote','average_prob','boltzmann_prob']

##Ensembling Method

def majority_vote(p1, p2, p3):
    '''
    Takes three different probability vectors in and outputs a randomly sampled 
    action from n_action according to majority voting scheme
    '''
    a = range(n_action)
    a1 = np.random.choice(a=a, p=p1)
    a2 = np.random.choice(a=a, p=p2)
    a3 = np.random.choice(a=a, p=p3)
    l = [a1, a2, a3]
    return max(set(l), key=l.count)

def average_prob(p1, p2, p3):
    '''
    Takes three different probability vectors in and outputs a randomly sampled 
    action from n_action with probability equals the average probability of the
    input vectors
    '''
    a = range(n_action)
    p = (p1 + p2 + p3)/3
    p = p/np.sum(p)
    a = np.random.choice(a=a, p=p)
    return a

def boltzmann_prob(p1, p2, p3, T=0.5):
    '''
    Takes three different probability vectors in and outputs a randomly sampled 
    action from n_action with probability equals the average probability of the 
    normalized exponentiated input vectors, with a temperature T controlling
    the degree of spread for the out vector
    '''
    a = range(n_action)
    boltz_ps = [np.exp(prob/T)/sum(np.exp(prob/T)) for prob in [p1, p2, p3]]
    p = (boltz_ps[0] + boltz_ps[1] + boltz_ps[2])/3
    p = p/np.sum(p)
    a = np.random.choice(a=a, p=p)
    return a


def ensembler_play(learners, env, episodes, vote="majority_vote"):
  '''
  Takes in the agents, the environment and number of episodes to perform
  ensemble learning for some episodes of play from the environment
  '''
  rewards = []
  n_action = env.action_space.n
  for episode in range(episodes):
    ep_reward = 0
    done=False
    state=env.reset()
    ps = []
    while not done:
      
      _, p = learners[0].get_action(state)
      ps.append(p)
      p = learners[1].get_action(state, 500)
      ps.append((p + np.max(p) + 1)/np.sum(p + np.max(p) + 1))
      q_values = learners[2].compute_q_values(state.reshape(1, 4))
      q_values = q_values.reshape((1, 2))
      probs=q_values[0]
      probs/=np.sum(probs)
      ps.append(probs)
      # print(ps)
      if vote == "majority_vote":
          action = majority_vote(ps[0], ps[1], ps[2])
      elif vote == "average_prob":
          action = average_prob(ps[0], ps[1], ps[2])
      elif vote == "boltzmann_prob":
          action = boltzmann_prob(ps[0], ps[1], ps[2])
      else: raise Exception("Not implemented voting scheme")
    
      next_state, reward, done,info=env.step(action)
      ep_reward += reward
      state=next_state

      if done:
        rewards.append(ep_reward)
        ep_reward = []
        env.reset()
  
  return np.mean(rewards),rewards

def save_logs_f(dic,name):

  with open(log_dir+'/'+name+'.json', 'w') as fp:
      json.dump(dic, fp)





game_runs=20
dic={}
for vote in votes:
    print(("Ensemble Running For: {} voting").format(vote))
    mean_rewards_list=[]
    dic[vote]={}
    dic[vote]['mean_reward']=[]
    dic[vote]['game_run_rewards']={}
    for game_run in range(game_runs):
        mean_reward,rewards=ensembler_play(agents, env, 100,vote=vote)
        dic[vote]['mean_reward'].append(mean_reward)
        dic[vote]['game_run_rewards'][str(game_run)]=list(rewards)
        
    if plot_save==True:
        plt.figure(figsize=(12,8))
        plt.hist( dic[vote]['mean_reward'])
        plt.xlabel("Average number of consecutive step")
        plt.title("Voting Mechanism: "+vote)
        plt.savefig(plot_dir+"/Ensemble_"+vote+".png")
        plt.close()
if save_logs==True:
    save_logs_f(dic,"Ensemble_rewards")



def learner_play1(learner, env, episodes, vote="majority_vote"):
  rewards = []
  n_action = env.action_space.n
  for episode in range(episodes):
    ep_reward = 0
    done=False
    state=env.reset()
    ps = []
    while not done:
      _, p = learner.get_action(state)
      action = np.argmax(p)
      next_state, reward, done,info=env.step(action)
      ep_reward += reward
      state=next_state

      if done:
        rewards.append(ep_reward)
        ep_reward = []
        env.reset()
  
  return np.mean(rewards),rewards

def learner_play2(learner, env, episodes, vote="majority_vote"):
  rewards = []
  n_action = env.action_space.n
  for episode in range(episodes):
    ep_reward = 0
    done=False
    state=env.reset()
    ps = []
    while not done:
      p = learner.get_action(state, 500)
      (p + np.max(p) + 1)/np.sum(p + np.max(p) + 1)
      action = np.argmax(p)
      next_state, reward, done,info=env.step(action)
      ep_reward += reward
      state=next_state

      if done:
        rewards.append(ep_reward)
        ep_reward = []
        env.reset()
  
  return np.mean(rewards),rewards

def learner_play3(learner, env, episodes, vote="majority_vote"):
  rewards = []
  n_action = env.action_space.n
  for episode in range(episodes):
    ep_reward = 0
    done=False
    state=env.reset()
    ps = []
    while not done:
      q_values = learner.compute_q_values(state.reshape(1, 4))
      q_values = q_values.reshape((1, 2))
      probs=q_values[0]
      probs/=np.sum(probs)
      action = np.argmax(probs)
      next_state, reward, done,info=env.step(action)
      ep_reward += reward
      state=next_state

      if done:
        rewards.append(ep_reward)
        ep_reward = []
        env.reset()
  
  return np.mean(rewards),rewards

dic={}
vote="majority_vote"
for vote in votes:
    agent_learn_funcs=[learner_play1,learner_play2,learner_play3]
    for i in range(len(agents)):
        print(("{} running for {}").format(vote,agent_name[i]))
        agent=agents[i]
        mean_rewards_list=[]
        
        dic[agent_name[i]]={}
        dic[agent_name[i]]['mean_reward']=[]
        dic[agent_name[i]]['game_run_rewards']={}
        for game_run in range(game_runs):
            mean_reward,rewards=agent_learn_funcs[i](agent, env, 100,vote=vote)
            dic[agent_name[i]]['mean_reward'].append(mean_reward)
            dic[agent_name[i]]['game_run_rewards'][str(game_run)]=list(rewards)
            
        if plot_save==True:
            plt.figure(figsize=(12,8))
            plt.hist(dic[agent_name[i]]['mean_reward'])
            plt.xlabel("Average number of consecutive step")
            plt.title("Agent: "+agent_name[i] + " Voting: "+vote)
            plt.savefig(plot_dir+"/"+agent_name[i]+"_"+vote+".png")
            plt.close()

if save_logs==True:
    save_logs_f(dic,"ind_"+vote)
