from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.deep_q_agent import DeepQAgent
from gym.wrappers import ResizeObservation
from max_frameskip_env import MaxFrameskipEnv
from reward_cache_env import RewardCacheEnv
from penalize_death_env import PenalizeDeathEnv

import tensorflow as tf
import keras

env = gym_super_mario_bros.make('SuperMarioBros-v1')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = RewardCacheEnv(env)
env = ResizeObservation(env, (64,64))
env = PenalizeDeathEnv(env)
env = MaxFrameskipEnv(env)

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

#env = gym.make('SpaceInvaders-v0')

agent = DeepQAgent(env=env,render_mode=None, dueling_network=True, prioritized_experience_replay=True)

agent.train()
#"""
#agent.model.load_weights("model.h5")

"""
print(env.observation_space)

done = True
for step in range(50000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(agent.predict(state, 0.05))
    env.render()
#"""
