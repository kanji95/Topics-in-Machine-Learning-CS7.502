import pickle

import gym

import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

from sklearn.kernel_approximation import RBFSampler

from utils import *

N_SAMPLES = 100000
BASIS_DIM = 26
N_BINS = 120

def discretize_state(state):
    state_d = (state - state_low) * np.array([x_m, v_m])
    state_d = np.round(state_d, 0).astype(int)
    
    return state_d

def get_next_state(state, action):
    
    env.reset()
    env.state = state
    
    next_state, _, _, _ = env.step(action)
    
    return next_state

env = gym.make("MountainCar-v0")

observation_space = env.observation_space

state_low = observation_space.low
state_high = observation_space.high

x_m = (N_BINS - 1)/(state_high[0] - state_low[0])
v_m = (N_BINS - 1)/(state_high[1] - state_low[1])

x_c = - x_m * state_low[0]
v_c = - v_m * state_low[1]

print(discretize_state(state_low))
print(discretize_state(state_high))

sampled_states = np.array([observation_space.sample() for _ in range(N_SAMPLES)])

state_x = sampled_states[:, 0]
state_x_norm = np.linalg.norm(state_x)

state_x = state_x / state_x_norm

gaussian_basis = RBFSampler(gamma=0.7, n_components=BASIS_DIM)
gaussian_basis.fit(state_x[:, None])

# alpha = np.random.rand(BASIS_DIM)

Q = q_learning(env, x_m, v_m, alpha=0.1, gamma=0.8, epsilon=0.9, num_iter=100000, bins=N_BINS)
# Q = np.load('q_optimal.npy')
print("Q shape: ", Q.shape)

V = value_iteration(env, x_m, v_m, Q, gaussian_basis, state_x_norm, alpha = 0.1, gamma = 0.8, bins=N_BINS, feature_dim=BASIS_DIM, MAX_ITER=10000)
# V = np.load('value_function.npy')
print("V shape: ", V.shape)

# num states
N = np.prod(Q.shape[:2])

# num actions
A = Q.shape[-1]

c = np.zeros(BASIS_DIM)
bounds = np.zeros((BASIS_DIM, 2))

bounds[:, 0] = -1
bounds[:, 1] = 1

for i in range(N_SAMPLES):
    
    state = sampled_states[i]
    state_d = discretize_state(state)
    
    a_star=np.argmax(np.array(Q[state_d[0], state_d[1]]))
    
    other_actions = np.arange(A)
    other_actions = np.delete(other_actions, a_star)
    
    other_action = np.random.choice(other_actions)
    
    state_ = get_next_state(state, a_star)
    state_d_ = discretize_state(state_)
    
    # other_states_ = [get_next_state(state, a) for a in other_actions]
    # other_states_d_ = np.array([discretize_state(s_) for s_ in other_states_])
    
    other_state_ = get_next_state(state, other_action)
    other_state_d_ = discretize_state(other_state_)
    
    v_diff = V[other_state_d_[0], other_state_d_[1], :] - V[state_d_[0], state_d_[1], :]
    v_diff[v_diff > 0] *= 2*v_diff[v_diff > 0]
    c += v_diff

solver=linprog(c, bounds=bounds)
alphas=solver['x']
print(alphas)

# x_states_orig = np.linspace(state_low[0], state_high[0], num=1000)
# x_states = x_states_orig / np.linalg.norm(x_states_orig)

x_states = np.linspace(state_low[0], state_high[0], num=1000)
phis = gaussian_basis.transform(x_states[:, None])

R = phis @ alphas[:, None]
R = R[:, 0]

plt.plot(x_states, R)
plt.show()