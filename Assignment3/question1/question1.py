import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

from gridworld import GridWorld


GRID_SIZE = 5
WIND = 0.3

LAMBDA = 1.05
GAMMA = 0.0

R_MAX = 10

grid_world = GridWorld(GRID_SIZE, WIND)

transition_probability = grid_world.transition_probability
optimal_policy = np.array(
    [grid_world.optimal_policy(state) for state in range(grid_world.state_space)]
)

N, A, _ = transition_probability.shape
print("transition_probability shape: ", transition_probability.shape)

c = np.zeros(3*N)
A_ub = np.zeros([2 * N * (A + 1), 3 * N])
b_ub = np.zeros([2 * N * (A + 1)])

c[N:2*N] = -1
c[2*N:] = LAMBDA

for state in range(N):
    a_star = optimal_policy[state]
    v_star = np.linalg.inv(
        np.identity(N) - GAMMA * transition_probability[:, a_star, :]
    )

    count = 0
    for a in range(A):
        if a != a_star:
            constraint = -np.dot(
                transition_probability[state, a_star, :]
                - transition_probability[state, a, :],
                v_star,
            )
            A_ub[state * (A - 1) + count, :N] = constraint
            A_ub[N * (A - 1) + state * (A - 1) + count, :N] = constraint
            A_ub[N * (A - 1) + state * (A - 1) + count, N + state] = 1
            count += 1

    A_ub[2 * N * (A - 1) + state, state] = 1
    A_ub[2 * N * (A - 1) + 2 * N + state, state] = 1
    A_ub[2 * N * (A - 1) + 3 * N + state, state] = 1

    A_ub[2 * N * (A - 1) + N + state, state] = -1
    A_ub[2 * N * (A - 1) + 2 * N + state, 2 * N + state] = -1
    A_ub[2 * N * (A - 1) + 3 * N + state, 2 * N + state] = -1

    b_ub[2 * N * (A - 1) + state] = R_MAX
    b_ub[2 * N * (A - 1) + N + state] = 0

solver = linprog(c, A_ub=A_ub, b_ub=b_ub)
rewards = solver.x[:N]

rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min())

x_states = np.arange(GRID_SIZE**2)
plt.plot(x_states, rewards)
plt.title(f"LAMBDA = {LAMBDA}")
plt.show()

plt.savefig(f"LAMBDA={LAMBDA}.png")


rewards = rewards.reshape(GRID_SIZE, GRID_SIZE)
print("Predicted Reward Matrix:")
print(np.around(rewards, decimals=4), "\n")

original_rewards = np.zeros((GRID_SIZE, GRID_SIZE))
original_rewards[0, 4] = 1
print("Original Reward Matrix:")
print(original_rewards)

print("Difference b/w rewards: ", np.linalg.norm(rewards - original_rewards))