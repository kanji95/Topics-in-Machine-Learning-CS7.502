import sys
from io import StringIO
import numpy as np
from contextlib import closing

import gym
from gym.envs.toy_text import discrete

# RIGHT = 0
# DOWN = 1
# LEFT = 2
# UP = 3

# ACTIONS = {UP: (0, -1), RIGHT: (1, 0), LEFT: (-1, 0), DOWN: (0, 1)}

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

ACTIONS = {UP: (-1, 0), RIGHT: (0, 1), LEFT: (0, -1), DOWN: (1, 0)}


# def categorical_sample(prob_n):

#     prob_n = np.asarray(prob_n)
#     csprob_n = np.cumsum(prob_n)
#     return (csprob_n > np.random.rand()).argmax()


class GridWorld(gym.Env):
    def __init__(self, grid_size, wind):

        self.grid_size = grid_size
        self.wind = wind

        self.shape = (grid_size, grid_size)
        self.start_state_index = np.ravel_multi_index(
            (grid_size - 1, 0), self.shape, order="C"
        )
        self.terminal_state = np.ravel_multi_index(
            (0, grid_size - 1), self.shape, order="C"
        )

        self.state = self.start_state_index

        self.state_space = np.prod(self.shape)
        self.action_space = 4

        self.transition_probability = np.array(
            [
                [
                    [
                        self._transition_probability(i, j, k)
                        for k in range(self.state_space)
                    ]
                    for j in range(self.action_space)
                ]
                for i in range(self.state_space)
            ]
        )

    def _transition_probability(self, S, A, S_):

        state = np.array(np.unravel_index(S, self.shape, order="C"))
        next_state = np.array(np.unravel_index(S_, self.shape, order="C"))
        action = np.array(ACTIONS[A])

        corner = {0, self.grid_size - 1}
        exp_state = state + action

        # Not Neighboring states
        if np.abs(state - next_state).sum() > 1:
            return 0.0

        # Reachable state through action A
        if np.all(exp_state == next_state):
            return 1.0 - self.wind + self.wind / self.action_space

        # Not Reachable through action A
        if not np.all(state == next_state):
            return self.wind / self.action_space

        # state == next_state
        if (state[0] in corner) and (state[1] in corner):
            if not (
                (0 <= exp_state[0] < self.grid_size)
                and (0 <= exp_state[1] < self.grid_size)
            ):
                return 1 - self.wind + 2 * self.wind / self.action_space
            else:
                return 2 * self.wind / self.action_space
        else:
            if (state[0] not in corner) and (state[1] not in corner):
                return 0.0
            if not (
                (0 <= exp_state[0] < self.grid_size)
                and (0 <= exp_state[1] < self.grid_size)
            ):
                return 1.0 - self.wind + self.wind / self.action_space
            else:
                return self.wind / self.action_space

    def optimal_policy(self, state):
        state = np.array(np.unravel_index(state, self.shape, order="C"))

        if state[0] == 0:
            return RIGHT
        if state[1] in {0, self.grid_size - 1} or state[0] == 2:
            return UP
        if state[0] == self.grid_size - 1:
            return RIGHT
        if state[1] == 3:
            return UP
        if state[0] == 1 or state[1] == 2:
            return RIGHT
        return UP

    def step(self, action):

        transitions = self.transition_probability[self.state][action]
        next_state = np.random.choice(self.state_space, p=transitions)
        transition_prob = transitions[next_state]

        reward = 0
        finished = False

        self.state = next_state
        if next_state == self.terminal_state:
            reward = 1
            finished = True
        return (transition_prob, next_state, reward, finished)

    def reset(self):
        self.state = self.start_state_index
        return self.state

    def render(self):
        outfile = sys.stdout

        for state in range(self.state_space):

            if self.state == state:
                output = " x "
            elif state == self.terminal_state:
                output = " T "
            else:
                output = " o "

            position = np.unravel_index(state, self.shape)
            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")


if __name__ == "__main__":
    gridworld = GridWorld(5, 0.3)
    gridworld.render()
    is_done = False
    while not is_done:
        action = np.random.choice(gridworld.action_space)
        print(action)
        prob, state, reward, is_done = gridworld.step(action)

        gridworld.render()
    # print(gridworld.P)
    # print(gridworld._transition_probability(0, 0, 1))
