import numpy as np
import gym


def epsilon_greedy_policy(Q, S, epsilon):
    if np.random.random() > epsilon:
        action = np.argmax(Q[S[0], S[1]])
    else:
        action = np.random.choice(Q.shape[-1])
    return action


def q_learning(env, x_m, v_m, alpha=0.2, gamma=0.5, epsilon=0.5, num_iter=5000, bins=120):

    NUM_ITER = num_iter

    action_space = env.action_space.n
    observation_space = env.observation_space

    state_low = observation_space.low
    state_high = observation_space.high

    Q = np.random.rand(bins, bins, action_space)

    rewards = []

    for i in range(NUM_ITER):

        total_reward = 0
        S = env.reset()

        S_disc = (S - state_low) * np.array([x_m, v_m]) 
        S_disc = np.round(S_disc, 0).astype(int)

        done = False
        while not done:

            action = epsilon_greedy_policy(Q, S_disc, epsilon)

            S_, R, done, _ = env.step(action)

            S_disc_ = (S_ - state_low) * np.array([x_m, v_m]) 
            S_disc_ = np.round(S_disc_, 0).astype(int)

            if done and S_[0] >= 0.50:
                # print("Solved MCar!!")
                Q[S_disc[0], S_disc[1], action] = R
            else:
                Q[S_disc[0], S_disc[1], action] = Q[
                    S_disc[0], S_disc[1], action
                ] + alpha * (
                    R
                    + gamma * np.max(Q[S_disc_[0], S_disc_[1]])
                    - Q[S_disc[0], S_disc[1], action]
                )

            total_reward += R
            S_disc = S_disc_

            epsilon = max(0.0, epsilon - epsilon / NUM_ITER)

        rewards.append(total_reward)

        if (i + 1) % 2000 == 0:
            avg_reward = np.mean(rewards)
            print("Episode {} Average Reward: {}".format(i + 1, avg_reward))

    env.close()
    
    np.save('q_optimal', Q)

    return Q


def value_iteration(env, x_m, v_m, Q, gaussian_basis, state_x_norm, alpha = 0.1, gamma = 0.5, bins=120, feature_dim=26, MAX_ITER=10000):
    # alpha = 0.1
    # gamma = 0.99
    
    action_space = env.action_space.n
    observation_space = env.observation_space

    state_low = observation_space.low
    state_high = observation_space.high

    V = np.zeros((bins, bins, feature_dim))

    step = 0
    while step < MAX_ITER:
        S = env.reset()
        
        while True:
            
            S_disc = (S - state_low) * np.array([x_m, v_m]) 
            S_disc = np.round(S_disc, 0).astype(int)
            
            action = np.argmax(np.array(Q[S_disc[0]][S_disc[1]]))

            S_, R, done, _ = env.step(action)
            R = gaussian_basis.transform(np.array([S_disc[0]/state_x_norm])[:, None])[0]

            S_disc_ = (S_ - state_low) * np.array([x_m, v_m]) 
            S_disc_ = np.round(S_disc, 0).astype(int)

            V[S_disc[0], S_disc[1]] += alpha * (R + gamma * V[S_disc_[0], S_disc_[1]] - V[S_disc[0], S_disc[1]])
            if done:
                break

        step += 1
        
        if step % 1000 == 0:
            print(f"Value Iteration, Step: {step}")
            
    np.save('value_function', V)
        
    return V