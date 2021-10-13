import gym
import numpy as np

class Agent:
    def __init__(self, environment='CliffWalking-v0', gamma=0.05, theta=1e-8) -> None:
        
        self.env = gym.make(environment)
        self.env.reset()
        self.env.render()
        
        self.gamma = gamma
        self.theta = theta
        
        self.A_space = self.env.action_space
        self.S_space = self.env.observation_space
        self.R_range = self.env.reward_range

        self.Num_A = self.A_space.n
        self.Num_S = self.S_space.n

        self.V = np.random.rand(self.Num_S)
        self.Pi = np.random.randint(0, self.Num_A, (self.Num_S, ))
        
        if environment == 'CliffWalking-v0':
            for s in range(self.Num_S):
                for a in range(self.Num_A):
                    P, S_, R_, T = self.env.P[s][a][0]
                    if T:
                        self.env.P[s][a] = [(P, S_, 0, T)]
                    # print(self.env.P[s][a])
                
    
    def policy_evaluation(self):
        while True:
            delta = 0
            
            for s in range(self.Num_S):
                v = self.V[s]
                a = self.Pi[s]
                self.V[s] = sum([P*(R_ + self.gamma*self.V[S_]) for P, S_, R_, _ in self.env.P[s][a]])
                delta = max(delta, abs(v - self.V[s]))
            if delta < self.theta: 
                break

    def policy_improvement(self):
        policy_stable = True
        for s in range(self.Num_S):
            a = self.Pi[s]
            self.Pi[s] = np.argmax([sum([P*(R_ + self.gamma*self.V[S_]) for P, S_, R_, _ in self.env.P[s][a_]]) for a_ in range(self.Num_A)])
            if a!=self.Pi[s]:
                policy_stable = False
        return policy_stable
    
    def policy_iteration(self):
        
        self.V = np.random.rand(self.Num_S)
        self.Pi = np.random.randint(0, self.Num_A, (self.Num_S, ))
        
        # self.V = np.zeros(self.Num_S)
        # self.Pi = np.zeros(self.Num_S, dtype=int)
        
        while True:
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            if policy_stable:
                break
        return self.V, self.Pi
    
    def value_iteration(self):
        
        self.env.reset()
        
        # self.V = np.random.rand(self.Num_S)
        # self.Pi = np.random.randint(0, self.Num_A, (self.Num_S, ))
        
        self.V = np.zeros(self.Num_S)
        self.Pi = np.zeros(self.Num_S, dtype=int)
        
        while True:
            delta = 0
            for s in range(self.Num_S):
                v = self.V[s]
                self.V[s] = max([sum([P*(R_ + self.gamma*self.V[S_]) for P, S_, R_, _ in self.env.P[s][a_]]) for a_ in range(self.Num_A)])
                self.Pi[s] = np.argmax([sum([P*(R_ + self.gamma*self.V[S_]) for P, S_, R_, _ in self.env.P[s][a_]]) for a_ in range(self.Num_A)])
                delta = max(delta, abs(v - self.V[s]))
            if delta < self.theta: 
                break
        return self.V, self.Pi
    
    def show_policy(self):
        self.env.reset()
        
        MAX_STEPS = 1000
        
        S = 0 #self.env.start_state_index
        T = False
        step = 0
        while not T and step<MAX_STEPS:
            A = self.Pi[S]
            S, R_, done, _ = self.env.step(A)
            # self.env.render()
            step += 1
            T = done
        self.env.render()
        print("Finished", done)
        
# agent = Agent(environment='CliffWalking-v0')
# agent = Agent(environment='FrozenLake-v1')
# agent = Agent(environment='Taxi-v3')

# V, Pi = agent.policy_iteration()
# agent.show_policy()

# print("Final State Values: ", V)
# print("Final Policy: ", Pi)


agent = Agent(environment='CliffWalking-v0')
V, Pi = agent.value_iteration()
agent.show_policy()

print("Final State Values: ", V)
print("Final Policy: ", Pi)