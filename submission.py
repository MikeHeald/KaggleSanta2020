import numpy as np
from tensorflow import keras
import sys
sys.path.append("/kaggle_simulations/agent")

class Agent():
    def __init__(self, k, model1, model2, n_init=20, c=0.1, op_mod=-0.2, my_mod=1.0):
        self.k = k
        self.model1 = model1
        self.model2 = model2
        self.sum_r = 0
        self.r = np.zeros(k)
        self.my_mod = my_mod
        self.op_mod = op_mod
        self.my_pulls = np.zeros(k)
        self.op_pulls = np.zeros(k)
        self.in_vec = np.zeros((k, 4))
        self.c = c
        self.N = np.ones(k) * n_init

    def step(self, observation, configuration):
        if observation.step == 0:
            return int(np.random.choice(np.arange(0, self.k)))

        a = observation.lastActions[observation.agentIndex]
        op_a = observation.lastActions[1 - observation.agentIndex]

        self.my_pulls[a] += 1
        self.op_pulls[op_a] += 1

        self.N[a] += self.my_mod
        self.N[op_a] += self.op_mod

        R = observation.reward - self.sum_r
        self.r[a] += R
        self.sum_r = observation.reward

        self.in_vec[:, 0] = observation.step / 2000.0
        self.in_vec[:, 1] = self.r / 100.0
        self.in_vec[:, 2] = self.my_pulls / 100.0
        self.in_vec[:, 3] = self.op_pulls / 100.0

        q = self.model1.predict(self.in_vec).flatten()
        q += self.model2.predict(self.in_vec).flatten()
        q /= 2.0

        u = self.c * np.sqrt(2 * np.log(observation.step) / self.N)

        return np.argmax(q + u)

agent = None
model1 = keras.models.load_model('/kaggle_simulations/agent/model1')
model2 = keras.models.load_model('/kaggle_simulations/agent/model2')

def step(observation, configuration):
    global agent, model1, model2
    if agent == None or observation.step == 0:
        agent = Agent(configuration.banditCount, model1, model2)

    return agent.step(observation, configuration)
