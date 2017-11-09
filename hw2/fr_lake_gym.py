import gym
import matplotlib.pyplot as plt
import numpy as np
import tqdm

# Solve frozen lake environment
def get_info():
    env = gym.make('FrozenLake-v0')
    a = env.action_space.sample()
    print(a)
    s = env.reset()
    info = env.step(a)
    nS, nA = env.env.nS, env.env.nA
    print(nS, nA)
    print(env.observation_space.n)



def value_iteration(env_name, gamma, nIter):
    env = gym.make(env_name)
    nS, nA, P = env.env.nS, env.env.nA, env.env.P
    s = env.reset()
    end = False
    # initialize V`s
    Vs = [np.zeros(nS)]
    pis = []
    for iter in range(nIter):
        oldpi = pis[-1] if len(pis) > 0 else None  # \pi^{(it)} = Greedy[V^{(it-1)}]. Just used for printout
        Vprev = Vs[-1]  # V^{(it)}
        # Values and policy on this iteration
        V = np.zeros(nS)
        pi = np.zeros(nS)
        # For each state
        for x in range(nS):
            # For each action in this state, get values, update Vk+1=max_a sum_{x_, r} p(x_, r|x, u)[r+Vk(x)]
            acs_val = np.zeros(nA)
            for u in range(nA):
                sum_ = 0
                # P[s][a] == [(prob, next_state, reward, terminal)...]
                for T, s_, r, _ in P[x][u]:
                    sum_+= T*(r+gamma*Vprev[s_])
                acs_val[u] = sum_
            V[x] = np.max(acs_val)
            pi[x] = np.argmax(acs_val)
        max_diff = np.abs(V - Vprev).max()
        if max_diff<0.00001:
            break
        nChgActions = "N/A" if oldpi is None else (pi != oldpi).sum()
        print("%4i      | %6.5f      | %4s          | %5.3f" % (iter, max_diff, nChgActions, V[0]))
        Vs.append(V)
        pis.append(pi)
    return Vs, pis


def act(env_name, policy, trials, render=False, record=False):
    env = gym.make(env_name)
    # recording
    if record:
        from gym import wrappers
        env = wrappers.Monitor(env, '/tmp/' + env_name,force=True)
    for _ in tqdm.tqdm(range(trials)):
        s = env.reset()
        end = False
        iter = 0
        rew = []

        while not end:
            a = int(policy[s])
            s_, r, end, _ = env.step(a)
            if render:
                env.render()
            iter+=1
            s = s_
        rew.append(r)
    env.close()
    print("Mean reward over {} trials: {}".format(trials, np.mean(rew)))

def policy_iteration(env_name, gamma, nIter):
    env = gym.make(env_name)
    nS, nA, P = env.env.nS, env.env.nA, env.env.P
    env.reset()
    def compute_vpi(pi, gamma):
        # v = (1-gamma*P)^{-1}*R - linear system, matrix form
        R = np.zeros(nS, dtype=np.float32)
        P_ = np.zeros((nS, nS), dtype=np.float32)

        for x in range(nS):
            # action according to policy
            u = pi[x]
            for T, x_, r, _ in P[x][u]:
                P_[x][x_] = T
                R[x] += T * r
        V = np.linalg.inv(np.eye(len(P_)) - gamma * P_).dot(R)
        return V

    def compute_qpi(vpi, gamma):
        # set Q-value matrix
        Qpi = np.zeros((nS, nA))
        for x in range(nS):
            for u in range(nA):
                sum_x = 0
                for T, x_, r,_ in P[x][u]:
                    sum_x += T * (r + gamma * vpi[x_])
                Qpi[x][u] = sum_x
        return Qpi

    Vs = []
    pis = []
    # Initialize policy
    pi_prev = np.zeros(nS, dtype='int')
    pis.append(pi_prev)
    print("Iteration | # chg actions | V[0]")
    print("----------+---------------+---------")
    for it in range(nIter):
        # policy evaluation
        vpi = compute_vpi(pi_prev, gamma)
        # Policy improvement
        qpi = compute_qpi(vpi, gamma)
        pi = qpi.argmax(axis=1)
        print("%4i      | %6i        | %6.5f" % (it, (pi != pi_prev).sum(), vpi[0]))
        Vs.append(vpi)
        pis.append(pi)
        pi_prev = pi
    return Vs, pis

if __name__=='__main__':
    get_info()
    num_steps = 250
    env_name = 'FrozenLake8x8-v0'
    values, policy = value_iteration(env_name, 0.90, num_steps)
    plt.figure()
    plt.plot(values)
    plt.title("Value iteration values for {} steps".format(len(values)))
    act(env_name, policy=policy[-1], trials=100, record=True)
    plt.show()
    # gym.upload('/tmp/'+env_name, api_key='sk_3N2vwXfNQguGzx3IIDGb9w')
    # values, policy = policy_iteration(env_name, 0.95, num_steps)
    # plt.figure()
    # plt.plot(values)
    # plt.title("Policy iteration values for {} steps".format(len(values)))
    # act(env_name, policy=policy[-1], trials=100)
    # plt.show()
