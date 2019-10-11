import numpy as np
import gym
import time
from pprint import pprint

THETA = 1e-20   # convergence_boundary
MAX_ITERATIONS = 1000 # maximum number of iterations we do for policy iteration


def run_episode(env, policy, gamma, render = False):
    '''
    runs an episode an returns the total reward
    :param env: given environment on which operation behaves
    :param policy: policy to be used (according to the policy we take a step in our environment)
    :param gamma: discountfactor
    :param render: per default false (if set to true we see for every episode the optimal step)
    :return: total reward (over all episodes)
    '''

    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        if render:
            env.render()
        obs, reward, done , info = env.step(int(policy[obs]))
        #total_reward += (gamma * reward) # it we want to take gamma into account here
        total_reward += reward
    return total_reward


def score_eval(env, policy, gamma, n_episodes = 100):
    '''
    high score is good means total reward is high
    :param env: environment on which we run the episodes
    :param policy: policy according to which the step is chosen in run_episode (function we call to compute the score)
    :param gamma: discountfactor
    :param n_episodes: total number of episodes (per default 100)
    :return:total score achieved over all episodes
    '''
    scores = [run_episode(env, policy, gamma, render= False) for episode in range(n_episodes)]
    return np.mean(scores)



def extract_policy(env, V, gamma):
    '''
    extracts the policy given a value function
    :param env:environment on which we act
    :param V:Value function (value function V(s): how good is a state for an agent to be in = expected total rewarrd for an agent
          starting from state s)
    :param gamma:discountfactor
    :return:policy by which the agent picks actions to perform
    env.P[s][a]: list of transition tuples (probability, next_state, reward, done)
    env.nS: gives total number of states in the env
    env.nA: gives total number of actions in the env
    '''

    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * V[s_new]) for p, s_new, r, _ in  env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy


def policy_eval(env, policy, gamma, theta = THETA):
    '''

    :param env: environment name [gym environment]
    :param policy: policy [array]
    :param gamma: discount factor [float]
    :return: computes the value function under the policy iteratively, returns value function V
    '''

    V = np.zeros(env.nS)
    #  V= np.ones(env.nS) # different initializations of V --> no noticeable effect
    #  V= np.random.choice(env.nA, size= (env.nS))
    while True:
        v_old = np.copy(V)  # store old value function
        for state in range(env.nS):
            policy_val = policy[state] # to compute v value for the policy
            V[state] = sum([p * (r + gamma * v_old[s_new]) for p, s_new, r, _ in env.P[state][policy_val]])
            delta = np.sum(np.abs(v_old-V))
        if (delta <= theta):
            # value converged
            break
    return V


def policy_improvement(env, gamma, use_zeros=True, theta=1e-20, max_iterations=MAX_ITERATIONS):
    '''

    :param env: environment
    :param gamma: discountfactor
    :param use_zeros: if this is true we initialize out policy with zeros (else with ones), per default zeros
    :param theta:convergence_boundary
    :param max_iterations: maximum number of policy iteration (when it does not converge it stops after MAX_ITERATIONS
    :return:policy and convergence_rounds (after how many rounds the policy iteration alg converges)
    '''

    #policy = np.random.choice(env.nA, size=(env.nS))  # initialize a random policy
    policy = np.zeros(env.nS) if use_zeros else np.ones(env.nS)
    #policy = np.ones(env.nS) --> with ones initialized more likely to end up in hole
    # no noticeable difference in performance
    for i in range(max_iterations):
        old_policy_value = policy_eval(env, policy, gamma, theta=theta)
        new_policy = extract_policy(env, old_policy_value, gamma)
        if np.all(policy == new_policy):
            convergence_rounds = i+1
            print('Policy-Iteration converges after {} rounds'.format(convergence_rounds))
            break
        policy = new_policy
    return policy, convergence_rounds


def env_test(env_name, gamma, theta=THETA, max_iterations=MAX_ITERATIONS):
    '''

    :param env_name: environment name, e.g. 'Taxi-v2'
    :param gamma: discountfactor
    :param theta: convergence_boundary parameter
    :param max_iterations:maximal iterations of policy improvement
    :return: convergence_rounds (number of rounds it needs until convergence), mean_score (mean of total reward) and total time
    '''
    print('Running %s with gamma=%s' % (env_name, gamma))
    print('Initial state:')
    env = gym.make(env_name).env
    env.reset()
    env.render()
    start_time = time.time()
    optimal_policy, convergence_rounds = policy_improvement(env, gamma=gamma)
    scores = score_eval(env, optimal_policy, gamma=gamma)
    mean_score = np.mean(scores)
    total_time = time.time() - start_time
    print('Average scores = {}'.format(mean_score))
    print('Runtime of the algorithm {}'.format(total_time))
    print('Final state:')
    env.render()
    return convergence_rounds, mean_score, total_time


if __name__ == '__main__':
    '''
    #env = gym.make('FrozenLake8x8-v0').env
    env = gym.make('Taxi-v2').env
    #env = gym.make('FrozenLake-v0').env
    env.reset()
    env.render()
    starttime= time.time()
    optimal_policy, convergence_rounds = policy_improvement(env, gamma = 0.99, theta=THETA) # 0.999 is enough to get average score of 1
    scores = score_eval(env, optimal_policy, gamma = 1.0) # the gamma here does not matter a lot
    endtime= time.time()
    total_time= endtime-starttime
    env.render() # to see if final result is right
print('Average scores = {}'.format(np.mean(scores)))
print('Runtime of the algorithm {}'.format(total_time))
'''
    env_name = 'Taxi-v2'
    envs= ['Taxi-v2', 'FrozenLake-v0', 'FrozenLake8x8-v0']
    gammas = [0.1, 0.6, 0.7, 0.8, 0.9, 0.99,0.999]
    statistics_dict = {}
    for gamma in gammas:
        convergence_rounds, mean_score, total_time = env_test(env_name=env_name, gamma=gamma)
        statistics_dict[gamma] = ['Convergence rounds: {}'.format(convergence_rounds),
                                  'Mean_score: {}'.format(mean_score),
                                'Total time: {}'.format(total_time)]

    print('Gamma results:')
    pprint(statistics_dict)
