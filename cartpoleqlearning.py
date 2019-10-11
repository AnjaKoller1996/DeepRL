import gym
import numpy as np


class QCartPoleSolver():
    def __init__(self, input=(1, 1, 6, 12), number_of_episodes=200, alpha=1, epsilon=1, gamma=1):
        self.input = input
        self.number_of_episodes = number_of_episodes
        self.alpha = alpha  # learningrate
        self.epsilon = epsilon  # explorationrate
        self.gamma = gamma  # discountfactor (no discount--> gamma=1)
        # models fact that future rewards are worth less than immediate rewards

        self.env = gym.make('CartPole-v0')  # CartPolev0 environment from gym

        self.Q = np.zeros(self.input + (self.env.action_space.n,))  # Q table
        #self.Q = np.zeros(self.env.action_space.n,)

        #  we have to discretize because in space of observations and actions is continuous
        #  and we have to discretize it down into a discrete set of possibilities

    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], np.math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -np.math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.input[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.input[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def action_choosing(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])
        #  with a probability epsilon choose a random action, else choose the maxstate of the Q table

    def update_Qtable(self, oldstate, action, reward, newstate, alpha):
        self.Q[oldstate][action] += alpha * (reward + self.gamma * np.max(self.Q[newstate]) - self.Q[oldstate][action])
        #  algorithm from script Q-learning
        #  the agent maps states and actions to Q-values, s.t. given any state the best
        #  action can be picked by choosing the highest q-value

    def get_alpha(self):
        return self.alpha  # gettermetho
        #alpha= 0 means that q values are never updated (nothing learned), high value--> learning quickly
        #learningrate

    def get_epsilon(self):
        return self.epsilon


    def runalg(self):
        #  run algorithm
        # iterate over the episodes
        scores= list()

        for episodes in range(self.number_of_episodes):
            current_state= self.env.reset()
            iterator = 0
            done = False  # boolean to check if we are finished or not
            alpha = self.get_alpha()
            epsilon = self.get_epsilon()

            while not done:
                action = self.action_choosing(current_state, epsilon)
                obs, reward, done, info = self.env.step(action)
                newstate= self.discretize(obs)
                self.update_Qtable(current_state, action, reward, newstate, alpha)
                current_state = newstate
                iterator += 1


            scores.append(iterator)
             #evaluate mean score using method from numpy
            mean_score = np.mean(scores)
            print(mean_score)
            if episodes >= 100:
                print('Ran {} episodes. Solved after {} trials'.format(episodes, episodes-100))
                return episodes -100
            if episodes % 100 == 0:
                print('Episode {} - Mean survival time over last 100 episodes was {} ticks.'.format(episodes, mean_score))

        if False:
         print ('Did not solve after {} episodes'.format(episodes))
        return episodes

if __name__ =="__main__":
    solver= QCartPoleSolver()
    solver.runalg

    #  try to have a small set of states as this results in a smaller Q-table so we need less steps
            #  until the agent learns its value
            #  original domains: cart_pos [-4.8,4.8], cart_velocity [-3.4 10^38, 3.4 10^38], angle [-0.42,0.42], angle_velocity[-3.4 10^38, 3.4 10^38]
            #  cart_pos, angle, angle_velocity (angular velocity) , pos is the position
            #  velocity extremely large --> scale down cart_velocity to [0,6] and angle_velocity[0,12]
            #  seen from @tuzzers post  --> drop cart_pos and cart_velocity completely (map their values to single scalar)
            #  because probability of cart leaving environment at border in only 200 time steps (after 200 it automatically resets itself)

#  implementation of Q-learning alg: function to fetch teh best action for a state from the q-table
#  and another function to update the q-tabe based on the last action

#  other parameters, alpha= learning rate, epsilon= explorationrate, gamma= discountfactor
#  as seen in the book for these values (with chance epsilon a random action is picked)
# learning rate: smoothes updates and makes them less critical
# exploration rate: regulates between exploitation and exploration (with chance epsilon a random action is picked)
# prevents the alg from stuck in a local minima
# gamma (discountfactor): used to penalize the agent if it takes to long to reach its goal (no discount means constant 1)
# if we set alpha and epsilon as constants no good score, so we use adaptive learning and exploration rate
# starts in a high value and decreases by time
