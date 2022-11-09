import gym
import numpy as np
from gym import Env
from gym import spaces
from gym.utils import seeding
from collections import deque, Counter
from chi2comb import chi2comb_cdf, ChiSquared

# parameters
pulse = 16

class RadarEnv(Env):
    def __init__(self):
        # Initialize: actions, observations/state
        # --------------------------------------------------------------------------------#
        # Action space: 27 different actions for one pulse
        self.action_space = spaces.Discrete(27)
        # Observation space: A '3*3*pulse' board, representing the action of radar and jammer
        # layer 0: radar
        # layer 1: jammer
        # layer 2: jammer specical action: Barrage jammer
        # Observation means all possible states
        low = np.zeros((3, 3, 3*pulse))
        high = np.stack((np.ones((3,3*pulse)), np.ones((3,3*pulse)), 2*np.ones((3,3*pulse))), 0)
        self.observation_space = spaces.Box(low, high, dtype=np.uint8)
        self.cpi = pulse
        self.length = 0
        self.state = np.zeros([3, 3, 3*pulse])
        self.last_pulse = []
        self.last_few_pulse = deque(maxlen=2)
        self.done = False
        self.seed(1024)
        self.type = 2

    def set_jammer_type(self, type):
        self.type = type
    def step(self, action):
        # Apply action to current state, output contains state/observation, reward, done, info
        #---------------------------------------------------------------------------#
        # New state: Add Radar and jammer action
        self.radar_action = self.Num2Act_Radar(action)  # Radar action translate
        self.jammer_action = self.Env_choose(self.radar_action, self.length)  # Corresponding jammer act
        self.state = self.next_RadarState(self.state, self.radar_action, self.length)  # Add radar state
        self.state = self.next_JammerState(self.state, self.jammer_action, self.length)  # Add jammer state
        # Get reward
        reward_vec = self.GetReward(self.state, self.length)
        reward = reward_vec.sum()/reward_vec.size

        # Check if it is done
        if self.length <= pulse-2:
            self.length += 1
            done = False
        else:
            done = True
        info = {}

        return self.state, reward, done, info

    def reset(self):
        # reset initial state
        self.state = np.zeros([3, 3, 3*pulse])
        self.length = 0

        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        # For visualize
        pass

    # Utils
    def Num2Act_Radar(self,a):
        z, z1 = a % 3, a // 3
        if z1 < 3:
            y, x = z1, 0
        else:
            y = z1 % 3
            x = a // 9
        return [x, y, z]


    def GetReward(self, state, cpi):
        state = np.copy(state)
        reward = []
        for i in range(cpi + 1):
            # get his_RadarFreq and his_sinr
            loc_RadarAct = np.where(state[0, :, 3 * i:3 * i + 3] == 1)[0]
            # get his_sigma
            if state[2, 0, 3 * i] == 2:  # no jammer
                reward.extend([1, 1, 1])
            elif state[2, 0, 3 * i] == 1:  # barrage jammer
                reward.extend([1/3, 1/3, 1/3])
            else:  # spot jammer
                radar_act = state[0, :, 3 * i:3 * i + 3]
                jammer_act = state[1, :, 3 * i:3 * i + 3]
                same_act = np.bitwise_and(radar_act == 1, jammer_act == 1)
                loc_SameAct = np.where(same_act == 1)[1]  # postion of all successful jamming
                if loc_SameAct.size == 0:
                    reward.extend([1, 1, 1])
                else:
                    unjammed_vec = [1, 1, 1]
                    for i in loc_SameAct:
                        unjammed_vec[i] = 0
                    reward.extend(unjammed_vec)
        reward = np.array(reward)

        return reward

    def next_RadarState(self, state, action, length):
        state = np.copy(state)
        for i in range(3):
            state[0, action[i], 3*length+i] = 1
        return state

    def next_JammerState(self, state, aj, length):
        ## 0: spot jammer; 1: barrage jammer; 2: no jammer
        state = np.copy(state)
        if aj[0] == 1: # no jammer
            state[2, :, 3*length:3*length+3] = 2
        elif aj[2] == 0: # barrage jammer
            state[2, :, 3*length:3*length+3] = 1
        else:
            state[1, aj[1], 3*length:3*length+3] = 1
        return state


    # Different jammer policies
    def Env_choose(self, a, cpi):
        #
        if self.type == 1:
            if cpi % 4 == 0:
                aj = np.array([0, 3, 0])
            else:
                rdm_freq = np.random.choice([0, 1, 2], p=[0.3, 0.3, 0.4])
                aj = np.array([0, rdm_freq, 1])
        elif self.type == 2:
            aj = np.array([0, a[0], 1])
        else:
            if cpi == 0 or cpi == 1:
                aj = np.array([0, a[0], 1])
                self.last_few_pulse.append(a)
            else:
                temp_ = np.array(self.last_few_pulse).reshape(-1, 6)[0]
                collect = Counter(temp_)  # Contain all the information
                subpulse = list(collect.keys())  # different pulses contained here: 0,1,2
                num = []
                for key in subpulse:
                    num.append(collect.get(key))  # Get the number of different pulses
                prob = np.array(num) / 6
                aj_ = self.np_random.choice(subpulse, p=prob)
                aj = np.array([0, aj_, 1])
                self.last_few_pulse.append(a)
        return aj

# if __name__ == "__main__":
#     # For one pulse, if no jammer, get its maximum pd
#     state1 = np.array([[[0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#                         [0,1,1,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                         [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
#                        [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]],
#                        [[0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                         [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                         [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]])
#     reward_vec = RadarEnv().GetReward(state1, 2)
#     print(reward_vec.sum()/reward_vec.size)

