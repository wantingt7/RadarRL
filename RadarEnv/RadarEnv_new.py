import gym
import numpy as np
from gym import Env
from gym import spaces
from gym.utils import seeding
from collections import deque, Counter
from chi2comb import chi2comb_cdf, ChiSquared

# parameters
pulse = 16
sigma = [5, 3, 1]
pf = 1e-4  # false alarm rate
gcoef = 2
freedom_value = 2
B = 2e6  # bandwidth for each subpulse. The radar does not transmit signals with only single frequency
T0 = 290
B0 = 500e6  # noise bandwidth
k = 1.38e-23
N = k * T0 * B  # noise power
P = 1e5  # the power of radar transmitter for each subpulse
G = 10 ** (30 / 10)  # the antenna gain
f = 3e9  # carrier frequency. Ignore the influence caused by the change of the carrier frequency
lam = 3e8 / f  # the wavelength
R = 1e5  # range between the radar and the jammer
P_j = 10  # the power of the jammer transmitter
G_j = 10 ** (3 / 10)  # the antenna gain


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
        self.type = 3

        # radar parameters
        self.pf = pf
        # calculate sinr
        # 未被干扰，三个频点回波信噪比
        self.sinr = P * G ** 2 * lam ** 2 * np.array(sigma) / ((4 * np.pi) ** 3 * R ** 4 * N)
        # 被瞄准式干扰干扰的情况下，三个频点分别回波信干噪比
        self.pj_receive_spot = P_j * G * G_j * lam ** 2 / ((4 * np.pi) ** 2 * R ** 2)
        # 被压制干扰干扰的情况下，三个频点回波信干噪比
        self.pj_receive_barrage = P_j / B0 * B * G * G_j * lam ** 2 / ((4 * np.pi) ** 2 * R ** 2)
        # Different sigma
        self.sigma_unjammed = 1
        self.sigma_spot = self.pj_receive_spot / N
        self.sigma_barrage = self.pj_receive_barrage / N
        # self.unjammed_vector = [self.sigma_unjammed, self.sigma_unjammed, self.sigma_unjammed]
    
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

        # sinr and sigma in this state
        freq_state, sinr_state, sigma_state = self.jamOrNot(self.state, self.length)
        # Reward: pd
        pd = self.pd_cal(freq_state, sinr_state, sigma_state)

        # Check if it is done
        if self.length <= pulse-2:
            self.length += 1
            done = False
        else:
            done = True
        info = {}

        return self.state, pd, done, info

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

    # def jammer_act01(self, a, cpi):
    #     # randomly choose a policy every 8 steps
    #     if cpi % 8 == 0:
    #         self.type = self.np_random.choice([1,2,3])
    #     aj = self.Env_choose(a, cpi)
    #     return aj
    #
    # def jammer_act02(self, a, cpi, state):
    #     # choose a policy based on previous actions
    #     if cpi % 8 == 0:
    #         radar_act = state[0, :, :3*(cpi+1)]
    #         jammer_act = state[1, :, :3*(cpi+1)]
    #         same_act = np.bitwise_and(radar_act == 1, jammer_act == 1)
    #         loc_SameAct = np.where(same_act == 1)[1]  # postion of all successful jamming
    #         num = np.array(loc_SameAct).size
    #         if num/(3*(cpi+1)) <= 0.3:
    #             self.type = self.np_random.choice([1, 2, 3])
    #     aj = self.Env_choose(a, cpi)
    #     return aj

    # def jammer_act_simple(self, a, cpi):
    #     aj = self.Env_choose(a, cpi)
    #     return aj

    def jamOrNot(self, state, cpi):
        sinr = np.array(self.sinr)
        his_RadarFreq = []
        his_sinr = []
        his_sigma = []
        state = np.copy(state)
        for i in range(cpi+1):
            # get his_RadarFreq and his_sinr
            loc_RadarAct = np.where(state[0, :, 3*i:3*i+3] == 1)[0]
            his_RadarFreq.extend(loc_RadarAct)
            his_sinr.extend([sinr[loc_RadarAct[0]], sinr[loc_RadarAct[1]], sinr[loc_RadarAct[2]]])
            # get his_sigma
            if state[2, 0, 3*i] == 2:  # no jammer
                his_sigma.extend([self.sigma_unjammed, self.sigma_unjammed, self.sigma_unjammed])
            elif state[2, 0, 3*i] == 1:  # barrage jammer
                his_sigma.extend([self.sigma_barrage, self.sigma_barrage, self.sigma_barrage])
            else:  # spot jammer
                radar_act = state[0, :, 3*i:3*i+3]
                jammer_act = state[1, :, 3*i:3*i+3]
                same_act = np.bitwise_and(radar_act == 1, jammer_act == 1)
                loc_SameAct = np.where(same_act == 1)[1]  # postion of all successful jamming
                if loc_SameAct.size == 0:
                    his_sigma.extend([self.sigma_unjammed, self.sigma_unjammed, self.sigma_unjammed])
                else:
                    unjammed_vec = [self.sigma_unjammed, self.sigma_unjammed, self.sigma_unjammed]
                    for i in loc_SameAct:
                        unjammed_vec[i] = None
                    his_sigma.extend(unjammed_vec)

        his_RadarFreq = np.array(his_RadarFreq)
        his_sigma = np.array(his_sigma)
        his_sinr = np.array(his_sinr)
        # print(his_RadarFreq)
        # print(his_sigma)
        # print(his_sinr)

        return his_RadarFreq, his_sinr, his_sigma  # History of all previous pulses

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

    def pd_cal(self, his_RadarFreq, his_sinr, his_sigma):
        # Remove None part (None means successfully spot jamming)
        his_RadarFreq = his_RadarFreq[his_sigma != None]
        his_sinr = his_sinr[his_sigma != None]
        his_sigma = his_sigma[his_sigma != None]

        # get cumulated sinr, sigma
        sinr_, sigma_ = [], []
        for i in range(3):
            sinr_part = his_sinr[his_RadarFreq == i]
            sigma_part = his_sigma[his_RadarFreq == i]
            if len(sinr_part):
                sinr_.append(np.sum(np.sqrt(sinr_part)) ** 2)
                sigma_.append(np.sum(sigma_part))
        sinr_ = np.array(sinr_)
        sigma_ = np.array(sigma_)
        T, _ = self.pfcalculation(sinr_, self.pf, sigma_)
        pd = self.pdcalculation(sinr_, T, sigma_)
        return pd


    # SWD
    def pfcalculation(self, snr, pf, Sigma):
        """
        :param snr: the estimation snr vector, which needs to be a numpy array type
        :param pf: given the probability of false alarm
        :return: the threshold
        """
        step, iter = 0.1, int(500)
        coefs = (snr / (snr + Sigma))
        dofs = freedom_value * np.ones(snr.shape[0])
        ncents = np.zeros(snr.shape[0])
        chi2s = [ChiSquared(coefs[i], ncents[i], dofs[i]) for i in range(snr.shape[0])]
        p = []
        interval = 1
        for k in range(0, iter, interval):
            result, _, _ = chi2comb_cdf(k * step, chi2s, gcoef)
            p.append(1 - result)
        posi = np.where(np.array(p) <= pf / 10)
        T = posi[0][0] * step * interval
        return T, p

    def pdcalculation(self, snr, T, Sigma):
        """
        :param snr: the estimation snr vector, which needs to be a numpy array type
        :param T: the threshold
        :return: the detection probability
        """
        coefs = snr / Sigma
        dofs = freedom_value * np.ones(snr.shape[0])
        ncents = np.zeros(snr.shape[0])
        chi2s = [ChiSquared(coefs[i], ncents[i], dofs[i]) for i in range(snr.shape[0])]
        result, _, _ = chi2comb_cdf(T, chi2s, gcoef)
        pd = 1 - result
        return pd

    # Different jammer policies
    def Env_choose(self, a, cpi):
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

