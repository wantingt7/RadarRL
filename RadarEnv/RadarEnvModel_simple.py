import gym
import numpy as np
import random
from gym import Env
from gym import spaces
from gym.utils import seeding
from collections import deque, Counter
from chi2comb import chi2comb_cdf, ChiSquared
import torch

# parameters
pulse = 16

class RadarEnvModel(Env):
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
        self.test_jammer_type = 3

        self.net_type = 2       # 1 for one jammer action, 2 for jammer action distribution
        self.nid = 0
    
    def set_jammer_type(self, type):
        self.test_jammer_type = type
    
    def set_type(self, type):
        self.net_type = type

    # modified to use the network
    def step(self, action):
        # Apply action to current state, output contains state/observation, reward, done, info
        #---------------------------------------------------------------------------#
        # New state: Add Radar and jammer action
        self.radar_action = self.Num2Act_Radar(action)  # Radar action translate
        # self.jammer_action = self.jammer_act_simple(self.radar_action, self.length)  # Corresponding jammer act
        # nid = random.randint(0, self.net_num-1) 
        temp = self.net_ensemble[self.nid](torch.FloatTensor(self.state.flatten()).float())
        if self.net_type == 1:
            self.jammer_action = np.around(temp.detach().numpy()).astype(np.int64)  # Predicted jammer act
        elif self.net_type == 2:
            # self.jammer_action = temp.argmax()
            temp = temp.detach().numpy()
            self.jammer_action = np.random.choice([0, 1, 2, 3, 4], size=1, replace=True, p=temp)
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


# ---------------------------------- Network -----------------------------------------
#     def initialize_network(self, learning_rate):
#         node_num = [3*3*3*pulse, 20, 20, 4]    # input: flatten the state, output: 4 jammer action
#         self.net = torch.nn.Sequential(
#             torch.nn.Linear(node_num[0], node_num[1]),
#             torch.nn.ReLU(),
#             torch.nn.Linear(node_num[1], node_num[2]),
#             torch.nn.ReLU(),
#             torch.nn.Linear(node_num[2], node_num[3]))
#         self.loss_fn = torch.nn.CrossEntropyLoss()     # CrossEntropyLoss() expect only label not one-hot as target
#         self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

    def initialize_network(self, num_of_net, learning_rate):
        self.net_num = num_of_net
        self.net_ensemble = []
        self.optimzer_ensemble = []
        input_dim = int(np.prod(self.observation_space.shape))
        hidden_layer = 64
        if self.net_type == 1:
            for _ in range(self.net_num):
                net = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_layer),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_layer, 1))
                self.net_ensemble.append(net)
                self.optimzer_ensemble.append(torch.optim.Adam(net.parameters(),lr = learning_rate))
            self.loss_fn = torch.nn.MSELoss()

        elif self.net_type == 2:
            for _ in range(self.net_num):
                net = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_layer),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_layer, 5),
                    torch.nn.Softmax())
                self.net_ensemble.append(net)
                self.optimzer_ensemble.append(torch.optim.Adam(net.parameters(),lr = learning_rate))
            self.loss_fn = torch.nn.CrossEntropyLoss()

    def update_network(self, train_collector, batch_size, batch_num, epoch_each_batch, id_list = None):
        if id_list is None: id_list = range(self.net_num)
        else:
            if max(id_list)+1 > self.net_num or min(id_list) < 0:
                AssertionError("The id list is invalid")
        for nid in id_list:
            for _ in range(batch_num):
                batch, indice = train_collector.buffer.sample(batch_size)
                x = list()
                for i in range(len(batch.obs)):
                    x.append(batch.obs[i].flatten())
                x = torch.FloatTensor(x).float()
                if self.net_type == 1: y = torch.tensor(self.get_jammer_act(batch.obs_next, self.observation_space.shape[2] / 3)).float()
                elif self.net_type == 2: y = torch.tensor(self.get_jammer_act(batch.obs_next, self.observation_space.shape[2] / 3)).long()
                for _ in range(epoch_each_batch):
                    y_pred = self.net_ensemble[nid](x)
                    loss = self.loss_fn(y_pred, y)
                    # print(loss)
                    self.optimzer_ensemble[nid].zero_grad()
                    loss.backward()
                    self.optimzer_ensemble[nid].step()


# ---------------------------------- Utils -----------------------------------------
    # get jammer action number from states
    # Version 1
    def get_jammer_act(self, obs_next, pulse):
        # 0,1,2: spot jammer; 3: barrage jammer; 4: no jammer
        y = list()
        for obs in obs_next:
            action = 0
            round = 0
            layer1 = obs[0]
            # find jammer's current pulse in obs_next
            while (round < pulse):
                flag = True
                for i in range(3):
                    for j in range(3*round, 3*(round+1)):
                        if layer1[i][j] == 1:
                            flag = False
                            break
                if flag: break
                else: round += 1
            round -= 1
            # check the third layer first
            layer3 = obs[2]
            if layer3[0][round*3] == 1: action = 3
            elif layer3[0][round*3] == 2: action = 4
            # check the second layer
            else:
                layer2 = obs[1]
                if layer2[0][round*3] == 1: action = 0
                elif layer2[1][round*3] == 1: action = 1
                elif layer2[2][round*3] == 1: action = 2
            y.append(action)

        return np.array(y)
    
    # Version 2
    def get_jammer_act_v2(self, obs_next, pulse):
        # spot jammer: 0-[1,0,0,0,0] 1-[0,1,0,0,0] 2-[0,0,1,0,0]
        # barrage jammer: 3-[0,0,0,1,0]
        # no jammer: 4-[0,0,0,0,1]

        y = list()
        y_label = list()
        for obs in obs_next:
            action = [0,0,0,0,0]
            action_label = 0
            round = 0
            layer1 = obs[0]
            # find jammer's current pulse in obs_next
            while (round < pulse):
                flag = True
                for i in range(3):
                    for j in range(3*round, 3*(round+1)):
                        if layer1[i][j] == 1:
                            flag = False
                            break
                if flag: break
                else: round += 1
            round -= 1
            # check the third layer first
            layer3 = obs[2]
            if layer3[0][round*3] == 1:
                action[3] = 1
                action_label = 3
            elif layer3[0][round*3] == 2: 
                action[4] = 1
                action_label = 4
            # check the second layer
            else:
                layer2 = obs[1]
                if layer2[0][round*3] == 1: 
                    action[0] = 1
                    action_label = 0
                elif layer2[1][round*3] == 1: 
                    action[1] = 1
                    action_label = 1
                elif layer2[2][round*3] == 1: 
                    action[2] = 1
                    action_label = 2
            y.append(action)
            y_label.append(action_label)

        return np.array(y), np.array(y_label)
    
    """
    only for pre-trained fictitious models' testing
    Note: here, we have already known jammer's policy
    """
    def get_jammer_act_distr(self, obs_next, pulse):
        # 0,1,2: spot jammer; 3: barrage jammer; 4: no jammer
        y = list()
        for obs in obs_next:
            action = [0,0,0,0,0]
            round = 0
            layer1 = obs[0]
            # find jammer's current pulse in obs_next
            while (round < pulse):
                flag = True
                for i in range(3):
                    for j in range(3*round, 3*(round+1)):
                        if layer1[i][j] == 1:
                            flag = False
                            break
                if flag: break
                else: round += 1
            round -= 1
            # calculate the distribution
            if self.test_jammer_type == 1:
                if round % 4 == 0:
                    action[3] = 1
                else:
                    action = [0.3, 0.3, 0.4, 0, 0]
            elif self.test_jammer_type == 2:
                # check radar's first action in the latest round
                radar_action = np.argwhere(layer1[:, 3*round] == 1)
                action[radar_action[0,0]] = 1
            elif self.test_jammer_type == 3:
                if round == 0 or round == 1: 
                    radar_action = np.argwhere(layer1[:, 3*round] == 1)
                    action[radar_action[0,0]] = 1
                else:
                    # only collect the latest first 6 radar's action （i.e. 2 rounds)
                    check_state = layer1[:, 3*(round-1):3*(round+1)]
                    radar_action = np.argwhere(check_state == 1)
                    temp_ = radar_action[:, 0]
                    collect = Counter(temp_)     # Contain all the information
                    subpulse = list(collect.keys())     # different pulses contained here: 0,1,2
                    for key in subpulse:
                        action[key] =  collect.get(key) / 6    # Get the number of different pulses
            y.append(action)
        return np.array(y)

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

    # function is modified to fit the network
    def next_JammerState(self, state, aj, length):
        '''
        :param state: jammer state, especally for layer 2
        0: spot jammer; 1: barrage jammer; 2: no jammer
        :param aj: jammer action
        0,1,2: spot jammer; 3: barrage jammer; 4: no jammer
        :param length: current pulse
        :return: state
        '''
        state = np.copy(state)
        if aj == 4:
            state[2, :, 3*length:3*length+3] = 2
        elif aj == 3:
            state[2, :, 3*length:3*length+3] = 1
        else:
            state[1, aj, 3*length:3*length+3] = 1
        return state


    """
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
    """

# ---------------------------------- Utils -----------------------------------------
    # allow to change env id mannually
    def set_nid(self, id):
        self.nid = id

    def test_accuracy(self, nid, test_collector, test_size, loss_fn_name = "KLDivLoss"):
        batch = test_collector.buffer[range(test_size)]
        x = list()
        for i in range(len(batch.obs)):
            x.append(batch.obs[i].flatten())
        x = torch.FloatTensor(x).float()
        if self.net_type == 1: y = torch.tensor(self.get_jammer_act(batch.obs_next, self.observation_space.shape[2] / 3)).float()
        elif self.net_type == 2: 
            if loss_fn_name == "CrossEntropyLoss":
                y = torch.tensor(self.get_jammer_act(batch.obs_next, self.observation_space.shape[2] / 3)).long()
            elif loss_fn_name == "KLDivLoss":
                y = torch.tensor(self.get_jammer_act_distr(batch.obs_next, self.observation_space.shape[2] / 3)).long()
        y_pred = self.net_ensemble[nid](x)
        if self.net_type == 1:
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(y_pred, y)
        elif self.net_type == 2:
            if loss_fn_name == "CrossEntropyLoss":
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(y_pred, y)
            elif loss_fn_name == "KLDivLoss":
                loss_fn = torch.nn.KLDivLoss()
                loss = loss_fn(torch.log(y_pred), y.float())

        return loss.detach().numpy()

    def net_selection(self, collector, test_size):
        collector.reset_env()
        collector.reset_buffer()
        collector.collect(n_step=test_size)
        scores = []
        for nid in range(self.net_num):
            score = self.test_accuracy(nid, collector, test_size, "KLDivLoss")
            scores.append(score)
        scores = np.array(scores)
        prob_temp = np.sum(scores)/scores
        prob = prob_temp / np.sum(prob_temp)
        # print(scores)
        # print(prob)
        self.nid = np.random.choice(self.net_num, p = prob)
        print("We select net {:d}, its kl-divergence is: {:.4f}".format(self.nid, scores[self.nid]))
        
        return prob