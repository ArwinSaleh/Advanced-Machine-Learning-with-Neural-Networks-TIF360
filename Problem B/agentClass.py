from keras.backend import conv2d
from keras.optimizer_v1 import Adam
import numpy as np
import itertools
from matplotlib import pyplot as plt
from collections import namedtuple
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from numpy.lib.shape_base import tile

def binatodeci(binary):
    deci = sum(val*(2**idx) for idx, val in enumerate(reversed(binary)))
    return int(deci)

class TQAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.episode=0
        self.episode_count=episode_count

    def fn_init(self,gameboard):
        self.gameboard=gameboard

        self.actions        = []
        action_perm         = []
        
        N_ACTION_ORIENTATIONS   = 4                # tile can rotate to 4 different orientations
        N_ACTION_POSITIONS      = gameboard.N_col  # len(gameboard.N_col) possible positions


        self.current_state = np.zeros((self.gameboard.N_row * self.gameboard.N_col + len(gameboard.tiles), ))



        action_perm.append(range(0, N_ACTION_ORIENTATIONS))
        action_perm.append(range(0, N_ACTION_POSITIONS))

        for i in itertools.product(*action_perm):
            self.actions.append(i)      # (action1, action2)

        self.actions = np.array(self.actions)

        self.Q_table = np.zeros((2**(self.gameboard.N_row * self.gameboard.N_col + len(gameboard.tiles)), len(self.actions)))
        self.reward_tots = np.zeros((self.episode_count, ))

    def fn_load_strategy(self,strategy_file):
        self.Q_table = strategy_file

    def fn_read_state(self):

        current_board = np.ndarray.flatten(self.gameboard.board)
        current_tile = self.gameboard.cur_tile_type

        current_tiles = []

        for i in range(len(self.gameboard.tiles)):
            if i == current_tile:
                current_tiles.append(1)
            else:
                current_tiles.append(-1)

        self.current_state[:len(self.gameboard.tiles)] = current_tiles
        self.current_state[len(self.gameboard.tiles):] = current_board

        binary_rep_state = np.where(self.current_state == -1, 0, self.current_state)
        self.current_state_idx = binatodeci(binary_rep_state)

    def fn_select_action(self):

        binary_rep_state = np.where(self.current_state == -1, 0, self.current_state)
        index = binatodeci(binary_rep_state)
    
        self.current_action_idx = None

        r = np.random.uniform(0, 1)

        done = False

        if r < self.epsilon:
            while(not done):
                self.current_action_idx = np.random.randint(0, len(self.actions))
                move = self.gameboard.fn_move(self.actions[self.current_action_idx][0], self.actions[self.current_action_idx][1])
                if move == 0:
                    done = True
        else:
            while(not done):
                self.current_action_idx = np.where(self.Q_table[index, :] == np.max(self.Q_table[index, :]))[0]
                #print(np.where(self.Q_table[index, :] == np.max(self.Q_table[index, :]))[0])
                #print("WHAT")
                if len(self.current_action_idx) > 1:
                    self.current_action_idx = self.current_action_idx[np.random.randint(0, len(self.current_action_idx))]
                else:
                    self.current_action_idx = self.current_action_idx[0]

                move = self.gameboard.fn_move(self.actions[self.current_action_idx][0], self.actions[self.current_action_idx][1])

                if move == 1:
                    self.Q_table[index, self.current_action_idx] = - np.inf
                else:
                    done = True
                    
    
    def fn_reinforce(self,old_state,reward):

        old_action = self.current_action_idx

        self.Q_table[old_state, old_action] = self.Q_table[old_state, old_action] + self.alpha * (reward + np.max(self.Q_table[self.current_state_idx, :]) - self.Q_table[old_state, old_action])

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1
            if self.episode%100==0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(round(np.sum(self.reward_tots[range(self.episode-100,self.episode)] / 100), 2)),')')
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-table to data files for plotting of the rewards and the Q-table can be used to test how the agent plays
                    #np.savetxt("Q_table_episode_" + str(self.episode) + ".csv", self.Q_table, delimiter=",")
                    np.savetxt("Rewards.csv", self.reward_tots)
            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later passed to fn_reinforce()


            old_state = self.current_state_idx

            # Drop the tile on the game board
            reward=self.gameboard.fn_drop()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later

            self.reward_tots[self.episode] += reward


            # Read the new state
            self.fn_read_state()
            # Update the Q-table using the old state and the reward (the new state and the taken action should be stored as attributes in self)
            self.fn_reinforce(old_state,reward)

"""
class DeepQNet(nn.Module):
            def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, N_actions):
                super(DeepQNet, self).__init__()
                self.learning_rate = learning_rate
                self.input_dims = input_dims
                self.fc1_dims = fc1_dims
                self.fc2_dims = fc2_dims
                self.N_actions = N_actions
                self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
                self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
                self.fc3 = nn.Linear(self.fc2_dims, self.N_actions)
                self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
                self.loss = nn.MSELoss()
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            def forward(self, state):
                x = F.relu(self.fc1(state))
                x = F.relu(self.fc2(x))
                actions = self.fc3(x)
                return actions
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from collections import deque

class DQN:
    def __init__(self, input_shape, output_shape, REPLAY_MEM_SIZE):
        self.model = self.create_model(input_shape, output_shape)

        self.target_model = self.create_model(input_shape, output_shape)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEM_SIZE)

        self.target_update_counter = 0

    def create_model(self, input_shape, output_shape):
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(output_shape, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_Qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]
    
    def train(self, terminal_state, batch_size, DISCOUNT, EP_SYNC):
        batch = random.sample(self.replay_memory, batch_size)
        
        current_states = np.array([transition[0] for transition in batch])
        current_Qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in batch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []
        
        for index, (current_state, action, reward, new_current_state, done) in enumerate(batch):
            if not done:
                max_future_Q = np.max(future_qs_list[index])
                new_Q = reward + DISCOUNT * max_future_Q
            else:
                new_Q = reward

            current_Qs = current_Qs_list[index]
            current_Qs[action] = new_Q

            X.append(current_state)
            y.append(current_Qs)

        self.model.fit(np.array(X), np.array(y), batch_size=batch_size, verbose=0, shuffle=False if terminal_state else None)

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > EP_SYNC:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
        
class TDQNAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,epsilon_scale,replay_buffer_size,batch_size,sync_target_episode_count,episode_count):
        
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.epsilon_scale=epsilon_scale
        self.replay_buffer_size=replay_buffer_size
        self.batch_size=batch_size
        self.sync_target_episode_count=sync_target_episode_count
        self.episode=0
        self.episode_count=episode_count

    def fn_init(self,gameboard):
        self.gameboard=gameboard
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could set up and initialize the states, actions, the Q-networks (one for calculating actions and one target network), experience replay buffer and storage for the rewards
        # You can use any framework for constructing the networks, for example pytorch or tensorflow
        # This function should not return a value, store Q network etc as attributes of self

        # Useful variables: 
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.alpha' the learning rate for stochastic gradient descent
        # 'self.episode_count' the total number of episodes in the training
        # 'self.replay_buffer_size' the number of quadruplets stored in the experience replay buffer

        self.actions            = []
        action_perm             = []
        N_ACTION_ORIENTATIONS   = 4                # tile can rotate to 4 different orientations
        N_ACTION_POSITIONS      = gameboard.N_col  # len(gameboard.N_col) possible positions
        action_perm.append(range(0, N_ACTION_ORIENTATIONS))
        action_perm.append(range(0, N_ACTION_POSITIONS))
        for i in itertools.product(*action_perm):
            self.actions.append(i)      # (action1, action2)
        self.actions = np.array(self.actions)

        self.DeepQnetwork = DQN(input_shape=(len(gameboard.tiles), gameboard.N_row, gameboard.N_col), output_shape=len(self.actions), REPLAY_MEM_SIZE=self.replay_buffer_size)

        self.reward_tots = np.zeros((self.episode_count, ))

    def fn_load_strategy(self,strategy_file):
        pass
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-network (to Q-network of self) from the strategy_file
        

    def fn_read_state(self):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could calculate the current state of the gane board
        # You can for example represent the state as a copy of the game board and the identifier of the current tile
        # This function should not return a value, store the state as an attribute of self

        # Useful variables: 
        # 'self.gameboard.N_row' number of rows in gameboard
        # 'self.gameboard.N_col' number of columns in gameboard
        # 'self.gameboard.board[index_row,index_col]' table indicating if row 'index_row' and column 'index_col' is occupied (+1) or free (-1)
        # 'self.gameboard.cur_tile_type' identifier of the current tile that should be placed on the game board (integer between 0 and len(self.gameboard.tiles))

        current_board = np.ndarray.flatten(self.gameboard.board)

        tiles = np.zeros((len(self.gameboard.tiles,)), dtype=np.int64)
        tiles[tiles == 0] = -1
        tiles[self.gameboard.cur_tile_type] = 1

        self.current_state = np.concatenate((current_board, tiles), axis=None)

    def fn_select_action(self):
        
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Choose and execute an action, based on the output of the Q-network for the current state, or random if epsilon greedy
        # This function should not return a value, store the action as an attribute of self and exectute the action by moving the tile to the desired position and orientation

        # Useful variables: 
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy
        # 'self.epsilon_scale' parameter for the scale of the episode number where epsilon_N changes from unity to epsilon

        # Useful functions
        # 'self.gameboard.fn_move(tile_x,tile_orientation)' use this function to execute the selected action
        # The input argument 'tile_x' contains the column of the tile (0 < tile_x < self.gameboard.N_col)
        # The input argument 'tile_orientation' contains the number of 90 degree rotations of the tile (0 < tile_orientation < # of non-degenerate rotations)
        # The function returns 1 if the action is not valid and 0 otherwise
        # You can use this function to map out which actions are valid or not

        self.current_action_idx = None

        r = np.random.uniform(0, 1)

        done = False
        
        epsilon_E = max(self.epsilon, 1 - self.episode / self.epsilon_scale)

        if r < epsilon_E:
            while(not done):
                self.current_action_idx = np.random.randint(0, len(self.actions))
                move = self.gameboard.fn_move(self.actions[self.current_action_idx][0], self.actions[self.current_action_idx][1])
                if move == 0:
                    done = True
        else:
            while(not done):
                self.current_action_idx = np.argmax(self.DeepQnetwork.get_Qs(self.current_state))
                move = self.gameboard.fn_move(self.actions[self.current_action_idx][0], self.actions[self.current_action_idx][1])

                if move == 1:
                    #self.Q_net[tensor_max] = torch.tensor(- 999)
                    while(not done):
                        self.current_action_idx = np.random.randint(0, len(self.actions))
                        move = self.gameboard.fn_move(self.actions[self.current_action_idx][0], self.actions[self.current_action_idx][1])
                        if move == 0:
                            done = True
                else:
                    done = True


    def fn_reinforce(self,batch):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q network using a batch of quadruplets (old state, last action, last reward, new state)
        # Calculate the loss function by first, for each old state, use the Q-network to calculate the values Q(s_old,a), i.e. the estimate of the future reward for all actions a
        # Then repeat for the target network to calculate the value \hat Q(s_new,a) of the new state (use \hat Q=0 if the new state is terminal)
        # This function should not return a value, the Q table is stored as an attribute of self

        # Useful variables: 
        # The input argument 'batch' contains a sample of quadruplets used to update the Q-network

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        self.DeepQnetwork.train(terminal_state=True, batch_size=batch, DISCOUNT=self.alpha, EP_SYNC=self.sync_target_episode_count)

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1
            if self.episode%100==0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.sum(self.reward_tots[range(self.episode-100,self.episode)])),')')
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    pass
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-network to data files
            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later stored in the ecperience replay buffer
            old_state = self.current_state

            # Drop the tile on the game board
            reward=self.gameboard.fn_drop()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later
            self.reward_tots[self.episode] += reward

            # Read the new state
            self.fn_read_state()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to store the state in the experience replay buffer
            self.DeepQnetwork.update_replay_memory((old_state, self.current_action_idx, reward, self.current_state))

            if len(self.DeepQnetwork.replay_memory) >= self.replay_buffer_size:
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to create a variable 'batch' containing 'self.batch_size' quadruplets 
                batch = random.sample(self.replay)

                self.fn_reinforce(batch)

                if self.episode_count % self.sync_target_episode_count == 0:
                    # TO BE COMPLETED BY STUDENT
                    # Here you should write line(s) to copy the current network to the target network
                    self.Q_target.load_state_dict(self.Q_net.state_dict())

class THumanAgent:
    def fn_init(self,gameboard):
        self.episode=0
        self.reward_tots=[0]
        self.gameboard=gameboard

    def fn_read_state(self):
        pass

    def fn_turn(self,pygame):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)
            if event.type==pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.reward_tots=[0]
                    self.gameboard.fn_restart()
                if not self.gameboard.gameover:
                    if event.key == pygame.K_UP:
                        self.gameboard.fn_move(self.gameboard.tile_x,(self.gameboard.tile_orientation+1)%len(self.gameboard.tiles[self.gameboard.cur_tile_type]))
                    if event.key == pygame.K_LEFT:
                        self.gameboard.fn_move(self.gameboard.tile_x-1,self.gameboard.tile_orientation)
                    if event.key == pygame.K_RIGHT:
                        self.gameboard.fn_move(self.gameboard.tile_x+1,self.gameboard.tile_orientation)
                    if (event.key == pygame.K_DOWN) or (event.key == pygame.K_SPACE):
                        self.reward_tots[self.episode]+=self.gameboard.fn_drop()
