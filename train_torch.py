import torch #  A library for deep learning. It provides tensor computations (like NumPy) with GPU acceleration.
import torch.nn as nn # Contains neural network layers and utilities.
import torch.nn.functional as F # Provides functions for neural network operations, like activation functions.
import torch.optim as optim # Contains optimization algorithms (like Adam).
import numpy as np #  For numerical operations
import collections # For dequeue for the memory buffer
import random # For random sampling.
import dill as pickle # For storing the buffer state (used for serializing (saving) Python objects.)

# Determines if a GPU is available and sets the device accordingly (GPU or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# This class manages the experience replay buffer, which stores experiences (state, action, reward, next state, terminal) during training.
class MemoryBuffer(object):
    def __init__(self, max_size):
        self.memory_size = max_size # The maximum number of experiences the buffer can hold. When this limit is reached, older experiences are discarded to make room for new ones.
        self.trans_counter=0 # num of transitions in the memory this count is required to delay learning until the buffer is sensibly full (used to determine when the buffer has enough data to start the learning process)
        self.index=0        # current pointer in the buffer
        self.buffer = collections.deque(maxlen=self.memory_size) # A deque (double-ended queue) that holds the experiences. It has a maximum length (maxlen=self.memory_size), so old experiences are automatically removed when the buffer is full.
        self.transition = collections.namedtuple("Transition", field_names=["state", "action", "reward", "new_state", "terminal"]) #  A named tuple to structure each experience.

    # Saves a transition (experience) into the buffer.
    def save(self, state, action, reward, new_state, terminal):
        t = self.transition(state, action, reward, new_state, terminal)
        self.buffer.append(t)
        self.trans_counter = (self.trans_counter + 1) % self.memory_size

    # Samples a random batch of experiences from the buffer for training.
    def random_sample(self, batch_size):
        assert len(self.buffer) >= batch_size # should begin sampling only when sufficiently full
        transitions = random.sample(self.buffer, k=batch_size) # number of transitions to sample
        states = torch.from_numpy(np.vstack([e.state for e in transitions if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in transitions if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in transitions if e is not None])).float().to(device)
        new_states = torch.from_numpy(np.vstack([e.new_state for e in transitions if e is not None])).float().to(device)
        terminals = torch.from_numpy(np.vstack([e.terminal for e in transitions if e is not None]).astype(np.uint8)).float().to(device)
  
        return states, actions, rewards, new_states, terminals

# This is a simple feedforward neural network representing the Q-network.
class QNN(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128) # fc1, fc2, fc3: Fully connected layers.
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    # Defines the forward pass through the network. The final layer does not have an activation function because it outputs Q-values directly.     
    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)

# This class handles the RL process.
class Agent(object):
    # gamma: Discount factor for future rewards.
    # epsilon: Exploration rate (for ε-greedy policy).
    # epsilon_dec: Decrement factor for epsilon.
    # epsilon_min: Minimum value for epsilon.
    # batch_size: Number of experiences sampled during learning.
    # memory: An instance of MemoryBuffer.
    def __init__(self, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
                 epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000):
        self.gamma = gamma # alpha = learn rate, gamma = discount
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec # decrement of epsilon for larger spaces
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.memory = MemoryBuffer(mem_size)

    # Stores a transition in memory.
    def save(self, state, action, reward, new_state, done):
        # self.memory.trans_counter += 1
        self.memory.save(state, action, reward, new_state, done)

    # Chooses an action based on the current policy (either exploiting the Q-network or exploring randomly).
    def choose_action(self, state):
        # state = state[np.newaxis, :]
        rand = np.random.random()
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.q_func.eval()
        with torch.no_grad():
            action_values = self.q_func(state)
        self.q_func.train()
        # print(state)
        if rand > self.epsilon: 
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # exploring: return a random action
            return np.random.choice([i for i in range(4)])     

    # Reduces epsilon to decrease exploration over time.
    def reduce_epsilon(self):
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                       self.epsilon_min else self.epsilon_min  

    # Placeholder for the learning process (implemented in DoubleQAgent).    
    def learn(self):
        raise Exception("Not implemented")
    
    # Save/load the model's parameters.
    def save_model(self, path):
        torch.save(self.q_func.state_dict(), path)

    def load_saved_model(self, path):
        self.q_func = QNN(8, 4, 42).to(device)
        self.q_func.load_state_dict(torch.load(path))
        self.q_func.eval()
        
# This extends the Agent class to implement Double DQN, which helps mitigate the overestimation bias in Q-learning.
# replace_q_target: Number of steps after which the target network is updated
class DoubleQAgent(Agent):
    def __init__(self, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
                 epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000, replace_q_target = 100):
        
        super().__init__(lr=lr, gamma=gamma, epsilon=epsilon, batch_size=batch_size,
             epsilon_dec=epsilon_dec,  epsilon_end=epsilon_end,
             mem_size=mem_size)

        self.replace_q_target = replace_q_target
        self.q_func = QNN(8, 4, 42).to(device)
        self.q_func_target = QNN(8, 4, 42).to(device)
        self.optimizer = optim.Adam(self.q_func.parameters(), lr=lr)
        
    # Implements the Double DQN learning process:
    def learn(self):
        if self.memory.trans_counter < self.batch_size: # wait before you start learning
            return
            
        # 1. Choose a sample from past transitions:     # Sample a batch of experiences from memory.
        states, actions, rewards, new_states, terminals = self.memory.random_sample(self.batch_size)
        
        # 2. Update the target values    # Compute the Q-values for the next states using the target network.
        q_next = self.q_func_target(new_states).detach().max(1)[0].unsqueeze(1)
        q_updated = rewards + self.gamma * q_next * (1 - terminals)
        q = self.q_func(states).gather(1, actions)
        
        # 3. Update the main NN    # Update the Q-values of the current network using the Bellman equation.
        loss = F.mse_loss(q, q_updated)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 4. Update the target NN (every N-th step)    # Update the target network after a fixed number of steps.
        if self.memory.trans_counter % self.replace_q_target == 0: # wait before you start learning
            for target_param, local_param in zip(self.q_func_target.parameters(), self.q_func.parameters()):
                target_param.data.copy_(local_param.data)
                
        # 5. Reduce the exploration rate
        self.reduce_epsilon()


    # Extended to save/load both the main Q-network and the target network.
    def save_model(self, path):
        super().save_model(path)
        torch.save(self.q_func.state_dict(), path+'.target')


    def load_saved_model(self, path):
        super().load_saved_model(path)
        self.q_func_target = QNN(8, 4, 42).to(device)
        self.q_func_target.load_state_dict(torch.load(path+'.target'))
        self.q_func_target.eval()
    