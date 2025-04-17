import gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 1e-3
MEMORY_SIZE = 10000
BATCH_SIZE = 64
TARGET_UPDATE = 10
EPISODES = 1000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

env = gym.make("FrozenLake-v1", is_slippery=True)

# Convert discrete state to one-hot encoding
def one_hot(state, state_size):
    vec = np.zeros(state_size)
    vec[state] = 1.0
    return vec

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.fc(x)

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START

        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

        self.update_target_network()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        current_q = self.policy_net(states).gather(1, actions)
        next_actions = torch.argmax(self.policy_net(next_states), dim=1, keepdim=True)
        next_q = self.target_net(next_states).gather(1, next_actions).detach()

        expected_q = rewards + (1 - dones) * GAMMA * next_q
        loss = self.loss_fn(current_q, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

# Main training loop
state_size = env.observation_space.n
action_size = env.action_space.n
agent = Agent(state_size, action_size)

for episode in range(EPISODES):
    state = env.reset()[0]
    state = one_hot(state, state_size)
    total_reward = 0

    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state_onehot = one_hot(next_state, state_size)

        agent.remember(state, action, reward, next_state_onehot, done)
        agent.replay()

        state = next_state_onehot
        total_reward += reward

    if episode % TARGET_UPDATE == 0:
        agent.update_target_network()

    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

print("Training complete.")


def evaluate(agent, episodes=100):
    success = 0
    for _ in range(episodes):
        state = env.reset()[0]
        state = one_hot(state, state_size)
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, _, _ = env.step(action)
            state = one_hot(state, state_size)
        if reward == 1.0:
            success += 1
    print(f"Success rate: {success}/{episodes} = {success / episodes * 100:.2f}%")

evaluate(agent)