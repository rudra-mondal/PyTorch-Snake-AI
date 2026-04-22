# model.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Optimization: Creating a tensor from a tuple of numpy arrays is extremely slow
        # We first convert the tuple to a single numpy array before passing it to torch.tensor
        # Expected performance impact: ~8x faster batch processing in train_long_memory
        if isinstance(state, tuple):
            state = torch.tensor(np.array(state), dtype=torch.float)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float)
            action = torch.tensor(np.array(action), dtype=torch.long)
            reward = torch.tensor(np.array(reward), dtype=torch.float)
        else:
            state = torch.tensor(state, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
            action = torch.tensor(action, dtype=torch.long)
            reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # Vectorized implementation for performance
        done_tensor = torch.tensor(done, dtype=torch.bool)
        next_pred = self.model(next_state)
        max_next_pred = torch.max(next_pred, dim=1)[0]

        Q_new = torch.where(done_tensor, reward, reward + self.gamma * max_next_pred)
        action_indices = torch.argmax(action, dim=1)
        target[torch.arange(len(done)), action_indices] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()