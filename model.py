import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
#from torchviz import make_dot

import torch.nn.functional as F


class ConvModel(nn.Module):
    def __init__(self, nregions):
        super(ConvModel, self).__init__()

        self.nregions = nregions

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=1, stride=2)

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=1, stride=2)

        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=1, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Adjust the size of fully connected layers based on the new spatial size
        fc_input_size = 16
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 16)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.global_avg_pool(x)

        x = x.view(-1, 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        #print(len(state.shape))
        #print("shape", state.shape)
        if len(state.shape) == 4:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        state = state.squeeze()
        pred = self.model(state)
        pred = torch.sigmoid(pred)
        target = pred.clone()

        #done = torch.unsqueeze(done, 0)
        #print(reward)
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # next_state = torch.randn(1, 3, 2048, 2048)
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # pred[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

    def train_step_long(self, state, action, reward, next_state, done):
        state = torch.stack(state)
        next_state = torch.stack(next_state)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # print(len(state.shape))
        """if len(state.shape) == 0:
            #(1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )"""

        # 1: predicted Q values with current state
        # state = state[0]
        state = state.squeeze(1)
        pred = self.model(state)
        target = pred.clone()
        max_action_index = torch.argmax(reward).item()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            # target = target.squeeze(0)
            #print(action)

            target[idx][action[max_action_index]] = Q_new
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # pred[torch.argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target.detach(), pred)
        loss.backward()

        self.optimizer.step()

'''
# Create an instance of ConvModel
model = ConvModel()

# Create a dummy input
dummy_input = torch.randn((1, 2048, 8, 8))  # Adjust the shape based on your actual input size

# Generate the computational graph
output = model(dummy_input)
graph = make_dot(output, params=dict(model.named_parameters()))

# Save or display the graph
graph.render("conv_model_graph")  # Save the graph as "conv_model_graph.pdf"
graph.view()
'''
