import os

import torch
import random
import numpy as np
from collections import deque
from environment_binary import *
from helper import plot
from model import QTrainer, ConvModel
import argparse
from multiprocessing import freeze_support

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='model-binary-1024-4.pth')
parser.add_argument('--dataset', default='BDD10K_Binary')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--imgsize', type=int, default=2048)
parser.add_argument('--nregions', type=int, default=4)
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_iterations = 0
        self.epsilon = 0 #randomness
        self.gamma = 0.9 #discount rate
        self.memory = deque(maxlen = MAX_MEMORY)
        self.model = ConvModel(nregions=4).to(device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, env):
        return env.get_state()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #pop left if max memory is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        #print(f"states:{states}, actions:{actions}, next_states:{states}, dones:{dones}")
        #print(actions)
        #print("Actions:", actions)

        self.trainer.train_step_long(states, actions, rewards, next_states, dones)



    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = max(20, 80 - self.n_iterations)
        final_move = -1
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0,15) #16 regions
            final_move = move
        else:
            state0 = torch.tensor(state, dtype=torch.float, device=device)
            with torch.no_grad():
                prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            #print("pred:", prediction)
            #print("move:", move)
            final_move  = move
        return final_move

def plot_scores_and_mean(plot_scores, plot_mean_scores):
    plt.plot(plot_scores, label='Score')
    plt.plot(plot_mean_scores, label='Mean Score')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.title('Score vs. Iterations')
    plt.legend()
    plt.show()

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    image_code = random.choice(os.listdir('/Users/eliandromelo/Downloads/bdd100k/images/10k/train'))
    mask_code = image_code.replace("jpg", "png")
    image_path = '/Users/eliandromelo/Downloads/bdd100k/images/10k/train/' + image_code
    mask_path = '/Users/eliandromelo/Downloads/bdd100k/labels/sem_seg/masks/train/' + mask_code

    image = resize(imread(image_path), (1024, 1024))
    mask = resize(imread(mask_path), (1024, 1024))
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    mask = torch.tensor(mask)
    env = Environment(args.model, nregions=args.nregions, device= device)
    env.reset(image, mask)
    while True:
        #get old state
        state_old = agent.get_state(env)

        #get move
        final_move = agent.get_action(state_old)

        #perform move to get new state
        reward, done, score = env.step(final_move)
        state_new = agent.get_state(env)

        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            env.reset(image, mask)
            agent.n_iterations += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # agent.model.save()
            print('Episode', agent.n_iterations, 'Score', score.item(), 'Record:', record.item())

            plot_scores.append(score.item())
            total_score += score.item()
            mean_score = total_score / agent.n_iterations
            plot_mean_scores.append(mean_score)
            if agent.n_iterations % 100 == 0:
                plot(plot_scores, plot_mean_scores, agent.n_iterations)



if __name__ == '__main__':
    from skimage.io import imread
    from skimage.transform import resize


    env = Environment(args.model, 4, device)

    train()