import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
import os
import sys
import time
import random
from collections import deque
from flappy import FlappyBird
import preTrainedModel


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.numberOfActions = 2  # Flip or do nothing
        self.gamma = 0.99
        self.initEpsilon = 0.1
        self.finalEpsilon = 0.05
        self.numberOfIterations = 1500000
        self.replayMemorySize = 500000
        self.minibatchSize = 32
        self.explore = 2000000

        self.conv1 = nn.Conv2d(in_channels = 4, out_channels = 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512) # (3136,512)
        self.fc5 = nn.Linear(512, self.numberOfActions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #make sure input tensor is flattened
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)

def preProcess(image):
    imageData = cv2.cvtColor(cv2.resize(image,(84,84)), cv2.COLOR_BGR2GRAY)
    imageData = np.reshape(imageData,(84,84,1))
    imageTensor = imageData.transpose(2,0,1)
    imageTensor = imageTensor.astype(np.float32)
    imageTensor = torch.from_numpy(imageTensor)
    if torch.cuda.is_available():
        imageTensor = imageTensor.cuda()
    return imageTensor

def initWeights(net):
    if type(net) == nn.Conv2d or type(net) == nn.Linear:
        torch.nn.init.uniform(net.weight, -0.01, 0.01)
        net.bias.data.fill_(0.01)

def train(network,start):

    if torch.cuda.is_available():
        print("Using Cuda.....")
    else:
        print("CPUUUU !!!!!")

    optimizer = optim.Adam(network.parameters() , lr=1e-4)

    criterion = nn.MSELoss()

    game = FlappyBird()

    D = deque()

    action = torch.zeros([network.numberOfActions], dtype=torch.float32)
    action[0] = 0

    imageData, reward, terminal = game.run(action)
    imageData = preProcess(imageData)

    state = torch.cat((imageData,imageData,imageData,imageData)).unsqueeze(0)
    #print("State Shape: ",state.shape)

    epsilon = network.initEpsilon
    iteration = 0


    while iteration < network.numberOfIterations:
        # get output from the neural network
        output = network(state)[0]

        action = torch.zeros([network.numberOfActions], dtype=torch.float32 )
        if torch.cuda.is_available():
            action = action.cuda()

        random_action = random.random() <= epsilon
        if random_action:
            print("Performed Random Action!")

        # Pick index of highest value of neural network's output
        action_index = [torch.randint(network.numberOfActions, torch.Size([]), dtype=torch.int )
                        if random_action else torch.argmax(output)][0]

        if torch.cuda.is_available():
            action_index = action_index.cuda()

        # Activate that index
        action[action_index] = 1
        
        if epsilon > network.finalEpsilon:
            epsilon -= (network.initEpsilon - network.finalEpsilon) / network.explore

        imageData_1, reward, terminal = game.run(action)
        imageData_1 = preProcess(imageData_1)

        state_1 = torch.cat((state.squeeze(0)[1:, :, :], imageData_1)).unsqueeze(0)
        #print("State_1 Size : ", state_1.shape)

        action = action.unsqueeze(0)

        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        D.append((state, action, reward, state_1, terminal))

        if len(D) > network.replayMemorySize:
            D.popleft()

        minibatch = random.sample(D, min(len(D), network.minibatchSize))
        # unpack minibatch
        state_batch   = torch.cat(tuple(d[0] for d in minibatch))
        #print("state_batch size: ", state_batch.shape)
        action_batch  = torch.cat(tuple(d[1] for d in minibatch))
        #print("action_batch size: ", action_batch.shape)
        reward_batch  = torch.cat(tuple(d[2] for d in minibatch))
        #print("reward_batch size: ", reward_batch.shape)
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))
        #print("state_1_batch size: ", state_1_batch.shape)

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()
        
        # get output for the next state
        output_1_batch = network(state_1_batch)
        #print("output_1_batch: " , output_1_batch.shape)    x-2

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q) Target Q value Bellman equation.
        # gamma = discounted factors
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + network.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))


        # extract Q-value -----> column1 * column1 + column2 * column2
        # The main idea behind Q-learning is that if we had a function Q∗ :State × Action → ℝ
        #that could tell us what our return would be, if we were to take an action in a given state,
        #then we could easily construct a policy that maximizes our rewards
        q_value = torch.sum(network(state_batch) * action_batch, dim=1)
        #print("q_value: ", q_value.shape)   x
        #print("y_batch: ", y_batch.shape)	 x

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # calculate loss
        loss = criterion(q_value, y_batch)

        # do backward pass
        loss.backward()
        optimizer.step()

        # set state to be state_1
        state = state_1
        iteration += 1

        if iteration % 10000 == 0:
            torch.save(model, "trained_model/current_model_" + str(iteration) + ".pth")
              
        print("total iteration: {} Elapsed time: {:.2f} epsilon: {:.5f}"
               " action: {} Reward: {:.1f}".format(iteration,((time.time()-start)/60),epsilon,action_index.cpu().detach().numpy(),reward.numpy()[0][0]))


def main(mode):
    cuda_avaliable = torch.cuda.is_available()
    if mode == 'test':
        model = torch.load('trained_model/current_model_420000.pth', map_location='cpu').eval()
        test(model)
    elif mode == 'train':
        if not os.path.exists('trained_model/'):
            os.mkdir('trained_model/')
        model = NeuralNetwork()
        if cuda_avaliable:
            model = model.cuda()
        model.apply(initWeights)
        start = time.time()
        train(model, start)
    elif mode == 'continue':
        model = torch.load('trained_model/current_model_420000.pth', map_location='cpu').eval()
        start = time.time()
        train(model, start)


if __name__ == "__main__":
    main(sys.argv[1])
