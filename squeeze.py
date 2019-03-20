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
from torchvision.models import squeezenet1_0
import torch.nn as nn
from collections import OrderedDict


class flappyNet(nn.Module):
    def __init__(self):
        super(flappyNet,self).__init__()

        self.numberOfActions = 2  # Flip or do nothing
        self.gamma = 0.99
        self.initEpsilon = 0.1
        self.finalEpsilon = 0.05
        self.numberOfIterations = 1500000
        self.replayMemorySize = 50000
        self.minibatchSize = 32
        self.explore = 2000000

        model = squeezenet1_0(pretrained=True)

        self.features = nn.Sequential(*list(model.features.children())[:7])
        self.conv3 = nn.Conv3d(in_channels=4,out_channels=1,kernel_size=3, stride=1,padding=1)
        self.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(256*9*9, 512)), # 20736, 512
                          ('relu', nn.ReLU()), 
                          ('fc2', nn.Linear(512, 2))]))

        for param in self.features.parameters():
            param.require_grad = False
        

    def forward(self, x):
        x = self.features(x)
        #print("X Forward 1: ",x.shape)
        x = self.conv3(x.view(-1, 4, 256, 9, 9))
        x = x.squeeze(1)
        #print("X Forward 2: ",x.shape)
        x = self.classifier(x.view(x.size(0), -1))
        #print("X output: ",x.shape)
        return x

network = flappyNet()

def preProcess(image):
    imageData = cv2.resize(image,(84,84))
                                   # Shape 84 x 84 x 3
    imageData = imageData.transpose(2,0,1)        # Shape 3  x 84 x 84
    imageData = imageData.astype(np.float32)
    imageTensor = torch.from_numpy(imageData)
    return imageTensor


def train(network,start):

    optimizer = optim.Adam(network.parameters() , lr=1e-3)

    criterion = nn.MSELoss()

    game = FlappyBird()

    D = deque()

    action = torch.zeros([network.numberOfActions], dtype=torch.float32)
    action[0] = 0

    imageData, reward, terminal = game.run(action)
    imageData = preProcess(imageData)

    state = torch.stack([imageData,imageData,imageData,imageData])
    #print("State Shape: ",state.shape)#  4 x 3 x 84 x 84
    #print("Input Shape: ",state.shape)

    epsilon = network.initEpsilon
    iteration = 0
    graphData = []
    reward_counter = 0


    while iteration < network.numberOfIterations:

        #print("Network iteration : 1.{}".format(iteration))
        # get output from the neural network
        output = network(state)[0]

        action = torch.zeros([network.numberOfActions], dtype=torch.float32 )

        random_action = random.random() <= epsilon
        if random_action:
            print("Performed Random Action!")

        # Pick index of highest value of neural network's output
        action_index = [torch.randint(network.numberOfActions, torch.Size([]), dtype=torch.int )
                        if random_action else torch.argmax(output)][0]

        # Activate that index
        action[action_index] = 1
        
        if epsilon > network.finalEpsilon:
            epsilon -= (network.initEpsilon - network.finalEpsilon) / network.explore

        imageData_1, reward, terminal = game.run(action)
        imageData_1 = preProcess(imageData_1)

        if not reward == -1:
            reward_counter += reward

        state_1 = torch.cat((state[1:],imageData_1.unsqueeze(0) ), dim=0)
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
        
        # get output for the next state
        output_1_batch = network(state_1_batch)
        #print("Network iteration : 2.{}".format(iteration))
        

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

        #   print("Last State Shape: ", state.shape)


        if iteration % 3000 == 0 and reward != -1:

            graphData = [iteration, reward_counter]

            with open('data_graph.csv', mode='a') as csv_file:
                
                writer = csv.writer(csv_file)
                writer.writerow(graphData)



        if iteration % 20000 == 0:
            torch.save(model, "trained_model/current_model_" + str(iteration) + ".pth")
              
        print("total iteration: {} Elapsed time: {:.2f} epsilon: {:.5f}"
               " action: {} Reward: {:.1f}".format(iteration,((time.time()-start)/60),epsilon,action_index.cpu().detach().numpy(),reward.numpy()[0][0]))


def main(mode):
    if mode == 'test':
        model = torch.load('trained_model/current_model_420000.pth', map_location='cpu').eval()
        test(model)

    if mode == 'train':
        if not os.path.exists('trained_model/'):
            os.mkdir('trained_model/')
        #model.apply(initWeights)
        start = time.time()
        train(network, start)



if __name__ == "__main__":
#    main(sys.argv[1])

    cuda_avaliable = torch.cuda.is_available()

    if not os.path.exists('trained_model/'):
        os.mkdir('trained_model/')


    if cuda_avaliable:
        network = network.cuda()

    #model.apply(initWeights)
    start = time.time()
    train(network, start)
    