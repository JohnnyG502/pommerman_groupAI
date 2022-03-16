"""
In this module the previously generated data will be used to predict the position of ones
teammate and the enemies. It consists of all all relevant classes to preprocess and learn 
the position prediction net. 

"""

import numpy as np
import pandas as pd
import time, random, math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split

# modules
from communication_protokoll import CommunicationProtocol, PositionDefinition
from utils import *

import os


class PositionPrediction(nn.Module):
    """ 
    Convolutional neural net for position prediction.
    """

    def __init__(self, input_dim, batch_size):
        super().__init__()

        self.input_dim = input_dim
        
        self.cnn1 = nn.Conv2d(input_dim, 15, 1, stride=1, padding=0)
        self.cnn2 = nn.Conv2d(15, 13, 1, stride=1, padding=0)
        self.cnn3 = nn.Conv2d(13, 9, 1, stride=1, padding=0) #1
        self.cnn4 = nn.Conv2d(9, 4, 1, stride=1, padding=0)
        self.cnn5 = nn.Conv2d(15, 7, 2, stride=1, padding=0)
        self.cnn6 = nn.Conv2d(11, 4, 2, stride=1)
        self.cnn7 = nn.Conv2d(4, 4, 1, stride=1, padding=0)
        self.l1 = nn.Linear(11*11, 11*11)
        
        
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=2)

        self.dropout = nn.Dropout()
        
        self.batch_size = batch_size

    def forward(self, obs, last_obs):
        #concate = torch.cat(obs, last_obs, 1)

        out = self.relu(self.cnn1(obs))
        out = self.dropout(out)
        out = self.relu(self.cnn2(out))
        out = self.relu(self.cnn3(out))
        out = self.dropout(out)
        out = self.relu(self.cnn4(out))
        #out = self.relu(self.cnn5(out))
        #out = self.relu(self.cnn6(out))

        out = out.view(self.batch_size, 4, 121, 1)
        out = self.soft(self.cnn7(out))
        #out = self.sig(self.l1(out))
        #print(obs.shape)
        #out = o
        out = out.view(self.batch_size, 4, 11, 11)

        return out

class Training():
    """
    Training and evaluation class which calculates the loss and learns the net
    """
    def __init__(self, epochs, batch_size, optimizer, criterion, model, device):
        self.optimizer, self.criterion, self.model = optimizer, criterion, model
        self.epochs, self.batch_size = epochs, batch_size
        
        self.best_train_loss = 5

        self.device = device

        self.last_inp = torch.zeros(self.batch_size, 4, 11, 11)
        self.last_label = torch.zeros(self.batch_size, 4, 11, 11)
        
    def evaluate(self, iterator):
        self.model.eval()
        
        epoch_loss = 0
        
        with torch.no_grad():
            for batch, input in enumerate(iterator):
                self.optimizer.zero_grad()
                output = self.model(input)
                
                loss = self.criterion(output, target)
                epoch_loss += loss.item()
            
        test_loss = epoch_loss / len(iterator)
        print(f'| Test Loss: {test_loss:.3f} ')
        
    def train(self, iterator, epoch, path):
        self.model.train()
        
        epoch_loss = 0

        for batch, (inp, label) in enumerate(iterator):
            self.optimizer.zero_grad()
            plane_dim = inp.shape[2]
            inp = inp.reshape(self.batch_size, plane_dim, 11, 11)
            label = label.reshape(self.batch_size, 4, 11, 11)
            #. label = label.reshape(self.batch_size, 4, 121, 1)
            #self.last_inp = inp.clone().to(self.device)
            #self.last_label = label.clone().to(self.device)
            inp = inp.to(self.device)
            label = label.to(self.device)

            #print(inp, inp.shape)

            #print(label.shape)

            

            output = self.model(inp, None)

            output_2 = output[:, 3, :, :]
            label_2= label[:, 3, :, :]

            loss = self.criterion(output_2, label_2)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()

            if batch == len(iterator) - 1:
                #for index, plane in enumerate(output[0].detach()):
                #    for i, row in enumerate(plane):
                #        for j, col in enumerate(row):
                #            if col < 0.5:
                #                output[0][index, i, j] = 0
                #            else:
                #                output[0][index, i, j] = 0
                out_test = output.view(self.batch_size, 4, 121, -1).detach()
                test = torch.zeros(self.batch_size, 4, 121, 1)
                max_val = torch.argmax(out_test, dim=2)
                for i, val in enumerate(max_val[0]):
                    test[0, i, val, 0] = 1
                test = test.reshape((self.batch_size, 4, 11, 11))
                label = label.view(self.batch_size, 4, 121, -1).detach()
                label_val = torch.argmax(label, dim=2)
                label_t = torch.zeros(self.batch_size, 4, 121, 1)
                for i, val in enumerate(label_val[0]):
                    label_t[0, i, val, 0] = 1
                label_t = label_t.reshape((self.batch_size, 4, 11, 11))

                x = []
                y = []

                for i in range(4):
                    for j in range(11):
                        for z in range(11):
                            if test[0, i, j, z] == 1:
                                x.append((i,j,z))
                            if label_t[0, i, j, z] == 1:
                                y.append((i,j,z))
#
                print(x, "\n", y)

                print(output_2[0], label_2[0])


            #print(output, output.shape)


            #if epoch_loss/len(iterator) < self.best_train_loss:
            #    path = os.path.join(path, "model.pt")
            #    torch.save({
            #                'model_state_dict': self.model.state_dict(),
            #                # TODO: not sure if self.optimizer.state_dict is the same thing
            #                # that timour also saves (Copied from timour)
            #                'optimizer_state_dict': self.optimizer.state_dict(),
            #                # TODO: not sure if 'iterator_state' is available here
            #                # (Copied from timour)
            #                #'iterator_state': iterator.sampler.get_state()
            #                }, path)
            
        return epoch_loss / len(iterator)
    
    def train_setup(self, iterator, path):
        for epoch in range(self.epochs):
            
            start_time = time.time()
            
            train_loss = self.train(iterator, epoch, path)
            
            end_time = time.time()
            
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            

            
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            
    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    

class Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = np.array(data)
        self.label = np.array(labels)
        self.transforms = transforms.Compose([transforms.ToTensor(),transforms.ConvertImageDtype(torch.float)])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transforms(self.data[idx]), self.transforms(self.label[idx])

def labelToBinLabel(label):
    planes = np.zeros((4, 11, 11))
    it = np.nditer(label, flags=["multi_index"])
    for obs_val in it:
        if obs_val == 10: planes[0][it.multi_index] = 1
        elif obs_val == 11:  planes[1][it.multi_index] = 1
        elif obs_val == 12:  planes[2][it.multi_index] = 1
        elif obs_val == 13:  planes[3][it.multi_index] = 1
    return planes

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
    
    
def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    np.random.seed(42)
    random.seed(42)

    
    param = {
            "epochs": 200,
            "batch_size": 15,
            "l_rate": 0.001,
            "path": "./saved_models/checkpoints",
            "test_size": 0.7
            }

    transformator = PositionDefinition()
    translator = CommunicationProtocol()

    data = np.load("data_position_prediction.npy", allow_pickle=True)

    plane_obj = obsToPlanes(11)

    inp = []
    labels = []

    for obs in data:
        quadrant = translator.messageToPosition(obs[-1])
        msg_planes = transformator.QuadrantToObeservationArr(quadrant)
        obs_arr = obs[0].reshape(1, 11, 11)
        temp_inp = np.concatenate((obs_arr, msg_planes, obs[2].reshape(1, 11, 11)), axis=0)
        #last_input.append(obs[1])
        inp.append(temp_inp)
        labels.append(obs[-2])
    inp = np.array(inp)
    labels = np.array(labels)
    #last_input = np.array(last_input)

    binary_labels = []
    for label in labels:
        planes = labelToBinLabel(label)
        binary_labels.append(planes)
    binary_labels = np.array(binary_labels)


    X_train, X_test, y_train, y_test = train_test_split(inp, binary_labels, test_size=param["test_size"], random_state=42)


    train_loader, test_loader = DataLoader(Dataset(X_train, y_train), batch_size=param["batch_size"], shuffle=True, drop_last=True), DataLoader(Dataset(X_test, y_test), batch_size=param["batch_size"], shuffle=True, drop_last=True)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device, " cuda av: ", torch.cuda.is_available(), " cuda device: ", torch.cuda.device(0), torch.cuda.device_count(), torch.cuda.get_device_name(0))
    
    model = PositionPrediction(5, param["batch_size"]).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=param["l_rate"])
    criterion = nn.CrossEntropyLoss()
    
    training = Training(param["epochs"], param["batch_size"], optimizer, criterion, model, device)
    
    training.train_setup(train_loader, param["path"])
    #training.evaluate(test_loader)

if __name__ == main():
    main()
        
        