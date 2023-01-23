import torch.nn as nn
import torch.nn.functional as F

class BinaryClassification(nn.Module):
    def __init__(self, inputFeatures=8):
        super(BinaryClassification, self).__init__()
        # Number of input features is 8.
        self.layer_1 = nn.Linear(inputFeatures, 64) 
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 2) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1) # 0.1
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x

class Net(nn.Module):
    def __init__(self, inputFeatures=8,hidden1=64, hidden2=128, hidden3=64,hidden4=64,out_features=2):
        super().__init__()
        self.f_connected1 = nn.Linear(inputFeatures,hidden1)
        self.f_connected2 = nn.Linear(hidden1,hidden2)
        self.f_connected3 = nn.Linear(hidden2,hidden3)
        self.f_connected4 = nn.Linear(hidden3,hidden4)
        self.out = nn.Linear(hidden4,out_features)
        
    def forward(self,x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = F.relu(self.f_connected3(x))
        x = F.relu(self.f_connected4(x))
        x = self.out(x)
        return x

