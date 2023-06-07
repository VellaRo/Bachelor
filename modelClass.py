import torch.nn as nn
import torch.nn.functional as F
import torch
# for even small changes cread a new model  

class BinaryClassification0HL16N(nn.Module):
    def __init__(self, inputFeatures=8, outputFeatures=2):
        super(BinaryClassification0HL16N, self).__init__()
        # HL hidden layer #N neruons per layer
        self.modelName = "BinaryClassification0HL16N"

        self.layer_1 = nn.Linear(inputFeatures, 16 ) #64
        self.layer_out = nn.Linear(16, outputFeatures) #64
        
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(p=0.1) # 0.1
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        #x = self.relu(self.layer_2(x))
        #x = self.dropout(x)
        x = self.layer_out(x)
        
        return x
    
class BinaryClassification1HL16N(nn.Module):
    def __init__(self, inputFeatures=8, outputFeatures=2):
        super(BinaryClassification1HL16N, self).__init__()
        # HL hidden layer #N neruons per layer
        self.modelName = "BinaryClassification1HL16N"

        self.layer_1 = nn.Linear(inputFeatures, 16 ) #64
        self.layer_2 = nn.Linear(16, 16) #64,64 
        self.layer_out = nn.Linear(16, outputFeatures) #64
        
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(p=0.1) # 0.1
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        #x = self.dropout(x)
        x = self.layer_out(x)
        
        return x


class BinaryClassification2HL64N(nn.Module):
    def __init__(self, inputFeatures=8, outputFeatures=2):
        super(BinaryClassification2HL64N, self).__init__()
        # HL hidden layer #N neruons per layer
        self.modelName = "BinaryClassification1HL16N"

        self.layer_1 = nn.Linear(inputFeatures, 64 ) #64
        self.layer_2 = nn.Linear(64,64) #64,64 
        self.layer_3 = nn.Linear(64,64) #64,64
        self.layer_out = nn.Linear(64, outputFeatures) #64
        
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(p=0.1) # 0.1
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        #x = self.dropout(x)
        x = self.layer_out(x)
        
        return x
    
    def predict(self,input_list):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

        # Convert input list to a PyTorch tensor
        input_data = torch.tensor(input_list)

        # Create a TensorDataset from the input data
        dataset = torch.utils.data.TensorDataset(input_data)

        # Initialize a DataLoader
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        # Move the model to the appropriate device
        self.to(device)
        self.eval()  # Set the model to evaluation mode
 
        # Make predictions and get predicted classes
        predictions= []
        with torch.no_grad():
            for batch in data_loader:
                input_batch = batch[0].to(device)  # Extract the input data from the batch

                # Forward pass and get predictions
                batch_predictions = self(input_batch)

                # Get predicted classes by selecting the class with the highest probability
                _, batch_predicted_classes = torch.max(batch_predictions, dim=1)

                # Append batch predicted classes to the list
                predictions.extend(batch_predicted_classes.cpu().tolist())

    
        return predictions


"""
only use small layer size to observe more easily

class Net(nn.Module):
    def __init__(self, inputFeatures=8,hidden1=512, hidden2=512, hidden3=512,hidden4=512,out_features=2):
        super().__init__()

        self.modelName = "Net"

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
"""

