import torch
import pandas as pd
import sklearn.model_selection # train_test_split
import sklearn.metrics  #  confusion_matrix  accuracy_score
import random

#Sebastian Class
import TorchRandomSeed

seed =0
#TorchRandomSeed.TorchRandomSeed(seed) # was mache ich falsch ?

import torch
torch.manual_seed(seed)
data = pd.read_csv(r"/home/rosario/explainable/Bachelor/Diabetes/Data/diabetes.csv")

X = data.drop('Outcome' , axis = 1) #independent Feature



### droping all features that i dont want in my Dataloader

#X = X.drop('Pregnancies' , axis = 1)
#X = X.drop('BloodPressure' , axis = 1)
#X = X.drop('Age' , axis = 1)
#X = X.drop('SkinThickness' , axis = 1)
#X = X.drop('Glucose' , axis = 1)
#X = X.drop('Insulin' , axis = 1)
y = data["Outcome"]

inputFeatures = 8 #
num_epochs = 3
batch_size =4 #4 

train_shuffele =True

#device ="cpu" #cpu is faster
device = "cuda:0" if torch.cuda.is_available() else "cpu"

X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y , test_size =0.2, random_state=0)


#convert them to tensors
X_train=torch.FloatTensor(X_train.values)
X_test=torch.FloatTensor(X_test.values)
y_train=torch.LongTensor(y_train.values)
y_test=torch.LongTensor(y_test.values)
X_test
#noramlize
#X_train = torch.nn.functional.normalize(X_train, p=1.0, dim = 0)
#X_test = torch.nn.functional.normalize(X_test, p=1.0, dim = 0)

#X_train = X_train.to(device)
#X_test = X_test.to(device)

#X_train.requires_grad = True
#y_train.requires_grad = True
#X_test.requires_grad = True
#y_test.requires_grad = True

#7467532

trainset = list(zip(X_train , y_train))            


trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=train_shuffele , num_workers=2)

testset = list(zip(X_test , y_test))

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

import torch.nn as nn
import torch.nn.functional as F

class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Number of input features is 8.
        self.layer_1 = nn.Linear(8, 64) 
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 2) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        #self.batchnorm1 = nn.BatchNorm1d(64)
        #self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        #x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        #x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x

class Net(nn.Module):
    def __init__(self,input_features=inputFeatures,hidden1=16, hidden2=64, hidden3=64,hidden4=16,out_features=2):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features,hidden1)
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

        #torch.manual_seed(20)
#model = Net()
model = BinaryClassification()
#model.parameters

# Backward Propergation - loss and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.00001) # 0.001 works # 0.0001 works good slows down # 




final_losses=[]
grads = [] 
epoch_final_losses = []
test_epoch_final_losses = []
training_acc = []
test_acc = []
loss = []
epoch_counter = 0
for epoch in range(num_epochs):

    running_corrects = 0

    for inputs, labels in trainloader:

        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs.requires_grad = True
        #inputs.retain_grad= True
        model.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs)
        #outputs = model.forward(inputs) # ist das nicht das gleiche wie model(inputs)

        loss = loss_function(outputs, labels)
        final_losses.append(loss.cpu())
        #print("outputs")
        #print(outputs)

        _, preds = torch.max(outputs, 1)
        #print(preds)
        running_corrects += torch.sum(preds == labels.data)
            
        #print(running_corrects)
        # backward
        loss.backward(retain_graph=True) # the grads are from now on no longer none #   same ? grad = torch.autograd.grad(torch.unbind(_outputs), inputs)[0]

        #x = x.requires_grad_(True)
        """
        -> torch.unbind erwartet einen Vektor. Wenn du y den shap=(batch_size,
        n_classes) hat und n_classes>1, dann musst du noch die genaue Variable
        festlegen nach der abgeleitet werden soll, was leider etwas umstaendlich
        ist:
        """

        _outputs_max_idx = torch.argmax(outputs, dim=1) # indexthat contains maximal value per row (prediction per sample in batch)
            
        _outputs = torch.gather(outputs, dim=1, index= _outputs_max_idx.unsqueeze(1)) # gather sammelt outputs aus y entlang der Axe 
                                                                                         # dim die in index spezifiziert sind, 
                                                                                         # wobei index einen tensor von shape(batch_size, 1)
                                                                                         # erwartet (->unsqueeze(1))
        #torch.autograd.backward(torch.unbind(_outputs))
        grad = torch.autograd.grad(torch.unbind(_outputs), inputs)[0]
        grads.append(grad)

        optimizer.step() # optimizer does not change the inputs.grads 

    ##evalutaion_step() #return test_accuracy per epoch 
    test_running_corrects = 0
    for inputs,labels in testloader:
        
        inputs = inputs.to(device)
        labels = labels.to(device)  
    
        with torch.no_grad():
    
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_running_corrects += torch.sum(preds == labels.data)
            test_loss = loss_function(outputs, labels)


    training_acc.append(running_corrects.item() /len(y_train))
    test_acc.append(test_running_corrects.item() /len(y_test)) 
    epoch_final_losses.append(loss.cpu())
    test_epoch_final_losses.append(test_loss.cpu())

    print("Epoch: " + str(epoch_counter) +" " + str(running_corrects.item() /len(y_train)))
    print("Epoch: Testing_acc" + str(epoch_counter) +" " + str(test_running_corrects.item() /len(y_test)))

    epoch_counter +=1 
print(training_acc)


test_predictions = []
train_predictions = []

# getting accuracy post training from trainset and testset 
X_test= X_test.to("cuda:0")
X_train= X_train.to("cuda:0")
with torch.no_grad():
    for i,data in enumerate(X_test):
        y_pred = model(data)
        test_predictions.append(y_pred.argmax().item())

    for i,data in enumerate(X_train):
        y_pred_train = model(data)
        train_predictions.append(y_pred_train.argmax().item())



#unpacking feature list in usale dimensions
import numpy as np

featureListALL = []

for i in range(inputFeatures):
    featureListALL.append([])

print(featureListALL[1])
for i in range(len(grads)):
    for j in range(len(grads[i])):
        for k in range(inputFeatures):
            featureListALL[k].append(grads[i][j][k].item())



from matplotlib import pyplot as plt
#plot the loss function
from datetime import date , datetime
import os
import pickle #figx = pickle.load(open('FigureObject.fig.pickle', 'rb')) 
              #figx.show() # Show the figure, edit it, etc.!


# datetime now to name results
datetimeNow =str(date.today()) + str(datetime.now().strftime("_%H%M%S"))
#print(datetimeNow)

 #dir_Path
path = './Results/'+  'Num_Epochs_' + str(num_epochs) +'batchSize_'+ str(batch_size)+ '_'+ str(datetimeNow) +"/"
#print(path)

isExist = os.path.exists(path)

if not isExist:
    os.makedirs(path)

#save data
with torch.no_grad():
    np.savez(str(path) + 'data.npz', featureListALL = featureListALL , # 
                                     training_acc = training_acc, 
                                     test_acc =test_acc,
                                     epoch_final_losses = epoch_final_losses, #training  loss
                                     test_epoch_final_losses =test_epoch_final_losses, #test loss
                                     final_losses = final_losses
                                           )

#load data according to datatime now same as above since same datetimeNow
data = np.load(str(path) + 'data.npz')

featureListALL = data["featureListALL"]

import seaborn as sns

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(5,3) )


cmTest = sklearn.metrics.confusion_matrix(y_test,test_predictions)
cmTrain = sklearn.metrics.confusion_matrix(y_train,train_predictions)

sns.heatmap(cmTest, linewidth=0.5 ,ax=axs[0])
axs[0].set_ylabel('Actual')
axs[0].set_xlabel('Predict')
axs[0].set_title('Test_Acc: ' + str(sklearn.metrics.accuracy_score(y_test , test_predictions)))

sns.heatmap(cmTrain, linewidth=0.5,ax=axs[1])
axs[1].set_xlabel('Actual')
axs[1].set_ylabel('Predicted')
axs[1].set_title('LastTrain_Acc: ' + str(sklearn.metrics.accuracy_score(y_train , train_predictions)))



fig.savefig(str(path) + "_TrueFalseHeatmap")
pickle.dump(fig, open(str(path) + '_TrueFalseHeatmap.pickle', 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`



fig, axs = plt.subplots(nrows=3, ncols=2)
fig.tight_layout(pad=2.0)

with torch.no_grad():
    axs[0][0].plot(range(len(final_losses)),final_losses) #change to loss_per_batch
    axs[0][0].set_xlabel('batch')
    axs[0][0].set_ylabel('training_loss')


    axs[1][0].set_xlabel('epochs')
    axs[1][0].set_ylabel('training_loss')
    axs[1][0].plot(range(len(epoch_final_losses)),epoch_final_losses) # change to loss_per_epoch

    axs[2][0].set_xlabel('epochs')
    axs[2][0].set_ylabel('training_acc')
    axs[2][0].plot(range(len(training_acc)),training_acc)
    
    axs[0][1].plot(range(len(final_losses)),final_losses) #change to loss_per_batch
    axs[0][1].set_xlabel('batch')
    axs[0][1].set_ylabel('test_loss')
    

    axs[1][1].set_xlabel('epochs')
    axs[1][1].set_ylabel('test_loss')
    axs[1][1].plot(range(len(test_epoch_final_losses)),test_epoch_final_losses) # change to loss_per_epoch

    axs[2][1].set_xlabel('epochs')
    axs[2][1].set_ylabel('test_acc')
    axs[2][1].plot(range(len(test_acc)),test_acc)

fig.savefig(str(path) + "_loss")
pickle.dump(fig, open(str(path) + '_loss.pickle', 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`


fig, axs = plt.subplots(inputFeatures)

fig.set_size_inches(10,15)
fig.tight_layout(pad=2.0)
#find maximum/minumum 

maxValue= 0
minValue= 0
#calculate metrics
gradientMean_Features = []
for i in range(len(featureListALL)):
    gradientMean_Features.append(featureListALL[i].mean())

    maxValue = max( maxValue, max( featureListALL[i]))    
    minValue = min( minValue, min(featureListALL[i]))

#plot for all features
for i in range(len(axs)):
    axs[i].set_ylim(minValue,maxValue)
    axs[i].set_xlabel("mean:" + str(gradientMean_Features[i]))
    axs[i].plot(featureListALL[i])

plt.savefig(str(path) +"featureListALL" )
pickle.dump(fig, open(str(path) + 'featureListALL.pickle', 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`


plt.tight_layout()  
plt.show()


"""

#save data as .npz according to datatime now

#TODO:
# Sava also different metrics accuracy , mean, 
# maybe accuracy over time 
# plot the gradients for every image over time 
#

##### add batch size and model structure 

#makeDir for all plotted things


plt.plot()
"""