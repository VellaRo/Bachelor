from datetime import date , datetime
import os
import numpy as np
import torch 

def getWeights(model):
    """returns flattend weights from a model
    
    parameters: model                   
    
    """
    weights = []
    for param in model.parameters():

        weights.extend((param.cpu().detach().numpy().flatten()))
    return weights


def calculatePredictions(model ,X_test, X_train , device):
    """
    returns:  train_predictions,test_predictions 
    for cm calculation
    """
    test_predictions = []
    train_predictions = []

    # getting accuracy post training from trainset and testset 
    X_test= X_test.to(device)
    X_train= X_train.to(device)
    with torch.no_grad():
        for i,data in enumerate(X_test):
            y_pred = model(data)
            test_predictions.append(y_pred.argmax().item())

        for i,data in enumerate(X_train):
            y_pred_train = model(data)
            train_predictions.append(y_pred_train.argmax().item())


    return train_predictions,test_predictions 

#unpacking feature list in usale dimensions
def unpackingFeatureList(inputFeatures , grads):
    featureListALL = []

    for i in range(inputFeatures):
        featureListALL.append([])

    for i in range(len(grads)):
        for j in range(len(grads[i])):
            for k in range(inputFeatures):
                featureListALL[k].append(grads[i][j][k].item())
    return featureListALL

def createDirPath(seed , modelName, datasetName, num_epochs, batch_size, lr):

    # datetime now to name results
    datetimeNow =str(date.today()) + str(datetime.now().strftime("_%H%M%S"))

    dirPath = './Results/'+ "seedNum_" + str(seed) + "_" +str(modelName) +"_"+ str(datasetName) +"_"+ 'Num_Epochs_' + str(num_epochs) +'batchSize_'+ str(batch_size)+ '_'+ str(lr)+ '_'+ str(datetimeNow) +"/"

    dirPath = "./test/" + dirPath

    isExist = os.path.exists(dirPath)

    if not isExist:
        os.makedirs(dirPath)
    
    return dirPath
    
def saveResultsToNPZ(dirPath, featureListALL, featureListALL_0 ,training_acc, test_acc, training_loss_epoch, training_loss_batch, test_loss_epoch, test_loss_batch):
    featureListALL = torch.tensor(featureListALL, device = 'cpu')
    featureListALL_0 = torch.tensor(featureListALL_0, device = 'cpu')
    training_acc = torch.tensor(training_acc, device = 'cpu')
    test_acc = torch.tensor(test_acc, device = 'cpu')
    training_loss_epoch = torch.tensor(training_loss_epoch, device = 'cpu')
    training_loss_batch = torch.tensor(training_loss_batch, device = 'cpu')
    test_loss_epoch = torch.tensor(test_loss_epoch, device = 'cpu')
    test_loss_batch = torch.tensor(test_loss_batch, device = 'cpu')
    

    #save data
    with torch.no_grad():
        np.savez(dirPath + 'data.npz', featureListALL = featureListALL , 
                                         featureListALL_0 = featureListALL_0,
                                         training_acc = training_acc, 
                                         test_acc =test_acc,
                                         training_loss_epoch = training_loss_epoch, 
                                         training_loss_batch = training_loss_batch,
                                         test_loss_epoch =test_loss_epoch, 
                                         test_loss_batch = test_loss_batch,
                                         )

def loadData(dirPath):
    #load data according to datatime now same as above since same datetimeNow
    data = np.load(dirPath + 'data.npz')

    return data    
