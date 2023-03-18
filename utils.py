from datetime import date , datetime
import os
import numpy as np
import torch 

def calc_grads(outputs, inputs):
    """
    calculates gradients 

    parameters:
        outputs: model output
        inputs: input batch 

    returns grad: gradients
    """
    _outputs_max_idx = torch.argmax(outputs, dim=1) # indexthat contains maximal value per row (prediction per sample in batch)
    _outputs = torch.gather(outputs, dim=1, index= _outputs_max_idx.unsqueeze(1)) # gather sammelt outputs aus y entlang der Axe 
                                                                                         # dim die in index spezifiziert sind, 
                                                                                         # wobei index einen tensor von shape(batch_size, 1)
                                                                                         # erwartet (->unsqueeze(1))
             
    grad = torch.autograd.grad(torch.unbind(_outputs), inputs)[0]

    return grad

def getWeights(model):
    """returns flattend weights from a model
    
    parameters: model                   
    
    """
    weights = []
    for param in model.parameters():

        weights.extend((param.cpu().detach().numpy().flatten()))
    return np.array(weights)


def calculatePredictions(model ,X, X_train , device):
    """
    returns:  train_predictions,test_predictions 
    for confusionmatrix calculation
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
def unpackingGradients(inputFeatures , grads):
    """
    reshapes gradients to gradients per feature 
    
    parameters:
            inputFeatures: number of input features

            grads: gradientList which needs to be reshaped 

    return: unpacked gradients
    """
    unpackedGradients = []

    for i in range(inputFeatures):
        unpackedGradients.append([])

    for i in range(len(grads)):
        for j in range(len(grads[i])):
            for k in range(inputFeatures):
                unpackedGradients[k].append(grads[i][j][k].item())
    return np.array(unpackedGradients)

def createDirPath(seed , modelName, datasetName, num_epochs, batch_size, lr):
    """
    creates a directory path accrding to the parameters selectet for the training process

    seed: seednumber (for reproducability)

    modelName: which model is beeing used

    dataSetName: which dataset is going to be used

    num_epochs: the number of epochs

    batch_size: size of the batch

    lr: learningrate 

    returns dirPath: the path to the directory where the results will be stored (string) 
    """
    # datetime now to name results
    datetimeNow =str(date.today()) + str(datetime.now().strftime("_%H%M%S"))
    ####./ missing below !!!!
    dirPath = '/Results/'+ "seedNum_" + str(seed) + "_" +str(modelName) +"_"+ str(datasetName) +"_"+ 'Num_Epochs_' + str(num_epochs) +'batchSize_'+ str(batch_size)+ '_'+ str(lr)+ '_'+ str(datetimeNow) +"/"

    dirPath = "./NEWtest" + dirPath

    isExist = os.path.exists(dirPath)

    if not isExist:
        os.makedirs(dirPath)
    
    return dirPath

def appendToNPZ(NPZPath, name, newData):
    """
    appends data to a existing .npz file

    parameters:
            NPZPath: the path to the existing .npz file

            name: name of the new entry to the .npz file

            data: data to save into the .npz file

    returns None
    """    
    data = np.load(NPZPath)
    data = dict(data)
    data[str(name)] = newData

    np.savez(NPZPath,**data)

    return None


def loadData(dirPath):
    """
    load the data from a .npz file  

    parameters:
            dirPath: the path to the data(.npz file) to be loaded
    
    returns : data(loaded data)            
    """
    data = np.load(dirPath + 'data.npz')

    return data    
