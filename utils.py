from datetime import date , datetime
import os
import numpy as np
import torch 
from torch.nn.functional import softmax

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

    grad = torch.autograd.grad(torch.unbind(_outputs), inputs)[0]

    return grad

def smooth_grad(input, n_samples, stdev_spread ,model,device):   
                
                #input_embedded = model.embed_input(input)
                try: 
                    input = model.embed_input(input)
                except:
                    pass
                if input.is_cuda:

                    input = input.detach().cpu()
                # Convert the tensor to a NumPy array
                
                ## embed input

                
                
                input_np = input.detach().numpy()
                    

                stdev = stdev_spread * (torch.max(input) - torch.min(input))

                total_gradients = torch.zeros_like(input.data,device=device)
                for i in range(n_samples):

                    # Create a batch of inputs with added Gaussian noise
                    noisy_inputs = input + np.random.normal(0, stdev, input_np.shape).astype(np.float32)
                    noisy_inputs = noisy_inputs.to(device)
                        
                    noisy_inputs.requires_grad = True


                    try: 
                        out = model.forward_embedded_softmax(noisy_inputs)

                    except:

                        out = model(noisy_inputs)
                        out = softmax(out, dim=1)

                    #CHECK FOR WHICH DIMENSION IS CORRECT
                    grad = calc_grads(out, noisy_inputs)
                    grad = grad.detach()
                    total_gradients +=grad 
                    #summed_total_gradients =  torch.sum(total_gradients, dim=-1)
                # Average the summed  up gradients
                avg_gradients = total_gradients / n_samples

                return avg_gradients.cpu().detach().numpy()

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
    ## UNUSED
##
####
    ###
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
    #for i in range(85):
        unpackedGradients.append([])

    for i in range(len(grads)):
        for j in range(len(grads[i])):
            for k in range(inputFeatures):
            #for k in range(85):
             
                unpackedGradients[k].append(grads[i][j][k].item())
    return np.array(unpackedGradients)

def flatten_gradients(grads):
    #UNUSED ???
    ##
    ###
    # get number of input features
    inputFeatures = grads.shape[0]
    
    # initialize unpackedGradients list
    unpackedGradients = [[] for _ in range(inputFeatures)]
    
    # flatten gradients and append to unpackedGradients list
    for i in range(len(grads)):
        for j in range(len(grads[i])):
            grad = grads[i][j]
            unpacked = np.array(grad)
            unpackedGradients = np.concatenate((unpackedGradients, unpacked), axis=1)
    
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
    with open(NPZPath, 'rb') as f:

        data = np.load(NPZPath, allow_pickle=True)
        data = dict(data)
        data[str(name)] = newData

    np.savez(NPZPath,**data )

    return None


def loadData(pathToNPZ):
    """
    load the data from a .npz file  

    parameters:
            dirPath: the path to the data(.npz file) to be loaded
    
    returns : data(loaded data)            
    """
    #data = np.load(dirPath + 'data.npz' , allow_pickle=True)
    data = np.load(pathToNPZ, allow_pickle=True)
    return data    

def binData(data , n):
    #print("THIS IS WINDOW DATA!!")
    binnedData = []
    indicesList = []
    for i in range(0,len(data),n):

        if i+n > len(data):
            averageBin = np.average(data[i:-1])
            indicesList.append(len(data))
        else:
            averageBin = np.average(data[i:i+n])
            indicesList.append(i)
        
        binnedData.append(averageBin)
        
        #indicesList.append(i)

    return binnedData,indicesList


def generate_windowed_values(arr, window_size):
    """
    Generate windowed values from an array.

    Parameters:
    arr (array-like): The input array.
    window_size (int): The size of the window.

    Returns:
    list of arrays: A list of windowed values.
    """
    windowed_values = []
    for i in range(len(arr) - window_size + 1):
        window = arr[i:i+window_size]
        windowed_values.append(np.mean(window))
    return windowed_values

def calculate_mean_of_lists(list_of_lists):
        means = []
        for sublist in list_of_lists:
            if len(sublist) == 0:
                means.append(-1)
            else:
                sublist_mean = np.mean(sublist)
                means.append(sublist_mean)
        return means