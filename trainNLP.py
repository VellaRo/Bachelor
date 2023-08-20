from time import time

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

from datasetsNLP import get_agnews
from modelsNLP import SentenceCNN, BiLSTMClassif
import evalModel
import plotResults
from matplotlib import pyplot as plt
import pickle   
import utils 
import numpy as np
import os 
from datetime import datetime
import cega_utils
import re 


def _get_outputs(inference_fn, data, model, device, batch_size=256):

    _data = DataLoader(TensorDataset(data), shuffle=False, batch_size=batch_size)

    try:
        _y_out = []
        for x in _data:
            _y = inference_fn(x[0].to(device))
            _y_out.append(_y.cpu())
        return torch.vstack(_y_out)
    except RuntimeError as re:
        if "CUDA out of memory" in str(re):
            model.to('cpu')
            outputs = _get_outputs(inference_fn, data, model, 'cpu')
            model.to('cuda')
            return outputs
        else:
            raise re


def _get_predictions(inference_fn, data, model, device):
    return torch.argmax(_get_outputs(inference_fn, data, model, device), dim=1)


def validate(inference_fn, model, X, Y):

    if inference_fn is None:
        inference_fn = model

    model.eval()
    device = next(model.parameters()).device

    _y_pred = _get_predictions(inference_fn, X, model, device)
    model.train()

    acc = torch.mean((Y == _y_pred).to(torch.float)).detach().cpu().item()  # mean expects float, not bool (or int)
    return acc

def train_loop(model, optim, loss_fn, tr_data: DataLoader, te_data: tuple, inference_fn=None, \
               n_batches_max=10, device='cuda'):
    
#    print(device)
    model.to(device)
    acc_val = []
    losses = []
    n_batches = 0
    _epochs, i_max = 0, 0
    accs = []
    
    # save model
    import os
    import shutil
    modelsdirPath = "./NLP_Models"

    if os.path.exists(modelsdirPath) and os.path.isdir(modelsdirPath):
        shutil.rmtree(modelsdirPath)

    os.mkdir(modelsdirPath)
    epoch_counter = 0
    iterationCounter = 0
    total_gradientsList = []
    while n_batches <= n_batches_max:
        for i, (text, labels) in enumerate(tr_data, 0):

            #break
            #trainÁcc = validate(inference_fn, model, *tr_data)
            acc = validate(inference_fn, model, *te_data)
            accs.append(acc)
            if i % 100 == 0:
                print(f"test acc @ batch {i+_epochs*i_max}/{n_batches_max}: {acc:.4f}")
                #print("train" + str(trainÁcc))
            text = text.to(device)
            labels = labels.to(device)
            out = model(text)
            loss = loss_fn(out, labels)
            optim.zero_grad()
            loss.backward()
            # Sum up the gradients of the weights in the neural network
            
            total_gradients = 0.0   
            # no softmax
            for param in model.parameters():                
                if param.grad is not None:
                    #print(param.grad.numel())
                    #print(param.grad)

                    total_gradients += (torch.abs(param.grad).sum() / param.grad.numel())
                    #print(total_gradients)
            total_gradientsList.append(total_gradients.cpu())


            # save model
            torch.save(model.state_dict(), modelsdirPath +"/"+str(iterationCounter))
            iterationCounter += 1
            #
            optim.step()
            losses.append(loss.item())
            
            n_batches += 1
            if n_batches > n_batches_max:
                break
        i_max = i
    #copy and name special Models 
    shutil.copyfile(modelsdirPath +"/"+str(0), modelsdirPath +"/initialModel")
    shutil.copy(modelsdirPath +"/"+str(iterationCounter -1), modelsdirPath +"/finalModel") # rename macht probleme ???

    print("NOTE: THESE SAVED MODELS ARE BEEING OVERWRITTEN ON NEXT RUN")
    ##
    acc_val.append(validate(inference_fn, model, *te_data))
    print("accuracies over test set")
    return model, losses, accs , total_gradientsList


if __name__ == '__main__':
    
    #SETUP

    size_train_batch = 64#64
    size_test_batch = 5    # 3h apriori
    n_batches = 5
    embedding_dim = 128
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # get dataset and dataloader
    train_set, test_set, size_vocab, n_classes, vocab = get_agnews(random_state=42, batch_sizes=(size_train_batch, size_test_batch))

    X_test, Y_test = next(iter(test_set))  # only use first batch as a test set
    

    #Y_test_distr = torch.bincount(Y_test, minlength=n_classes)/size_test_batch
    #print(f"class distribution in test set: {Y_test_distr}")  # this should roughly be uniformly distributed

    #model = BiLSTMClassif(n_classes=n_classes, embed_dim=embedding_dim, vocab_size=size_vocab, hid_size=64)
    model = SentenceCNN(n_classes=n_classes, embed_dim=embedding_dim, vocab_size=size_vocab) # learns faster then LSTM
    optimizer = Adam(model.parameters())
    loss_fun = torch.nn.CrossEntropyLoss()

    
    _t_start = time()
    
    model, loss, test_accuracies , total_gradientsList= \
        train_loop(model, optimizer, loss_fun, train_set, (X_test, Y_test),
                   inference_fn=model.forward_softmax, device=device, n_batches_max=n_batches)
    
    dirPath ="./"
    modelsDirPath = dirPath + "NLP_Models"

    loaderList = [test_set] # testLoader
    nameList = ["test"]
    yList = [Y_test]

    inputFeatures = size_vocab  
    num_epochs = n_batches # just for tracking progress
    datasetType = "NLP"
    
    now = datetime.now()
    date_time_string = now.strftime("%Y-%m-%d %H:%M:%S")
    # Replace space with underscore
    date_time_string = date_time_string.replace(" ", "_")
    
    # evaluate trained model
    print("eval")
    evalModel.doALLeval(model, modelsDirPath,dirPath, loaderList, device,optimizer, loss_fun, num_epochs, nameList, yList, inputFeatures, NLP=True)
#
####################################
#  
    if datasetType == "NLP":
        dataPath= dirPath+ "NLP_Results/Trainingresults/"
    else:
        dataPath= dirPath+ "Results/Trainingresults/"

    utils.appendToNPZ(dataPath+ "data.npz", "Total_gradientsList_iteration", total_gradientsList)

    data = utils.loadData(dataPath+ "data.npz")
    
    
    plotResults.plotTrainingResults(data, dataPath)
    

#### CEGA ??
 
    pathToNPZ =  cega_utils.runCEGA(dirPath, modelsDirPath, model, X_test, device, data,date_time_string, test_set , datasetType,vocab )
    rules_data = np.load(pathToNPZ , allow_pickle=True)

    pathToDiscriminative_rules = "./NLP_Results/rulesResults/discriminative_rules/"
    pathToCharacteristic_rules = "./NLP_Results/rulesResults/characteristic_rules"
    resultPaths_dicriminative_rules = os.listdir(pathToDiscriminative_rules)
    resultPaths_characteristic_rules = os.listdir(pathToCharacteristic_rules)
    resultPaths_dicriminative_rules= np.sort(resultPaths_dicriminative_rules)

    #get last generated rule
    mostRecentResultPaths_discriminative = pathToDiscriminative_rules + (resultPaths_dicriminative_rules[-1])

    data = utils.loadData(mostRecentResultPaths_discriminative)
    temp_rules_list_overIterations = data["rules_list_overIterations"]
    trackedRules_OHE , precsicionDict = cega_utils.trackRulesList(temp_rules_list_overIterations, data["rulePrecisionListPerRule_overIterations"])
    utils.appendToNPZ(pathToNPZ, "trackedRules_OHE", trackedRules_OHE)
    utils.appendToNPZ(pathToNPZ, "precsicionDict", precsicionDict)
    
    trackedRules_OHE_NOTFILTERED , precsicionDict_NOTFILTERED = cega_utils.trackRulesList(temp_rules_list_overIterations, data["rulePrecisionListPerRule_overIterations"])
    utils.appendToNPZ(pathToNPZ, "trackedRules_OHE_NOTFILTERED", trackedRules_OHE_NOTFILTERED)
    utils.appendToNPZ(pathToNPZ, "precsicionDict_NOTFILTERED", precsicionDict_NOTFILTERED)

    plotResults.plotRulesResults(data)

    _t_end = time()
    print(f"Training finished in {int(_t_end - _t_start)} s")


    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('#batches')
    ax1.set_ylim(0, 1.)
    ax1.set_ylabel('test accuracy', color=color)
    ax1.plot(test_accuracies,  color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    utils.appendToNPZ(pathToNPZ, "test_accuracies", test_accuracies)

    fig2, ax2 = plt.subplots()  # instantiate a second axes that shares the same x-axis

    color = 'tab:orange'
    ax2.set_ylabel('losses', color=color)  # we already handled the x-label with ax1
    ax2.set_ylim(min(0, min(loss)), max(loss))
    ax2.plot(loss, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_xlabel('#batches')
    utils.appendToNPZ(pathToNPZ, "loss", loss)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    

    if datasetType == "NLP":
        dataPath= dirPath+ "NLP_Results/Trainingresults/"
    else:
        dataPath= dirPath+ "Results/Trainingresults/"
    
    fig.savefig(str(dataPath) + str("Acc"))
    pickle.dump(fig, open(str(dataPath) + str("Acc"), 'wb'))
    fig2.tight_layout()  # otherwise the right y-label is slightly clipped
    
    fig2.savefig(str(dataPath) + str("Loss"))
    pickle.dump(fig2, open(str(dataPath) + str("Loss"), 'wb'))
    plt.show()
