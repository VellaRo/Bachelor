import torch
import torch.nn as nn
import numpy as np

import sklearn 
import utils


def train(trainloader, model, num_epochs, device, y_train, lr):
    """returns 
    grads = [] 
    grads0 = []
    training_loss_epoch = []
    training_acc = []
    """
    # Backward Propergation - loss and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # after loss.backwards
    grads = [] 
    grads_epoch = []

    ### before loss.backwawrds
    grads_0 = []
    grads_epoch_0 = []

    ### grads with evaluations set for each batch in trainingloop
    grads_eval =[]

    training_loss_epoch = []
    training_acc = []

    # save model
    import os
    import shutil
    modelsdirPath = "./Models"
    if os.path.exists(modelsdirPath) and os.path.isdir(modelsdirPath):
        shutil.rmtree(modelsdirPath)

    os.mkdir(modelsdirPath)
    epoch_counter = 0
    iterationCounter = 0
    for epoch in range(num_epochs):
        running_corrects = 0

        for inputs, labels in trainloader:    
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            inputs.requires_grad = True

            model.to(device)
            model.train()

            outputs = model(inputs)
            softmax = nn.Softmax(dim=1) # for normalised gradients between -1,1
            outputs = softmax(outputs)
            grad_0 = utils.calc_grads(outputs, inputs)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            loss.backward()

            # save model
            torch.save(model.state_dict(), modelsdirPath +"/"+str(iterationCounter))
            iterationCounter += 1
            optimizer.step()
    
            optimizer.zero_grad()
        
            outputs = model(inputs)
            softmax = nn.Softmax(dim=1) # for normalised gradients between -1,1
            outputs = softmax(outputs)
            grad = utils.calc_grads(outputs, inputs)
            
            #after every Batch
            grads.append(grad) # das hier auch im nachhiein ?
            grads_0.append(grad_0) 

        #after every Epoch
        training_acc.append(running_corrects.item() /len(y_train))
        grads_epoch.append(grad)
        grads_epoch_0.append(grad_0)
        training_loss_epoch.append(loss) 
  
        print("Epoch: " + str(epoch_counter))
        print("      Training_acc: " + str(running_corrects.item() /len(y_train)))
        print("-------------------")
        print()

        epoch_counter +=1 

    #copy and name special Models 
    shutil.copyfile(modelsdirPath +"/"+str(0), modelsdirPath +"/initialModel")
    shutil.copy(modelsdirPath +"/"+str(iterationCounter -1), modelsdirPath +"/finalModel") # rename macht probleme ???

    print("NOTE: THESE SAVED MODELS ARE BEEING OVERWRITTEN ON NEXT RUN")
    return grads,grads_eval, grads_0, grads_epoch, grads_epoch_0, training_loss_epoch, training_acc, loss_function# weights_batch, weights_epoch, intitial_weights#, final_weights

