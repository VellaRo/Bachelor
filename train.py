import torch
import torch.nn as nn
import numpy as np

import sklearn 
import utils


def eval(evalloader, model, num_epochs, device, y_train,y_test, lr,optimizer,loss_function, grads_eval):
    # look into a evaluation set how do the gradients change on this set ? for every interation (after every batch do a eval grads  on eval data)
    for e_inputs, e_labels in evalloader:

                e_inputs = e_inputs.to(device)
                e_labels = e_labels.to(device)

                e_inputs.requires_grad = True

                optimizer.zero_grad()
                
                outputs_eval = model(e_inputs)

                loss = loss_function(outputs_eval, e_labels)

                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()

                outputs_eval = model(e_inputs)

                _outputs_eval_max_idx = torch.argmax(outputs_eval, dim=1) # indexthat contains maximal value per row (prediction per sample in batch)
                _outputs_eval = torch.gather(outputs_eval, dim=1, index= _outputs_eval_max_idx.unsqueeze(1)) # gather sammelt outputs aus y entlang der Axe 
                                                                                         # dim die in index spezifiziert sind, 
                                                                                         # wobei index einen tensor von shape(batch_size, 1)
                                                                                         # erwartet (->unsqueeze(1))
             
                grad_eval = torch.autograd.grad(torch.unbind(_outputs_eval), e_inputs)[0]
                grads_eval.append(grad_eval)

    return grads_eval

def calc_grads(outputs, inputs):

    _outputs_max_idx = torch.argmax(outputs, dim=1) # indexthat contains maximal value per row (prediction per sample in batch)
    _outputs = torch.gather(outputs, dim=1, index= _outputs_max_idx.unsqueeze(1)) # gather sammelt outputs aus y entlang der Axe 
                                                                                         # dim die in index spezifiziert sind, 
                                                                                         # wobei index einen tensor von shape(batch_size, 1)
                                                                                         # erwartet (->unsqueeze(1))
             
    grad = torch.autograd.grad(torch.unbind(_outputs), inputs)[0]

    return grad

def test_model(model, device, testloader, loss_function):
    #model.load_state_dict(torch.load("./model"))
    model.eval()
    with torch.no_grad():
        test_running_corrects = 0
        for t_inputs,t_labels in testloader:
        
            t_inputs = t_inputs.to(device)
            t_labels = t_labels.to(device)  
    
            t_outputs = model(t_inputs)
                
            t__, t_preds = torch.max(t_outputs, 1)
            test_running_corrects += torch.sum(t_preds == t_labels.data)
            test_loss = loss_function(t_outputs, t_labels)
                
    return test_loss, test_running_corrects

def train_eval(trainloader,evalloader, testloader, model, num_epochs, device, y_train,y_test, lr, doEval= False):
    """returns 
    training_loss_batch =[] 
    grads = [] 
    training_loss_epoch = []
    test_loss_epoch = []
    test_loss_batch = []
    training_acc = []
    test_acc = []
    loss = []
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

 


    training_loss_batch =[] 
    training_loss_epoch = []
    training_acc = []

    test_loss_batch = [] 
    test_loss_epoch = [] 
    test_acc = []
    #temp var
    loss = []

    weights_epoch = []
    weights_batch = []

    # save model
    import os
    import shutil
    modelsdirPath = "./Models"
    if os.path.exists(modelsdirPath) and os.path.isdir(modelsdirPath):
        shutil.rmtree(modelsdirPath)

    os.mkdir(modelsdirPath)
    #torch.save(model.state_dict(), modelsdirPath +"/initialModel")
    #epoch_counter = 0
    intitial_weights = []
    # for param in model.parameters():
        #print(param.detach().numpy().flatten())
        # to much ram# intitial_weights.extend((param.detach().numpy().flatten()))
    ##
    epoch_counter = 0
    iterationCounter = 0
    for epoch in range(num_epochs):
        running_corrects = 0
        test_running_corrects = 0

        for inputs, labels in trainloader:    
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            inputs.requires_grad = True

            model.to(device)
            model.train()

            outputs = model(inputs)
            softmax = nn.Softmax(dim=1)
            outputs = softmax(outputs)
            grad_0 = calc_grads(outputs, inputs)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_function(outputs, labels)
            training_loss_batch.append(loss)# training_loss_batch.append(loss.cpu())       test1
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

            loss.backward()#retain_graph=True) # the grads are from now on no longer none 
           # #
           # hier wurden die weights gespeichert
           # #       
            #print(iterationCounter)
            # save model

            torch.save(model.state_dict(), modelsdirPath +"/"+str(iterationCounter))
            iterationCounter += 1
            optimizer.step()
            
          
            if doEval:
                grads_eval = eval(evalloader, model, num_epochs, device, y_train,y_test, lr,optimizer,loss_function, grads_eval=grads_eval)

            optimizer.zero_grad()
            outputs = model(inputs)
            softmax = nn.Softmax(dim=1)
            outputs = softmax(outputs)

            grad = calc_grads(outputs, inputs)


            grads.append(grad) # das hier auch im nachhiein ?
            grads_0.append(grad_0) 
            
            #do that after Training
            #test_loss, test_running_corrects = test_model(model, device, testloader, loss_function)
            #test_loss_batch.append(test_loss) #test1 test_loss_batch.append(test_loss.cpu())


        ##test
        ##testEnd

        training_acc.append(running_corrects.item() /len(y_train))
        #print(len(y_train)/len())

        #test_acc.append(test_running_corrects.item() /len(y_test)) 
        grads_epoch.append(grad)
        grads_epoch_0.append(grad_0)
        # takes to much ram# weights_epoch.append(weights)
        #cosine_similarity_epoch.append(cosine_similarity)

        #print(len(y_test))
        training_loss_epoch.append(loss) # test1 training_loss_epoch.append(loss.cpu())
        #test_loss_epoch.append(test_loss) # test1 test_loss_epoch.append(test_loss.cpu())

        print("Epoch: " + str(epoch_counter))
        print("      Training_acc: " + str(running_corrects.item() /len(y_train)))
        #print("      Testing_acc : " + str(test_running_corrects.item() /len(y_test)))
        print("-------------------")
        print()

        epoch_counter +=1 
    #print("jo")
    #print(iterationCounter)
    shutil.copyfile(modelsdirPath +"/"+str(0), modelsdirPath +"/initialModel")
    #model.load_state_dict(torch.load(modelsdirPath +"/0"))
    #weights = utils.getWeights(model)
    #print(weights[255:256])
    shutil.copy(modelsdirPath +"/"+str(iterationCounter -1), modelsdirPath +"/finalModel") # rename macht probleme ???
    #model.load_state_dict(torch.load(modelsdirPath +"/169"))
    #weights = utils.getWeights(model)
    #print(weights[255:256])
    print("NOTE: THESE SAVED MODELS ARE BEEING OVERWRITTEN ON NEXT RUN")
    return grads,grads_eval, grads_0, grads_epoch, grads_epoch_0, training_loss_batch, training_loss_epoch, training_acc, test_loss_batch, test_loss_epoch, test_acc, loss_function# weights_batch, weights_epoch, intitial_weights#, final_weights

