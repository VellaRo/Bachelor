import torch
import os 
import re 
import numpy as np
import utils
import random


from sklearn.metrics.pairwise import cosine_similarity

def doALLeval(model, modelsDirPath,dirPath, loaderList, device,optimizer, loss_function, num_epochs, nameList, yList, inputFeatures):
    """
        parameters:
            model: an ititialised model with the same parameters, as for training

            modelsDirPath: path to the list of the models per iteration


        important variables:

        
        
        returns: 

        evalSetGradients:   gradients on the evaluationSet ; for each iteration of the training models  
        trainSetGradients:  gradients for training process 
    """
    """
    TODO: do a loop for ["train","eval","test"] to go in one run 
    for i in nameList: 
        name = i
        ... (RUN ALL)
    """
    #np.savez(dirPath + 'data.npz', exec(f'{name}acc = "{acc}"')) #exec :  executes the string that it gets 
    np.savez(dirPath + 'data.npz', y_train = yList[0]) 
    utils.appendToNPZ(dirPath + 'data.npz',"y_eval", yList[1])
    utils.appendToNPZ(dirPath + 'data.npz',"y_test", yList[2])
    utils.appendToNPZ(dirPath + 'data.npz',"inputFeatures", inputFeatures)

    initialModel = model
    #initialModel.load_state_dict(torch.load(modelsDirPath + "/" +str(0)))
    initialModel.load_state_dict(torch.load(modelsDirPath +"/initialModel"))
    initialModel.eval()
    initalWeights = np.array(utils.getWeights(initialModel))

    finalModel = model
    #finalModel.load_state_dict(torch.load(modelsDirPath + "/" +str(len(modelsDirPath))))
    finalModel.load_state_dict(torch.load(modelsDirPath +"/finalModel"))
    finalModel.eval()
    finalWeights = utils.getWeights(finalModel)
    
    # filter out any special models
    r = re.compile("^[0-9]*$")
    modelsDirFiltered = list(filter(r.match, os.listdir(modelsDirPath))) # remove any special models
    

    for name,loader,y in zip(nameList,loaderList,yList) :
  
#### ACC LOSS ------------------------
        accPerEpochList = [] 
        accPerIterationList = [] # everyBatch 
        lossPerEpochList = []
        lossPerIterationList = []
### GRADS 
        gradientList= []#  gradients for each epoch on traoinset / for each iteration on the evaluationSet 

### COSINE_SIMILARITY
        cosine_similarity_toInitialList = []
        cosine_similarity_toFinalList = []

### WEIGHT_SIGN_DIFFERENCES
        weightSignDifferences_toInitialList = []
        weightSignDifferences_toFinalList = []

### WEIGHT_MAGNITUDE
        absoluteIterationWeightsList = []

### L2DIST
        l2Dist_toInitialList = []
        l2Dist_toFinalList = []
### WEIGHT_TRACE
        random10WeightsList = []
        temp1 = []
        temp2 = []
        temp3 = []
        temp4 = []
        temp5 = []
        temp6 = []
        temp7 = []
        temp8 = []
        temp9 = []
        temp10 = []
        random10WeightsList.append(temp1)
        random10WeightsList.append(temp2)
        random10WeightsList.append(temp3)
        random10WeightsList.append(temp4)
        random10WeightsList.append(temp5)
        random10WeightsList.append(temp6)
        random10WeightsList.append(temp7)
        random10WeightsList.append(temp8)
        random10WeightsList.append(temp9)
        random10WeightsList.append(temp10)
        picked = False

### PREDICTIONS_LIST
       # predictionList = []        

#temporary non results list
        temp_Loss_epoch =[]
        tempLoss_iteration = [] # i will mean this along the batches  
        running_corrects_epoch= 0
        counter = 0
        tempIterationCounter= 0


        #trainSetGradients = []


        # load and loop through all model Iterations
        for modelNumber,filename in enumerate(np.sort(list(eval(i) for i in modelsDirFiltered))): #(os.listdir(modelsDirPath)))): #
            model.load_state_dict(torch.load(modelsDirPath + "/" +str(filename)))
            model.eval()

            counter +=1
            tempIterationCounter +=1


            ### calculate grads
            if name == "eval" or name == "train":
                for inputs, labels in loader:

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    inputs.requires_grad = True
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    #
                    grads = utils.calc_grads(outputs, inputs)     

                    if name== "train" or name == "eval":       
                        gradientList.append(grads.cpu())
                    #elif name == "train":
                    #    trainSetGradients.append(grads)


            ### end calculate grads
            

            #get weights of model at iteration
            iterationWeights = utils.getWeights(model)
            ### WEIGHT_SIGNS_DIFFRENCES
            initialWeightSigns = np.sign(initalWeights)
            finalWeightSigns =  np.sign(finalWeights)
           # initialWeightsSigns= initialWeightsSigns.flatten()
            #finalWeights = finalWeights.flatten()
            #iterationWeights = iterationWeights.flatten()


            ### COSINE_SIMILARITY
            cosine_similarity_toInitialList.append(cosine_similarity([initalWeights],[iterationWeights]).item())
            cosine_similarity_toFinalList.append(cosine_similarity([finalWeights], [iterationWeights]).item())
            
           #print(cosine_similarity_toInitialList)
            ### WEIGHT_SIGNS_DIFFRENCES
            iterationWeightsSigns = np.sign(iterationWeights)
            
            differenceCounter_toInitial = 0
            for i,sign in enumerate(initialWeightSigns):

                if iterationWeightsSigns[i] != sign:
                    differenceCounter_toInitial +=1

            weightSignDifferences_toInitialList.append(differenceCounter_toInitial)


            differenceCounter_toFinal = 0
            for i,sign in enumerate(finalWeightSigns):

                if iterationWeightsSigns[i] != sign:
                    differenceCounter_toFinal +=1
            weightSignDifferences_toFinalList.append(differenceCounter_toFinal)

            ### WEIGHT_MAGNITUDE
            absoluteIterationWeights = np.average(np.absolute(iterationWeights))
            absoluteIterationWeightsList.append(absoluteIterationWeights)

            ### L2DIST
            l2Dist_toInitial = np.linalg.norm(initalWeights-iterationWeights)
            l2Dist_toFinal = np.linalg.norm(finalWeights-iterationWeights)
        
            l2Dist_toInitialList.append(l2Dist_toInitial)
            l2Dist_toFinalList.append(l2Dist_toFinal)

            ### WEIGHT_TRACE
            if not(picked):
                picked = True
                randomIndicesList = random.sample(range(0, len(iterationWeights)), 10)

            for i, indices in enumerate(randomIndicesList):
                random10WeightsList[i].append(iterationWeights[indices])

            ### ACC / LOSS / PREDICTIONS_LIST
            predictionList = []     # from last epoch

            with torch.no_grad():
                running_corrects_iteration = 0
                for inputs,labels in loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)  
                    #print(inputs)
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    running_corrects_iteration += torch.sum(preds == labels.data)
                    tempLoss_iteration.append(loss_function(outputs, labels).item())
                    
                    #print(preds.argmax(axis =-1))
                    #print(varaaa)
                    predictionList.extend(preds.cpu())

                running_corrects_epoch += running_corrects_iteration
                temp_Loss_epoch.append(lossPerIterationList)                      

                ## per iteration
                acc= running_corrects_iteration.item() /len(y)
                accPerIterationList.append(acc) 
                lossPerIterationList.append(np.mean(tempLoss_iteration))
                ## per epoch
                if counter == int(len(modelsDirFiltered)/ num_epochs):
                    acc = running_corrects_epoch.item() / (tempIterationCounter *len(y))
                    accPerEpochList.append(acc)
                    loss = np.mean(temp_Loss_epoch)
                    lossPerEpochList.append(loss)

                    #Progress
                    print("Progess: " + "{:.2f}".format( (modelNumber/ len(modelsDirFiltered) *100) )+"%")
                    print( name + " acc: " + "{:.2f}".format(acc*100) +"%") 
                    print( name + " Loss: " + "{:.2f}".format(loss))
                    print("-------------------")

                    #reset temp variables for Epochs
                    tempIterationCounter= 0
                    temp_Loss_epoch = []
                    running_corrects_epoch= 0
                    counter = 0
        ### WEIGHT_SIGNS_DIFFRENCES
        percentageWeightSignDifferences_toInitialList= np.array(weightSignDifferences_toInitialList)/ len(initalWeights)
        percentageWeightSignDifferences_toFinalList= np.array(weightSignDifferences_toFinalList)/ len(finalWeights)

        ### GRADIENT_MAGNITUDE
        if name == "train" or name== "eval": 
            unpackedGradiends = utils.unpackingGradients(inputFeatures, gradientList)
            averagedGradientMagnitude = np.average(np.absolute(unpackedGradiends), axis=0) 
            gradientMagnitudePerFeature = np.absolute(unpackedGradiends)
        
        # save all to NPZ
        utils.appendToNPZ(dirPath+ "data.npz", name + "LossPerEpochList", lossPerEpochList)
        utils.appendToNPZ(dirPath+ "data.npz", name + "LossPerIterationList", lossPerIterationList)
        utils.appendToNPZ(dirPath+ "data.npz", name + "AccPerEpochList", accPerEpochList)
        utils.appendToNPZ(dirPath+ "data.npz", name + "AccPerIterationList", accPerIterationList)
        utils.appendToNPZ(dirPath+ "data.npz", name + "Cosine_similarity_toInitialList", cosine_similarity_toInitialList) # Default always Per Iteration
        utils.appendToNPZ(dirPath+ "data.npz", name + "Cosine_similarity_toFinalList", cosine_similarity_toFinalList)
        utils.appendToNPZ(dirPath+ "data.npz", name + "PercentageWeightSignDifferences_toInitialList", percentageWeightSignDifferences_toInitialList)
        utils.appendToNPZ(dirPath+ "data.npz", name + "PercentageWeightSignDifferences_toFinalList", percentageWeightSignDifferences_toFinalList)
        utils.appendToNPZ(dirPath+ "data.npz", name + "AbsoluteIterationWeightsList", absoluteIterationWeightsList) 
        
        utils.appendToNPZ(dirPath+ "data.npz", name + "L2Dist_toInitialList", l2Dist_toInitialList)
        utils.appendToNPZ(dirPath+ "data.npz", name + "L2Dist_toFinalList", l2Dist_toFinalList)
        utils.appendToNPZ(dirPath+ "data.npz", name + "Random10WeightsList", random10WeightsList)
        utils.appendToNPZ(dirPath+ "data.npz", name + "PredictionList", predictionList) # from last epoch
        

        if name == "train" or name == "eval":
            utils.appendToNPZ(dirPath+ "data.npz", name + "GradientsPerFeature", unpackedGradiends)
            utils.appendToNPZ(dirPath+ "data.npz", name + "GradientMagnitudePerFeature", gradientMagnitudePerFeature)
            utils.appendToNPZ(dirPath+ "data.npz", name + "AveragedGradientMagnitude", averagedGradientMagnitude)

       
    return None # just save to npz
"""

calculates gradiends for evaluationSet

DOES NOT WORK need to adapt

TODO: If i need the gradients of evaluationSet 

"""
def calculateGrads( model,modelsDirPath,evalloader, device,optimizer,loss_function):
    # look into a evaluation set how do the gradients change on this set ? for every interation (after every batch do a eval grads  on eval data)

    ###
    #Pseudotrains a new model for a Epoch with the eval data for each model iteration
    grads_eval = []
    model = model

    r = re.compile("^[0-9]*$")
    modelsDirFiltered = list(filter(r.match, os.listdir(modelsDirPath))) # remove any special models
    
    for i,filename in enumerate(np.sort(list(eval(i) for i in modelsDirFiltered))): #(os.listdir(modelsDirPath)))): #
        model.load_state_dict(torch.load(modelsDirPath + "/" +str(filename)))
        model.eval()

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

def calcAccLoss(model,modelsDirPath, loader, name , device, loss_function, num_epochs, y):
    """calculate Cosine Similarity according to initial and final weights

    patrameters:
                model: model for load state dict
                modelsDirPath: path where the modelsStateDict are saved

        returns:  cosine_similarity_toInitial , cosine_similarity_toFinal"""
    
    print("calcAccLoss: "+ name)
    model = model

    r = re.compile("^[0-9]*$")
    modelsDirFiltered = list(filter(r.match, os.listdir(modelsDirPath))) # remove any special models
    
    Acc_epoch = [] 
    Acc_iteration = [] # everyBatch 
    Loss_epoch = []
    Loss_iteration = []
    
    #temporary non results list
    temp_Loss_epoch =[]
    tempLoss_iteration = [] # i will mean this along the batches  
    running_corrects_epoch= 0
    counter = 0
    tempIterationCounter= 0
    for i,filename in enumerate(np.sort(list(eval(i) for i in modelsDirFiltered))): #(os.listdir(modelsDirPath)))): #
        model.load_state_dict(torch.load(modelsDirPath + "/" +str(filename)))

        model.eval()
        counter +=1
        tempIterationCounter +=1
        with torch.no_grad():
            running_corrects_iteration = 0
            for inputs,labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)  
    
                outputs = model(inputs)
                
                _, preds = torch.max(outputs, 1)
                running_corrects_iteration += torch.sum(preds == labels.data)
                tempLoss_iteration.append(loss_function(outputs, labels).item())
            running_corrects_epoch += running_corrects_iteration
            temp_Loss_epoch.append(Loss_iteration)                      


            ## per iteration
            Acc= running_corrects_iteration.item() /len(y)
            Acc_iteration.append(Acc) 
            Loss_iteration.append(np.mean(tempLoss_iteration))
            ## per epoch
            if counter == int(len(modelsDirFiltered)/ num_epochs):
                Acc = running_corrects_epoch.item() / (tempIterationCounter *len(y))
                Acc_epoch.append(Acc)
                Loss = np.mean(temp_Loss_epoch)
                Loss_epoch.append(Loss)

                print("Progess: " + "{:.2f}".format( (i/ len(modelsDirFiltered) *100) )+"%")
                print( name + " acc: " + "{:.2f}".format(Acc*100) +"%")  ### epoch falsch ?
                print( name + " Loss: " + "{:.2f}".format(Loss))
                print("-------------------")
                #reset temp variables for Epochs
                tempIterationCounter= 0
                temp_Loss_epoch = []
                running_corrects_epoch= 0
                counter = 0
                
                
        

    return Acc_epoch, Acc_iteration, Loss_epoch, Loss_iteration

def testModel():
    """
    final test


    """
    
    pass



def calcConsineSimilarity(model,modelsDirPath):
    """calculate Cosine Similarity according to initial and final weights

    patrameters:
                model: model for load state dict
                modelsDirPath: path where the modelsStateDict are saved

        returns:  cosine_similarity_toInitial , cosine_similarity_toFinal"""
    
    initialModel = model
    #initialModel.load_state_dict(torch.load(modelsDirPath + "/" +str(0)))
    initialModel.load_state_dict(torch.load(modelsDirPath +"/initialModel"))
    initialModel.eval()
    initalWeights = utils.getWeights(initialModel)

    finalModel = model
    #finalModel.load_state_dict(torch.load(modelsDirPath + "/" +str(len(modelsDirPath))))
    finalModel.load_state_dict(torch.load(modelsDirPath +"/finalModel"))
    finalModel.eval()
    finalWeights = utils.getWeights(finalModel)

    model = model

    r = re.compile("^[0-9]*$")
    modelsDirFiltered = list(filter(r.match, os.listdir(modelsDirPath))) # remove any special models
    cosine_similarity_toInitial = []
    cosine_similarity_toFinal = []
    for filename in np.sort(list(eval(i) for i in modelsDirFiltered)): #(os.listdir(modelsDirPath)))): #
        model.load_state_dict(torch.load(modelsDirPath + "/" +str(filename)))
        model.eval()

        #get weights of model at iteration
        iterationWeights = utils.getWeights(model)

        cosine_similarity_toInitial.append(cosine_similarity([initalWeights],[iterationWeights]))
        cosine_similarity_toFinal.append(cosine_similarity([finalWeights], [iterationWeights]))
    
    return cosine_similarity_toInitial , cosine_similarity_toFinal

def calcWeightSignDifferences(model, modelsDirPath):
    """" calculates the percentage of weight singns changes according to the initial weights
    patrameters:
                model: model for load state dict
                modelsDirPath: path where the modelsStateDict are saved

    return percentageWeightSignDifferences_toInitial"""

    initialModel = model
    #initialModel.load_state_dict(torch.load(modelsDirPath + "/" +str(0)))
    initialModel.load_state_dict(torch.load(modelsDirPath +"/initialModel"))
    initialModel.eval()
    initalWeights = utils.getWeights(initialModel)
    initialWeightsSigns = np.sign(initalWeights)
    
    r = re.compile("^[0-9]*$")
    modelsDirFiltered = list(filter(r.match, os.listdir(modelsDirPath))) # remove any special models
    weightSignDifferences_toInitial = []
    for filename in np.sort(list(eval(i) for i in modelsDirFiltered)): #(os.listdir(modelsDirPath)))): #
        model.load_state_dict(torch.load(modelsDirPath + "/" +str(filename)))
        model.eval()

        #get weights of model at iteration
        iterationWeights = utils.getWeights(model)
        iterationWeightsSigns = np.sign(iterationWeights)
        differenceCounter = 0
        for i,sign in enumerate(initialWeightsSigns):
            #print(str(iterationWeightsSigns[i])+ "  " + str(sign))

            if iterationWeightsSigns[i] != sign:
                differenceCounter +=1

        weightSignDifferences_toInitial.append(differenceCounter)

    percentageWeightSignDifferences_toInitial= np.array(weightSignDifferences_toInitial)/ len(initalWeights)
    
    return percentageWeightSignDifferences_toInitial

def calcGradientMagnitude(featureListALL, perFeature=False):
    """
    calculates the gradientMagnitude averaged across the features 

    returns averagedAbsoluteGrads (shape len(featureListAll))
    """
    if not(perFeature):
        averagedAbsoluteGrads = np.average(np.absolute(featureListALL), axis=0) 
        return averagedAbsoluteGrads
    else:
        absoluteGrads = np.absolute(featureListALL)
        return absoluteGrads 


def calcWeightMagnitude(model, modelsDirPath):

    r = re.compile("^[0-9]*$")
    modelsDirFiltered = list(filter(r.match, os.listdir(modelsDirPath))) # remove any special models
    absoluteIterationWeightsList =  []
    for filename in np.sort(list(eval(i) for i in modelsDirFiltered)): #(os.listdir(modelsDirPath)))): #
        model.load_state_dict(torch.load(modelsDirPath + "/" +str(filename)))
        model.eval()

        #get weights of model at iteration
        iterationWeights = utils.getWeights(model)

        absoluteIterationWeights = np.average(np.absolute(iterationWeights))
        absoluteIterationWeightsList.append(absoluteIterationWeights)

    #percentageWeightSignDifferences_toInitial= np.array(weightSignDifferences_toInitial)/ len(initalWeights)
    
    return absoluteIterationWeightsList

def calcL2distance(model, modelsDirPath):
    """calculate l2 distance according to initial and final weights

    patrameters:
                model: model for load state dict
                modelsDirPath: path where the modelsStateDict are saved

        returns:  l2Dist_toInitial , l2Dist_toFinal"""
    
    initialModel = model
    #initialModel.load_state_dict(torch.load(modelsDirPath + "/" +str(0)))
    initialModel.load_state_dict(torch.load(modelsDirPath +"/initialModel"))
    initialModel.eval()
    initalWeights = np.array(utils.getWeights(initialModel))

    finalModel = model
    #finalModel.load_state_dict(torch.load(modelsDirPath + "/" +str(len(modelsDirPath))))
    finalModel.load_state_dict(torch.load(modelsDirPath +"/finalModel"))
    finalModel.eval()
    finalWeights = np.array(utils.getWeights(finalModel))

    model = model

    r = re.compile("^[0-9]*$")
    modelsDirFiltered = list(filter(r.match, os.listdir(modelsDirPath))) # remove any special models
    l2Dist_toInitialList = []
    l2Dist_toFinalList = []
    for filename in np.sort(list(eval(i) for i in modelsDirFiltered)): #(os.listdir(modelsDirPath)))): #
        model.load_state_dict(torch.load(modelsDirPath + "/" +str(filename)))
        model.eval()

        #get weights of model at iteration
        iterationWeights = np.array(utils.getWeights(model))

        #l2Dist_toInitial.append(cosine_similarity([initalWeights],[iterationWeights]))
        #l2Dist_toFinal.append(cosine_similarity([finalWeights], [iterationWeights]))
        l2Dist_toInitial = np.linalg.norm(initalWeights-iterationWeights)
        l2Dist_toFinal = np.linalg.norm(initalWeights-finalWeights)
        
        l2Dist_toInitialList.append(l2Dist_toInitial)
        l2Dist_toFinalList.append(l2Dist_toFinal)
    
    return l2Dist_toInitialList , l2Dist_toFinalList


def calcWeightTrace(model, modelsDirPath):
    model = model

    r = re.compile("^[0-9]*$")
    modelsDirFiltered = list(filter(r.match, os.listdir(modelsDirPath))) # remove any special models
    random10WeightsList = []
    temp1 = []
    temp2 = []
    temp3 = []
    temp4 = []
    temp5 = []
    temp6 = []
    temp7 = []
    temp8 = []
    temp9 = []
    temp10 = []
    random10WeightsList.append(temp1)
    random10WeightsList.append(temp2)
    random10WeightsList.append(temp3)
    random10WeightsList.append(temp4)
    random10WeightsList.append(temp5)
    random10WeightsList.append(temp6)
    random10WeightsList.append(temp7)
    random10WeightsList.append(temp8)
    random10WeightsList.append(temp9)
    random10WeightsList.append(temp10)
    #l2Dist_toFinal = []
    picked = False

    for filename in np.sort(list(eval(i) for i in modelsDirFiltered)): #(os.listdir(modelsDirPath)))): #
        model.load_state_dict(torch.load(modelsDirPath + "/" +str(filename)))
        model.eval()
        iterationWeights = np.array(utils.getWeights(model))
        if not(picked):
            picked = True
            randomIndicesList = random.sample(range(0, len(iterationWeights)), 10)

        for i, indices in enumerate(randomIndicesList):
            random10WeightsList[i].append(iterationWeights[indices])

        #random10WeightsList.append(weightsPerIndices)
   
        #get weights of model at iteration
    return random10WeightsList

