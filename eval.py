import torch
import os 
import re 
import numpy as np

import utils


from sklearn.metrics.pairwise import cosine_similarity

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
    cosine_similarity_toInitial = []
    cosine_similarity_toFinal = []
    
    
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


def calcWeightsMagnitude(model, modelsDirPath):

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

import random

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
    weightsPerIndices = np.empty([10])

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

