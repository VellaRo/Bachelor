import torch
import os 
import re 
import numpy as np
import utils
import random


from sklearn.metrics.pairwise import cosine_similarity

def doALLeval(model, modelsDirPath,dirPath, loaderList, device,optimizer, loss_function, num_epochs, nameList, yList, inputFeatures, random_indices_test):
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
    dataPath= dirPath+ "Trainingresults/"
    #np.savez(dirPath + 'data.npz', exec(f'{name}acc = "{acc}"')) #exec :  executes the string that it gets 
    np.savez(dataPath + 'data.npz', y_test = yList[0]) 
    #utils.appendToNPZ(dirPath + 'data.npz',"y_eval", yList[1])
    #utils.appendToNPZ(dirPath + 'data.npz',"y_test", yList[2])
    #utils.appendToNPZ(dirPath + 'data.npz',"y_test", yList[0])
    utils.appendToNPZ(dataPath + 'data.npz',"inputFeatures", inputFeatures)

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
        gradientList_iteration = []
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

        #counterTest = 0

        # load and loop through all model Iterations
        for modelNumber,filename in enumerate(np.sort(list(eval(i) for i in modelsDirFiltered))): #(os.listdir(modelsDirPath)))): # iterations time 
            model.load_state_dict(torch.load(modelsDirPath + "/" +str(filename)))
            model.eval()

            counter +=1
            tempIterationCounter +=1

            ### calculate grads

            if True :#name == "eval" or name == "train":
                gradAtitearation = [None] * len(yList[0] -1)
                random_indices_testCOUNTER = 0
                for inputs, labels in loader:
                    #inputs = inputs.to(device)
                    #labels = labels.to(device)
#
                    #inputs.requires_grad = True
                    #optimizer.zero_grad()
                    #outputs = model(inputs)
                    #loss = loss_function(outputs, labels)
                    #loss.backward()
                    #optimizer.step()
                    #optimizer.zero_grad()
                    #outputs = model(inputs)
                    ##
                    #grads = utils.calc_grads(outputs, inputs)     
                    #gradientList.append(grads.cpu())
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    n_samples = 25
                    stdev_spread = 0.2

                    #inputs = F.softmax(input, dim=1) # softmax

                    grad = utils.smooth_grad(inputs, n_samples, stdev_spread ,model) # for test dataset # grad for a Sample
                    #print(np.shape(grad)) # 32 ,8 , batchsize , featuresize
                    ##
                    for i in range(len(inputs)):  # for batch size
                        #print(random_indices_testCOUNTER)
                        gradAtitearation[random_indices_test[random_indices_testCOUNTER]] = grad[i] # grads for the whole test dataset at given iteration 
                        random_indices_testCOUNTER +=1
                    #print(counterTest)
                    #counterTest += 1
                    #print(len(gradAtitearation))
                gradientList_iteration.append(gradAtitearation)# 11,154,
                
                #get weights of model at iteration
                #if name=="train":
                #iterationWeightsF = utils.getWeights(finalModel)
                #elif name == "eval":

                iterationWeights = utils.getWeights(model)
                
                #if (iterationWeightsF.all() !=iterationWeights.all()):
                #    print(str(iterationWeightsF !=iterationWeights))
                #    print(name)
                ### WEIGHT_SIGNS_DIFFRENCES
                initialWeightSigns = np.sign(initalWeights)
                finalWeightSigns =  np.sign(finalWeights)

                ### COSINE_SIMILARITY
                cosine_similarity_toInitialList.append(cosine_similarity([initalWeights],[iterationWeights]).item())
                cosine_similarity_toFinalList.append(cosine_similarity([finalWeights], [iterationWeights]).item())

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

                    gradientList.append(gradAtitearation)  
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
        if True:#name == "train" or name== "eval": 
            #print(np.shape(gradientList))
            unpackedGradiends = utils.unpackingGradients(inputFeatures, gradientList)
            averagedGradientMagnitude = np.average(np.absolute(unpackedGradiends), axis=0) 
            gradientMagnitudePerFeature = np.absolute(unpackedGradiends)
            ## per iteration:
            unpackedGradiends_iteration = utils.unpackingGradients(inputFeatures, gradientList_iteration)
            averagedGradientMagnitude_iteration = np.average(np.absolute(unpackedGradiends_iteration), axis=0) 
            gradientMagnitudePerFeature_iteration = np.absolute(unpackedGradiends_iteration)
        
        # save all to NPZ
        utils.appendToNPZ(dataPath+ "data.npz", name + "LossPerEpochList", lossPerEpochList)
        utils.appendToNPZ(dataPath+ "data.npz", name + "LossPerIterationList", lossPerIterationList)
        utils.appendToNPZ(dataPath+ "data.npz", name + "AccPerEpochList", accPerEpochList)
        utils.appendToNPZ(dataPath+ "data.npz", name + "AccPerIterationList", accPerIterationList)
        utils.appendToNPZ(dataPath+ "data.npz", name + "PredictionList", predictionList) # from last epoch

        if True:#name == "train" or name == "eval":
            utils.appendToNPZ(dataPath+ "data.npz", name + "Cosine_similarity_toInitialList", cosine_similarity_toInitialList) # Default always Per Iteration
            utils.appendToNPZ(dataPath+ "data.npz", name + "Cosine_similarity_toFinalList", cosine_similarity_toFinalList)
            utils.appendToNPZ(dataPath+ "data.npz", name + "PercentageWeightSignDifferences_toInitialList", percentageWeightSignDifferences_toInitialList)
            utils.appendToNPZ(dataPath+ "data.npz", name + "PercentageWeightSignDifferences_toFinalList", percentageWeightSignDifferences_toFinalList)
            utils.appendToNPZ(dataPath+ "data.npz", name + "AbsoluteIterationWeightsList", absoluteIterationWeightsList) 
            utils.appendToNPZ(dataPath+ "data.npz", name + "L2Dist_toInitialList", l2Dist_toInitialList)
            utils.appendToNPZ(dataPath+ "data.npz", name + "L2Dist_toFinalList", l2Dist_toFinalList)
            utils.appendToNPZ(dataPath+ "data.npz", name + "Random10WeightsList", random10WeightsList)

            utils.appendToNPZ(dataPath+ "data.npz", name + "GradientsPerSamplePerFeature", gradientList) # for every trainingEpoch we calc the grads on testDataset
            utils.appendToNPZ(dataPath+ "data.npz", name + "GradientsPerFeature", unpackedGradiends)
            utils.appendToNPZ(dataPath+ "data.npz", name + "GradientMagnitudePerFeature", gradientMagnitudePerFeature)
            utils.appendToNPZ(dataPath+ "data.npz", name + "AveragedGradientMagnitude", averagedGradientMagnitude)
            
            utils.appendToNPZ(dataPath+ "data.npz", name + "GradientsPerSamplePerFeature_iteration", gradientList_iteration) # for every trainingIteration we calc the grads on testDataset
            utils.appendToNPZ(dataPath+ "data.npz", name + "GradientsPerFeature_iteration", unpackedGradiends_iteration)
            utils.appendToNPZ(dataPath+ "data.npz", name + "GradientMagnitudePerFeature_iteration", gradientMagnitudePerFeature_iteration)
            utils.appendToNPZ(dataPath+ "data.npz", name + "AveragedGradientMagnitude_iteration", averagedGradientMagnitude_iteration)

       
    return None # just save to npz

