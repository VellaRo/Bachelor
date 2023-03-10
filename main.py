import torch
import pandas as pd
import sklearn.model_selection # train_test_split
import sklearn.metrics  #  confusion_matrix  accuracy_score
import torch.nn as nn 
#Sebastian Class
import TorchRandomSeed

seed =0
seedObject = TorchRandomSeed.TorchRandomSeed(seed=1) # was mache ich falsch ?

#from torch import manual_seed
#manual_seed(seed)

#my classes
import dataloader
import modelClass
import train
import utils
import plotResults

with seedObject:

     # add to classes like with inputFeatures in dataloader
     # add to classes like with inputFeatures in dataloader

    #inputFeatures = 4 #
    droplist = []#pi["BloodPressure", "Pregnancies", "Age", "SkinThickness"]
    num_epochs = 500
    batch_size = 4
    test_size = 0.4 # is going to be split again in eval and test
    device = "cuda:0" #if torch.cuda.is_available() else "cpu"
    modelsDirPath = "./Models"
    print(device)
    #device = "cpu"
    lr =0.001 # 0.001 slowed learningrate

    doEval = False

    # load data
    #trainloader ,evalloader, testloader , X_train , X_test,  y_train ,  y_test, inputFeatures, outputFeatures, datasetName, featureNames= dataloader.load_kaggle_diabetes_dataset(batch_size=batch_size , droplist= droplist)
    trainloader ,evalloader, testloader , X_train , X_test,  y_train ,  y_test, inputFeatures, outputFeatures, datasetName, featureNames= dataloader.BreastCancerUCI(batch_size= batch_size, droplist=droplist, test_size=test_size)
    #trainloader ,evalloader,  testloader , X_train , X_test,  y_train ,  y_test, inputFeatures, outputFeatures, datasetName= dataloader.dryBeanUCI(batch_size=batch_size , droplist= droplist)

    #model = modelClass.Net(inputFeatures= inputFeatures, out_features=outputFeatures)
    model= modelClass.BinaryClassification(inputFeatures= inputFeatures, outputFeatures= outputFeatures)
    modelName = model.modelName
    grads,grads_eval, grads_0, grads_epoch, grads_epoch_0,training_loss_batch, training_loss_epoch, training_acc, test_loss_batch, test_loss_epoch, test_acc= train.train_eval(trainloader,evalloader, testloader, model, num_epochs, device, y_train,y_test, lr, doEval= doEval)

    train_predictions,test_predictions = utils.calculatePredictions(model ,X_test, X_train , device)
    featureListALL = utils.unpackingFeatureList(inputFeatures , grads)
    if eval:
        featureListALL_eval = utils.unpackingFeatureList(inputFeatures , grads_eval)

    featureListALL_0 = utils.unpackingFeatureList(inputFeatures , grads_0)
    featureListALL_epoch = utils.unpackingFeatureList(inputFeatures , grads_epoch)
    featureListALL_0_epoch = utils.unpackingFeatureList(inputFeatures , grads_epoch_0)
    dirPath = utils.createDirPath(seed , modelName, datasetName, num_epochs, batch_size, lr)
    utils.saveResultsToNPZ(dirPath, featureListALL,featureListALL_0, training_acc, test_acc, training_loss_epoch, training_loss_batch, test_loss_epoch, test_loss_batch)
    
    #cosine_similarity_toInitial_weights_epoch= utils.consine_similarity1(intitial_weights, weights_epoch)
    #cosine_similarity_toFinal_weights_epoch= utils.consine_similarity1(final_weights, weights_epoch)
    #TEST_cosine_similarity_toInitial_weights_epoch= utils.consine_similarity1([1,1,1], [[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]])

    #TEST_cosine_similarity_toFinal_weights_epoch= utils.consine_similarity1([9,9,9], [[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]])


    #data= utils.loadData(dirPath)


    #def eval():
    model.load_state_dict(torch.load("./model"))
    model.eval()

    runningCorr = 0
    for inputs, labels in testloader:
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        model.to(device)
        outputs = model(inputs)
        
        _, preds = torch.max(outputs, 1)
            

        runningCorr += torch.sum(preds == labels.data)
    
    print(runningCorr.item()/ len(y_test))
    #print(intitial_weights[0:5])
    #print(final_weights[0:5])
    print("ploting...")
    ##plotReults.cosine_similarity(dirPath, "cosine_simialarity",TEST_cosine_similarity_toInitial_weights_epoch, TEST_cosine_similarity_toFinal_weights_epoch)
    plotResults.plot_cosine_similarity(dirPath, "cosine_simialarity",model, modelsDirPath)

    #plotResults.plotGradientsPerSample(featureListALL, num_epochs, X_train, dirPath, "featureListALLPerSample")

    #plotResults.plot_CM(y_train,y_test,  train_predictions, test_predictions, dirPath)
    #plotResults.plot_Loss_Acc(dirPath, training_loss_batch, training_loss_epoch, training_acc,test_loss_batch, test_loss_epoch, test_acc)
    #plotResults.plot_features(dirPath, inputFeatures, featureListALL, plotName = "fetureListALL")
    #if doEval:
    #    plotReults.plot_features(dirPath, inputFeatures, featureListALL_eval, plotName = "featureListALL_eval")
    
    #plotResults.plot_features(dirPath, inputFeatures, featureListALL_0, plotName = "featureListALL_0")
    ###s#
    #plotResults.plot_features(dirPath, inputFeatures, featureListALL_epoch, plotName = "fetureListALL_epoch")
    #plotResults.plot_features(dirPath, inputFeatures, featureListALL_0_epoch, plotName = "fetureListALL_0_epoch")

    