import torch
from torch import nn
import TorchRandomSeed
from matplotlib import pyplot as plt

seed =0
seedObject = TorchRandomSeed.TorchRandomSeed(seed=1) 

import dataloader
import modelClass
import train
import utils
import plotResults
import eval

with seedObject:

    droplist = []#["BloodPressure", "Pregnancies", "Age", "SkinThickness"]
    num_epochs = 2
    batch_size = 4
    test_size = 0.4 # is going to be split again in eval and test
    device = "cuda:0" #if torch.cuda.is_available() else "cpu"
    modelsDirPath = "./Models"

    print("calculating on: " +str(device))
    lr =0.1 # 0.001 slowed learningrate

    # load data
    trainloader ,evalloader, testloader ,X_train ,X_eval, X_test,  y_train ,y_eval, y_test, inputFeatures, outputFeatures, datasetName, featureNames= dataloader.load_kaggle_diabetes_dataset(batch_size=batch_size , droplist= droplist)
    #trainloader ,evalloader, testloader ,X_train ,X_eval, X_test,  y_train ,y_eval, y_test, inputFeatures, outputFeatures, datasetName, featureNames= dataloader.BreastCancerUCI(batch_size= batch_size, droplist=droplist, test_size=test_size)
    #trainloader ,evalloader, testloader ,X_train ,X_eval, X_test,  y_train ,y_eval, y_test, inputFeatures, outputFeatures, datasetName= dataloader.dryBeanUCI(batch_size=batch_size , droplist= droplist)

    #model = modelClass.Net(inputFeatures= inputFeatures, out_features=outputFeatures)
    model= modelClass.BinaryClassification0HL16N(inputFeatures= inputFeatures, outputFeatures= outputFeatures)
    modelName = model.modelName
    
    # Backward Propergation - loss and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    train.train(trainloader, model, num_epochs, device, y_train,loss_function, optimizer)

    dirPath = utils.createDirPath(seed , modelName, datasetName, num_epochs, batch_size, lr)

    print("evaluating ...")
    loaderList = [trainloader,evalloader,testloader]
    nameList = ["train","eval", "test"]
    yList = [y_train, y_eval,y_test]
    eval.doALLeval(model, modelsDirPath,dirPath, loaderList, device,optimizer, loss_function, num_epochs, nameList, yList, inputFeatures)

    print("plotting...")
    plotResults.plotCosineSimilarity(dirPath, "cosine_simialarity", set="train")
    plotResults.plotCosineSimilarity(dirPath, "cosine_simialarity", set="eval")
    plotResults.plotCosineSimilarity(dirPath, "cosine_simialarity", set="test")

    plotResults.plotWeightSignDifferences(dirPath, "percentageWeightsSignDifference1" , "train")
    plotResults.plotWeightSignDifferences(dirPath, "percentageWeightsSignDifference2" , "eval")
    plotResults.plotWeightSignDifferences(dirPath, "percentageWeightsSignDifference3" , "test")
    
    plotResults.plotWeightMagnitude(dirPath, "weightsMagnitude1","train")
    plotResults.plotWeightMagnitude(dirPath, "weightsMagnitude2","eval")
    plotResults.plotWeightMagnitude(dirPath, "weightsMagnitude3","test")
    
    plotResults.plotL2Distance(dirPath, "L2Distance1","train")
    plotResults.plotL2Distance(dirPath, "L2Distance2","eval")
    plotResults.plotL2Distance(dirPath, "L2Distance3","test")
    
    plotResults.plotWeightTrace(dirPath, "weightTrace1","train")
    plotResults.plotWeightTrace(dirPath, "weightTrace2","eval")
    plotResults.plotWeightTrace(dirPath, "weightTrace3","test")    
    
    plotResults.plotGradientsPerFeature(dirPath,"gradientsPerFeature1" ,False)
    plotResults.plotGradientsPerFeature(dirPath,"gradientsPerFeature2" ,True )

    plotResults.plotGradientMagnitude(dirPath, "averageGradientMagnitude1","train", perFeature=False)
    plotResults.plotGradientMagnitude(dirPath, "averageGradientMagnitude2","eval", perFeature=False)
    plotResults.plotGradientMagnitude(dirPath, "averageGradientMagnitude3","test", perFeature=False)

    plotResults.plotGradientMagnitude(dirPath, "GradientMagnitudePerFeature1","train", perFeature=True)
    plotResults.plotGradientMagnitude(dirPath, "GradientMagnitudePerFeature2","eval", perFeature=True)
    plotResults.plotGradientMagnitude(dirPath, "GradientMagnitudePerFeature3","test", perFeature=True)
   
    plotResults.plotLoss_Acc(dirPath,"loss_acc1", False)
    plotResults.plotLoss_Acc(dirPath,"loss_acc2",True)

    plotResults.plotConfusionMatrix(dirPath, "confusionMatrix1", set="train")
    plotResults.plotConfusionMatrix(dirPath, "confusionMatrix2", set="eval")
    plotResults.plotConfusionMatrix(dirPath, "confusionMatrix3", set="test")

    plt.show()

    print(dirPath)

