import TorchRandomSeed#Sebastian Class


seed =0
seedObject = TorchRandomSeed.TorchRandomSeed(seed=1) # was mache ich falsch ?

#my classes
import dataloader
import modelClass
import train
import utils
import plotResults
import eval

with seedObject:

    #inputFeatures = 4 #
    droplist = []#["BloodPressure", "Pregnancies", "Age", "SkinThickness"]
    num_epochs = 2
    batch_size = 4
    test_size = 0.4 # is going to be split again in eval and test
    device = "cuda:0" #if torch.cuda.is_available() else "cpu"
    modelsDirPath = "./Models"
    print(device)
    #device = "cpu"
    lr =0.1 # 0.001 slowed learningrate

    # load data
    trainloader ,evalloader, testloader ,X_train ,X_eval, X_test,  y_train ,y_eval, y_test, inputFeatures, outputFeatures, datasetName, featureNames= dataloader.load_kaggle_diabetes_dataset(batch_size=batch_size , droplist= droplist)
    #trainloader ,evalloader, testloader ,X_train ,X_eval, X_test,  y_train ,y_eval, y_test, inputFeatures, outputFeatures, datasetName, featureNames= dataloader.BreastCancerUCI(batch_size= batch_size, droplist=droplist, test_size=test_size)
    #trainloader ,evalloader, testloader ,X_train ,X_eval, X_test,  y_train ,y_eval, y_test, inputFeatures, outputFeatures, datasetName= dataloader.dryBeanUCI(batch_size=batch_size , droplist= droplist)

    #model = modelClass.Net(inputFeatures= inputFeatures, out_features=outputFeatures)
    model= modelClass.BinaryClassification(inputFeatures= inputFeatures, outputFeatures= outputFeatures)
    modelName = model.modelName
    grads,grads_eval, grads_0, grads_epoch, grads_epoch_0, training_loss_epoch, training_acc, loss_function = train.train(trainloader, model, num_epochs, device, y_train, lr)

    train_predictions,test_predictions = utils.calculatePredictions(model ,X_test, X_train , device)
    featureListALL = utils.unpackingFeatureList(inputFeatures , grads)
    if eval:
        featureListALL_eval = utils.unpackingFeatureList(inputFeatures , grads_eval)

    featureListALL_0 = utils.unpackingFeatureList(inputFeatures , grads_0)
    featureListALL_epoch = utils.unpackingFeatureList(inputFeatures , grads_epoch)
    featureListALL_0_epoch = utils.unpackingFeatureList(inputFeatures , grads_epoch_0)
    dirPath = utils.createDirPath(seed , modelName, datasetName, num_epochs, batch_size, lr)
    print("REPAIR AND UPDATE SAVERESULTS TO NPZ")
    #utils.saveResultsToNPZ(dirPath, featureListALL,featureListALL_0, training_acc, training_loss_epoch)

    print("ploting...")
    plotResults.plotCosineSimilarity(dirPath, "cosine_simialarity",model, modelsDirPath)
    plotResults.plotWeightSignDifferences(dirPath, "percentageWeightsSignDifference",model, modelsDirPath)
    plotResults.plotWeightMagnitude(dirPath, "averageWeightsMagnitude",model, modelsDirPath)
    plotResults.plotL2Distance(dirPath, "L2Distance",model, modelsDirPath)
    plotResults.plotWeightTrace(dirPath, "weightTrace",model, modelsDirPath)
    plotResults.plotGradientsPerFeature(dirPath, inputFeatures, featureListALL, plotName = "fetureListALL")
    plotResults.plotGradientMagnitude(dirPath, "averageGradientMagnitude",featureListALL, perFeature=False)
    plotResults.plotGradientMagnitude(dirPath, "GradientMagnitudePerFeature",featureListALL, perFeature=True)
    plotResults.plotLoss_Acc(dirPath,model,modelsDirPath, trainloader,evalloader,testloader, device, loss_function, num_epochs, y_train, y_eval, y_test)
    plotResults.plotConfusionMatrix(y_train,y_test,  train_predictions, test_predictions, dirPath)
    plotResults.plotGradientsPerSample(featureListALL, num_epochs, X_train, dirPath, "featureListALLPerSample")
