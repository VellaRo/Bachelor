import torch
from torch import nn
import pickle
import train
import TorchRandomSeed
import modelClass
import dataloader
from torch.utils.data import SubsetRandomSampler

seed =1
seedObject = TorchRandomSeed.TorchRandomSeed(seed=1) 

with seedObject:
    droplist = []#["BloodPressure", "Pregnancies", "Age", "SkinThickness"]
    num_epochs =1
    batch_size = 32
    test_size = 0.02 #0.2 # is going to be split again in eval and test
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #dirPath = "/home/rosario/explainable/Bachelor/"# root
    #dirPath= "/home/rosario/explainable/test/Bachelor/"
    dirPath= "./" 

    modelsDirPath = dirPath+ "Models"

    print("calculating on: " +str(device))
    lr =0.1 # 0.001 slowed learningrate

    # load data
  
    trainloader ,random_indices_train, testloader,random_indices_test,X_train , X_test,  y_train , y_test, inputFeatures, outputFeatures, datasetName, featureNames= dataloader.load_kaggle_diabetes_dataset(batch_size=batch_size , droplist= droplist)
    #trainloader ,random_indices_train, testloader,random_indices_test,X_train , X_test,  y_train , y_test, inputFeatures, outputFeatures, datasetName, featureNames= dataloader.BreastCancerUCI(batch_size= batch_size, droplist=droplist, test_size=test_size)
    #trainloader ,random_indices_train, testloader,random_indices_test,X_train , X_test,  y_train , y_test, inputFeatures, outputFeatures, datasetName, featureNames= dataloader.dryBeanUCI(batch_size=batch_size , droplist= droplist)
    
    #model = modelClass.Net(inputFeatures= inputFeatures, out_features=outputFeatures)
    model= modelClass.BinaryClassification2HL64N(inputFeatures= inputFeatures, outputFeatures= outputFeatures)
    modelName = model.modelName
    
    #print(random_indices_test)

    #for i,c in testloader:
    #    print(i[0])
    #    print(X_test[random_indices_test[0]])
    #    break
    
    # Backward Propergation - loss and optimizer
    loss_function = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    optimizer = torch.optim.Adam(model.parameters())
    #grads = train.train(trainloader, model, num_epochs, device, y_train,loss_function, optimizer)    
    grads =  train.train(trainloader,random_indices_train, testloader,random_indices_test, model, num_epochs, device, y_train, y_test, loss_function, optimizer)
 
    #train.train(trainloader,random_indices_train, testloader,random_indices_test, model, num_epochs, device, y_train, y_test, loss_function, optimizer)
    print(dirPath)

    import eval
    import plotResults
    from matplotlib import pyplot as plt
    print("evaluating ...")
    loaderList = [testloader]
    nameList = ["test"]
    yList = [y_test]
    eval.doALLeval(model, modelsDirPath, dirPath, loaderList, device,optimizer, loss_function, num_epochs, nameList, yList, inputFeatures, random_indices_test)
    print(dirPath)
    print(modelsDirPath)
    ##
    dataPath= dirPath+ "Trainingresults/"
print("plotting...")
import utils 
import numpy as np

#unpackedGradiends = utils.unpackingGradients(inputFeatures, grads)
#averagedGradientMagnitude = np.average(np.absolute(unpackedGradiends), axis=0) 
#gradientMagnitudePerFeature = np.absolute(unpackedGradiends)
#utils.appendToNPZ(dirPath+ "data.npz", "test" + "GradientsPerFeature", unpackedGradiends)
#utils.appendToNPZ(dirPath+ "data.npz", "test" + "GradientMagnitudePerFeature", gradientMagnitudePerFeature)
#utils.appendToNPZ(dirPath+ "data.npz", "test" + "AveragedGradientMagnitude", averagedGradientMagnitude)
#plotResults.plotCosineSimilarity(dirPath, "cosine_simialarity", set="train")
#plotResults.plotCosineSimilarity(dirPath, "cosine_simialarity", set="eval")
print("cosine_similarity")
plotResults.plotCosineSimilarity(dataPath, "cosine_simialarity", set="test")
#plt.show()
#plotResults.plotWeightSignDifferences(dirPath, "percentageWeightsSignDifference1" , "train")
#plotResults.plotWeightSignDifferences(dirPath, "percentageWeightsSignDifference2" , "eval")
print("percentageWeightsSignDifference3")
plotResults.plotWeightSignDifferences(dataPath, "percentageWeightsSignDifference3" , "test")
#plt.show()
print("weightsMagnitude3")
#plotResults.plotWeightMagnitude(dirPath, "weightsMagnitude1","train")
#plotResults.plotWeightMagnitude(dirPath, "weightsMagnitude2","eval")
plotResults.plotWeightMagnitude(dataPath, "weightsMagnitude3","test")
#plt.show()
print("L2Distance3")
#plotResults.plotL2Distance(dirPath, "L2Distance1","train")
#plotResults.plotL2Distance(dirPath, "L2Distance2","eval")
plotResults.plotL2Distance(dataPath, "L2Distance3","test")
#plt.show()
print("weightTrace3")
#plotResults.plotWeightTrace(dirPath, "weightTrace1","train")
#plotResults.plotWeightTrace(dirPath, "weightTrace2","eval")
plotResults.plotWeightTrace(dataPath, "weightTrace3","test")    
#plt.show()
#plotResults.plotGradientsPerFeature(dirPath,"gradientsPerFeature1" ,False)
#plotResults.plotGradientsPerFeature(dirPath,"gradientsPerFeature2" ,True )
#plt.show()
print("averageGradientMagnitude3")
#plotResults.plotGradientMagnitude(dirPath, "averageGradientMagnitude1","train", perFeature=False)
#plotResults.plotGradientMagnitude(dirPath, "averageGradientMagnitude2","eval", perFeature=False)
plotResults.plotGradientMagnitude(dataPath, "averageGradientMagnitude3","test", perFeature=False)
#plt.show()
print("GradientMagnitudePerFeature3")
#plotResults.plotGradientMagnitude(dirPath, "GradientMagnitudePerFeature1","train", perFeature=True)
#plotResults.plotGradientMagnitude(dirPath, "GradientMagnitudePerFeature2","eval", perFeature=True)
plotResults.plotGradientMagnitude(dataPath, "GradientMagnitudePerFeature3","test", perFeature=True)
#plt.show()

figAcc, axsAcc = plt.subplots(nrows=1, ncols=1)

data = utils.loadData(dataPath+ "data.npz")

axsAcc.set_title("testAccuracyPerIteration")
axsAcc.set_xlabel("iteration")
axsAcc.set_ylabel("accuracy")
axsAcc.plot(data["testAccPerIterationList"])
figAcc.savefig(dataPath + "testAccuracyPerIteration")
pickle.dump(figAcc, open(dataPath + "testAccuracyPerIteration", 'wb'))
#figAcc.show()

#plotResults.plotLoss_Acc(dirPath,"loss_acc1", False)
#plotResults.plotLoss_Acc(dirPath,"loss_acc2",True)
#plt.show()
#print("confusionMatrix3")
#plotResults.plotConfusionMatrix(dirPath, "confusionMatrix1", set="train")
#plotResults.plotConfusionMatrix(dirPath, "confusionMatrix2", set="eval")
#plotResults.plotConfusionMatrix(dataPath, "confusionMatrix3", set="test")
#plt.show()

import cega_utils

#data 
trainedModelPrediction_Test = model.predict(X_test.to("cuda:0"))
                                    # data   
#print(trainedModelPrediction_Test)
cega_utils.calculateAndSaveOHE_Rules(X_test, featureNames,trainedModelPrediction_Test, data["testGradientsPerSamplePerFeature_iteration"], debug= False) #OHEresults


import warnings
warnings.filterwarnings('ignore')
#     frequent_itemsets = apriori(basket_sets.astype('bool'), min_support=0.07, use_colnames=True) https://stackoverflow.com/questions/74114745/how-to-fix-deprecationwarning-dataframes-with-non-bool-types-result-in-worse-c
debug = False

import os 
from datetime import datetime

pos_label = '1'
neg_label = '0'


rulesResultDataPath = dirPath + "rulesResultData/" 

featureDict= {'Pregnancies':0, 'Glucose':1, 'BloodPressure':2, 'SkinThickness':3, 'Insulin':4, \
              'BMI':5, 'DiabetesPedigreeFunction':6, 'Age':7}

# Get the current date and time
now = datetime.now()

# Format the date and time as a string without leading zeros
#date_time_string = now.strftime("%Y-%-m-%-d %-H:%-M:%-S")

date_time_string = now.strftime("%Y-%m-%d %H:%M:%S")
# Replace space with underscore
date_time_string = date_time_string.replace(" ", "_")

discriminative_rules_overIterations = []
charachteristic_rules_overIterations = []
#
rules_list_overIterations   = []
labelList_rules_overIterations = []
rulePrecisionList_overIterations =[]
predictionComparisonList_overIterations = []
rulesComplexityList_overIterations = []
coverageList_overIterations = []
ruleSupportList_overIterations = []
numberOfGeneratedRules_overIterations = []
jaccardSimilarity_overIterations = []

tempRules_list = None
from tqdm import tqdm
for i in tqdm(range(len(os.listdir("./OHEresults/")))):
    ohe_df = cega_utils.loadOHE_Rules(i)
    all_rules, pos_rules , neg_rules =  cega_utils.runApriori(ohe_df,len(X_test), pos_label ,neg_label)
    discriminative_rules = cega_utils.getDiscriminativeRules(all_rules, pos_label, neg_label )
    charachteristic_rules = cega_utils.getCharasteristicRules(pos_rules, pos_label, neg_rules,neg_label )
    
    resultName = "discriminative_rules"
    #resultName = "charachteristic_rules"
    #rules_list, labelList_rules, rulePrecisionList, predictionComparisonList, rulesComplexityList , coverageList,  ruleSupportList,   numberOfGeneratedRules,  =cega_utils.calculateRulesMetrics(discriminative_rules, resultName ,featureDict, testloader, trainedModelPrediction_Test, rulesResultDataPath)
    rules_list, labelList_rules, rulePrecisionList, predictionComparisonList, rulesComplexityList , coverageList,  ruleSupportList,   numberOfGeneratedRules,  =cega_utils.calculateRulesMetrics(discriminative_rules, featureDict, testloader, trainedModelPrediction_Test)
    #resultName = "charachteristic_rules"
    #rules_list, labelList_rules, rulePrecisionList, predictionComparisonList, rulesComplexityList , coverageList,  ruleSupportList,  = numberOfGeneratedRules,  =cega_utils.calculateRulesMetrics(charachteristic_rules, resultName ,featureDict, testloader, trainedModelPrediction_Test, rulesResultDataPath, debug=True )
    discriminative_rules_overIterations.append(discriminative_rules)
    charachteristic_rules_overIterations.append(charachteristic_rules) 
    #
    #print(rules_list)
    rules_list_overIterations.append(rules_list)
    labelList_rules_overIterations.append(labelList_rules)
    
    rulePrecisionList_overIterations.append(rulePrecisionList)
    #print(rulePrecisionList_overIterations)
    predictionComparisonList_overIterations.append(predictionComparisonList)
    rulesComplexityList_overIterations.append(rulesComplexityList)
    coverageList_overIterations.append(coverageList)
    ruleSupportList_overIterations.append(ruleSupportList)
    numberOfGeneratedRules_overIterations.append(numberOfGeneratedRules)

    if tempRules_list is not None:
        jaccardSimilarity_overIterations.append(cega_utils.jaccard_similarity(rules_list , tempRules_list))
    tempRules_list = rules_list

if debug:
    pathToNPZ =  dirPath + f"DEBUG.npz"
else:    
    pathToNPZ =  dirPath +"rulesResults/" f"{resultName}/_{date_time_string}.npz"
print(pathToNPZ)
np.savez(pathToNPZ ,rules_list_overIterations = rules_list_overIterations) 
utils.appendToNPZ(pathToNPZ, "labelList_rules_overIterations", labelList_rules_overIterations)
utils.appendToNPZ(pathToNPZ, "rulePrecisionList_overIterations", rulePrecisionList_overIterations)
utils.appendToNPZ(pathToNPZ, "predictionComparisonList_overIterations", predictionComparisonList_overIterations)
utils.appendToNPZ(pathToNPZ, "rulesComplexityList_overIterations", rulesComplexityList_overIterations)
utils.appendToNPZ(pathToNPZ, "coverageList_overIterations", coverageList_overIterations)
utils.appendToNPZ(pathToNPZ, "ruleSupportList_overIterations", ruleSupportList_overIterations)
utils.appendToNPZ(pathToNPZ, "numberOfGeneratedRules_overIterations", numberOfGeneratedRules_overIterations)
utils.appendToNPZ(pathToNPZ, "jaccardSimilarity_overIterations", jaccardSimilarity_overIterations)

#utils.appendToNPZ(rules_data)
    #charachteristic_rules
x = utils.loadData(pathToNPZ)
for i in x:
    print(i)

#x["rulePrecisionList_overIterations"]

rules_data = np.load(pathToNPZ , allow_pickle=True)

import statistics


def calculate_mean_of_lists(list_of_lists):
    means = []
    for sublist in list_of_lists:
        try:
            sublist_mean = statistics.mean(sublist)
            means.append(sublist_mean)
        except statistics.StatisticsError:
            means.append(0)  # or any other value to indicate the empty sublist
    return means

import pickle5 as pickle

#plot
pathToDiscriminative_rules = "/home/rosario/explainable/test/Bachelor/rulesResults/discriminative_rules/"
pathToCharachteristic_rules = "/home/rosario/explainable/test/Bachelor/rulesResults/charachteristic_rules"
resultPaths_dicriminative_rules = os.listdir(pathToDiscriminative_rules)
resultPaths_charachteristic_rules = os.listdir(pathToCharachteristic_rules)

print(resultPaths_dicriminative_rules)
resultPaths_dicriminative_rules= np.sort(resultPaths_dicriminative_rules)
print(resultPaths_dicriminative_rules[-1])

mostRecentResultPaths_discriminative = pathToDiscriminative_rules + (resultPaths_dicriminative_rules[-1])
print(mostRecentResultPaths_discriminative)

#/home/rosario/explainable/test/Bachelor/rulesResults/discriminative_rules/_2023-06-07 12:18:40.npz

data = utils.loadData(mostRecentResultPaths_discriminative)
#print(data["rulePrecisionList_overIterations"])
#rules_list_overIterations
#labelList_rules_overIterations
#rulePrecisionList_overIterations
#predictionComparisonList_overIterations
#rulesComplexityList_overIterations                  )

pathToRulesResults = "/home/rosario/explainable/test/Bachelor/rulesResults/"

#plt.show()
fig1, axs1 = plt.subplots(nrows=1, ncols=1)


axs1.plot(calculate_mean_of_lists(data["rulePrecisionList_overIterations"]))
axs1.set_title("rulePrecisionList_overIterations")
axs1.set_xlabel("iteration")
axs1.set_ylabel("precision")

#pickle_file_path = str(pathToRulesResults) + "ruleSupportList_overIterations"
fig1.savefig(str(pathToRulesResults) + "rulePrecisionList_overIterations")    
pickle.dump(fig1, open(pathToRulesResults + "rulePrecisionList_overIterations", 'wb'))
#fig1.show()

fig2, axs2 = plt.subplots(nrows=1, ncols=1)

axs2.plot(calculate_mean_of_lists(data["ruleSupportList_overIterations"]))
#pickle_file_path = str(pathToRulesResults) + "ruleSupportList_overIterations"
axs2.set_title("ruleSupportList_overIterations")
axs2.set_xlabel("iteration")
axs2.set_ylabel("support")

fig2.savefig(str(pathToRulesResults) + "ruleSupportList_overIterations")    
pickle.dump(fig2, open(pathToRulesResults + "ruleSupportList_overIterations", 'wb'))
#fig2.show()

fig3, axs3 = plt.subplots(nrows=1, ncols=1)

axs3.plot(calculate_mean_of_lists(data["rulesComplexityList_overIterations"]))
axs3.set_title("rulesComplexityList_overIterations")
axs3.set_xlabel("iteration")
axs3.set_ylabel("complexity")

fig3.savefig(str(pathToRulesResults) + "rulesComplexityList_overIterations")    
pickle.dump(fig3, open(pathToRulesResults + "rulesComplexityList_overIterations", 'wb'))
#fig3.show()

fig4, axs4 = plt.subplots(nrows=1, ncols=1)


axs4.plot(data["coverageList_overIterations"])
axs4.set_title("rulesComplexityList_overIterations")
axs4.set_xlabel("iteration")
axs4.set_ylabel("coverage")


fig4.savefig(str(pathToRulesResults) + "coverageList_overIterations")    
pickle.dump(fig4, open(pathToRulesResults + "coverageList_overIterations", 'wb'))
#fig4.show()


fig5, axs5 = plt.subplots(nrows=1, ncols=1)
axs5.plot(data["numberOfGeneratedRules_overIterations"])
axs5.set_title("numberOfGeneratedRules_overIterations")
axs5.set_xlabel("iteration")
axs5.set_ylabel("numGeneratedRules")

fig5.savefig(str(pathToRulesResults) + "numberOfGeneratedRules_overIterations")    
pickle.dump(fig5, open(pathToRulesResults + "numberOfGeneratedRules_overIterations", 'wb'))
#fig5.show()

fig6, axs6 = plt.subplots(nrows=1, ncols=1)


axs6.plot(data["jaccardSimilarity_overIterations"])
fig6.savefig(str(pathToRulesResults) + "jaccardSimilarity_overIterations") 
axs6.set_title("jaccardSimilarity_overIterations")
axs6.set_xlabel("iteration")
axs6.set_ylabel("similarity")

pickle.dump(fig6, open(pathToRulesResults + "jaccardSimilarity_overIterations", 'wb'))

#fig6.show()