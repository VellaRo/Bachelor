from matplotlib import pyplot as plt
import pickle #figx = pickle.load(open('FigureObject.fig.pickle', 'rb')) 
              #figx.show() # Show the figure, edit it, etc.!
import sklearn
import seaborn as sns
import numpy as np
import utils

"""
TODO ADD TITLE AND AJUST DESCRIPTION...

"""
def plotConfusionMatrix(dirPath, plotName, set):

    """
    plots the Confusionmatrix for trainingSet or testSet  or evalSet

    saves the plot as png
    saves the plot as pickel( for interactive plot in dataDiscovery)

    parameters:
        dirPath: where to save the plots

        plotName: how is the plot named (name to save plot ) 

        set : chose a data set from "train" , "test", "eval

    returns: None
    """
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4,4) )

    data = utils.loadData(dirPath +"data.npz")
    if set == "train":
        y = data["y_train"]
        predictions = data["trainPredictionList"]
        acc = data["trainAccPerEpochList"]
    elif set == "eval":
        y = data["y_eval"]
        predictions = data["evalPredictionList"]
        acc = data["evalAccPerEpochList"]
    elif set == "test":
        y = data["y_test"]
        acc = data["testAccPerEpochList"]
        predictions = data["testPredictionList"]

    cm = sklearn.metrics.confusion_matrix(y,predictions)

    sns.heatmap(cm, linewidth=0.5,ax=axs)
    axs.set_xlabel('Actual')
    axs.set_ylabel('Predicted')
    axs.set_title('accuracy: ' + str(acc[-1]))

    fig.savefig(str(dirPath) + plotName)
    pickle.dump(fig, open(str(dirPath) + plotName+'.pickle', 'wb')) 
    return None

#CLEANED
def plotLoss_Acc(dirPath,plotName, separatly =False):
    """
    plots the Loss and Accuracy over time(epochs/batches) for trainingSet and testSet 

    saves the plot as png
    saves the plot as pickel( for interactive plot)

    parameters:
        dirPath: where to save the plots

        plotName: how is the plot named (name to save plot ) 

        separatly:
            False: plot all in one figure
            True:  plot all in separate figures 


    returns: None

    """
    data = utils.loadData(dirPath +"data.npz")
    trainAccPerEpochList =     data["trainAccPerEpochList"]
    trainAccPerIterationList = data["trainAccPerIterationList"]
    trainLossPerEpochList =    data["trainLossPerEpochList"]
    trainLossPerIterationList =data["trainLossPerIterationList"]

    #evalAccPerEpochList =     data["evalAccPerEpochList"]
    #evalAccPerIterationList = data["evalAccPerIterationList"]
    #evalLossPerEpochList =    data["evalLossPerEpochList"]
    #evalLossPerIterationList =data["evalLossPerIterationList"]

    testAccPerEpochList =     data["testAccPerEpochList"]
    testAccPerIterationList = data["testAccPerIterationList"]
    testLossPerEpochList =    data["testLossPerEpochList"]
    testLossPerIterationList =data["testLossPerIterationList"]


    stackedInputs = [[trainAccPerEpochList ,trainAccPerIterationList, trainLossPerEpochList,trainLossPerIterationList], 
                                      [testAccPerEpochList ,testAccPerIterationList, testLossPerEpochList,testLossPerIterationList],]
    #                 [evalAccPerEpochList ,evalAccPerIterationList, evalLossPerEpochList,evalLossPerIterationList],
   
    if not(separatly):

        print("plotting LA: Loss_accuracy")
        figLA, axsLA = plt.subplots(nrows=3, ncols=4, figsize=(3,4) )
        figLA.tight_layout(pad=0.5)
        for i in range(len(stackedInputs)):
            for j in range(len(stackedInputs[i])):
                name = ""
                if i == 0:
                    name = "train"
                elif i ==1:
                    name == "eval"
                elif i==2:
                    name == "test"
                axsLA[i][j].plot(range(len(stackedInputs[i][j])),stackedInputs[i][j]) 
                if j == 0 or j ==1:
                    axsLA[i][j].set_ylabel('accuracy')
                    name += "Accuracy"
                elif j == 2 or j == 3:
                    axsLA[i][j].set_ylabel('loss')
                    name += "Loss"
                if j == 0 or j == 2:
                    axsLA[i][j].set_xlabel('epoch')
                    name += "PerEpoch"
                elif j ==1 or j ==3:
                    axsLA[i][j].set_xlabel('iteration')
                    name += "PerIteration"
                axsLA[i][j].set_title(str(name))

        figLA.savefig(dirPath +plotName)
        pickle.dump(figLA, open(dirPath + plotName +'.pickle', 'wb')) 

    else:
        print("plotting LA: Loss_accuracy")
        for i in range(len(stackedInputs)):
            for j in range(len(stackedInputs[i])):
                exec("figLA" +str(i) + str(j)+ ", axsLA" +str(i) + str(j) +"= plt.subplots(nrows=1, ncols=1, figsize=(3,4) )")
                
                name = ""
                if i == 0:
                    name = "train"
                elif i ==1:
                    name == "eval"
                elif i==2:
                    name == "test"
                exec("axsLA" +str(i) + str(j)+".plot(range(len(stackedInputs[i][j])),stackedInputs[i][j])") 
                if j == 0 or j ==1:
                    exec("axsLA" +str(i) + str(j)+ ".set_ylabel('accuracy')")
                    name += "Accuracy"
                elif j == 2 or j == 3:
                    exec("axsLA" +str(i) + str(j)+ ".set_ylabel('loss')")
                    name += "Loss"
                if j == 0 or j == 2:
                    exec("axsLA" +str(i) + str(j)+ ".set_xlabel('epoch')")
                    name += "PerEpoch"
                elif j ==1 or j ==3:
                    exec("axsLA" +str(i) + str(j)+ ".set_xlabel('iteration')")
                    name += "PerIteration"
                
                exec("axsLA" +str(i) + str(j)+ ".set_title(str(name))")

        exec("figLA" +str(i) + str(j)+".savefig(dirPath +plotName + str(i)+ str(j))")
        exec("pickle.dump(figLA" +str(i) + str(j)+", open(dirPath + plotName + str(i)+ str(j) +'.pickle', 'wb'))")

    return None

#CLEANED # do also as set == ... ?
def plotGradientsPerFeature(dirPath, plotName , sepratly=False):
    """
    plots the gradients for each feature  

    saves the plot as png
    saves the plot as pickel( for interactive plot in dataDiscovery)
    
    parameters:
        dirPath: where to save the plots

        plotName: how is the plot named (name to save plot ) 

        separatly:
            False: plot all in one figure
            True:  plot all in separate figures 
    
    returns: None
    """
    # Load data
    data = utils.loadData(dirPath +"data.npz")
    trainGradientPerFeature = data["trainGradientsPerFeature"]
    evalGradientPerFeature = data["evalGradientsPerFeature"]
    inputFeatures = data["inputFeatures"]
    inputFeatures = inputFeatures.item()
    tempStacked = [trainGradientPerFeature, evalGradientPerFeature]

    #find maximum/minumum 

    maxValue= 0
    minValue= 0
    #calculate metrics
    for i in range(len(trainGradientPerFeature)):

        maxValueT = max( maxValue, max(trainGradientPerFeature[i]))    
        maxValueE = max( maxValue, max(evalGradientPerFeature[i]))
        minValueT = min( minValue, min(trainGradientPerFeature[i]))
        minValueE = min( minValue, min(evalGradientPerFeature[i]))
        maxValue  = max(maxValueT, maxValueE)
        minValue  = min(minValueT, minValueE)
    #plot for all features

    if not(sepratly):
        figGPF, axsGPF = plt.subplots(inputFeatures,2) 

        figGPF.set_size_inches(10,15)
        figGPF.tight_layout(pad=2.5)
        for i in range(len(axsGPF)):
            for j in range(len(tempStacked)):
                axsGPF[i][j].set_ylim(minValue,maxValue)
            
                axsGPF[i][j].set_xlabel("iteration")
                axsGPF[i][j].set_ylabel("gradients")

                axsGPF[i][j].plot(tempStacked[j][i])
        
        plt.savefig(str(dirPath) +str(plotName) )
        pickle.dump(figGPF, open(str(dirPath) + str(plotName), 'wb')) 

    elif sepratly:
        for i in range(len(tempStacked)):
            exec("figGPF" +str(i) +", axsGPF"+str(i)+ "= plt.subplots(inputFeatures)")
            for j in range(len(tempStacked[i])):
                exec("axsGPF"+str(i)+"[j]"+ ".set_ylim(minValue,maxValue)")
                exec("axsGPF"+str(i)+"[j]"+ ".set_xlabel('iteration')")
                exec("axsGPF"+str(i)+"[j]"+ ".set_ylabel('gradients')")
                exec("axsGPF"+str(i)+"[j]"+ ".plot(tempStacked[i][j])")
        
        exec("figGPF"+str(i)+".savefig(str(dirPath) +str(plotName) +str(i))")
        exec("pickle.dump(figGPF"+str(i)+", open(str(dirPath) + str(plotName) +str(i), 'wb'))") 

    return None

# NOT CLEANED YET ALSO NOT FIXED FOR SHUFFELED VALUES
def plotGradientsPerSample(featureListAll, num_epochs,data):
    
    print("SHUFFLE NEEDS TO BE FALSE AT THE MOMENT") 
    featureListAll = np.array(featureListAll)

    def reshapeTogradientsPerSample():

        def loopthroughFeature(index):
            samplesPerFeature = []
            for i in range(len(data)):

                tempSamples = []
                for j in range(i, num_epochs*len(data),len(data)):
                
                    tempSamples.append(featureListAll[index][j])

                samplesPerFeature.append(tempSamples)
            return samplesPerFeature
        
        gradientsPerSample = []
        for k in range(len(featureListAll)):
            gradientsPerSample.append(loopthroughFeature(k))
        
        return gradientsPerSample
    
    gradientsPerSample = reshapeTogradientsPerSample()
    def plotSubplots(index):
        fig, axs = plt.subplots(nrows=len(featureListAll), ncols=2 )
        fig1, axs1 = plt.subplots(nrows=len(featureListAll), ncols=2 )

        for i in range(len(featureListAll)):

            axs1[i][0].plot(gradientsPerSample[i][index])
            axs1[i][1].plot(gradientsPerSample[i][index+1])

    howManySamplesToLookAt = 2

    for i in range(0,howManySamplesToLookAt):
        plotSubplots(i)
        ("press a button for the next Sample")
        w = plt.waitforbuttonpress()

#CLEANED
def plotCosineSimilarity(dirPath, plotName, set):
    """
    plots the cosine_Similarity for trainingSet or testSet 

    saves the plot as png
    saves the plot as pickel( for interactive plot in dataDiscovery)

    parameters:
        dirPath: where to save the plots

        plotName: how is the plot named (name to save plot ) 

        set : chose a data set from "train" , "test", "eval
        
    returns: None
    """
    fig, axs = plt.subplots(nrows=1, ncols=1)
    data = utils.loadData(dirPath +"data.npz")

    if set == "train":
        cosine_similarity_toInitialList= data["trainCosine_similarity_toInitialList"]    
        cosine_similarity_toFinalList= data["trainCosine_similarity_toFinalList"]
    elif set == "eval":
        cosine_similarity_toInitialList= data["evalCosine_similarity_toInitialList"]    
        cosine_similarity_toFinalList= data["evalCosine_similarity_toFinalList"]
    elif set == "test":
        cosine_similarity_toInitialList= data["testCosine_similarity_toInitialList"]    
        cosine_similarity_toFinalList= data["testCosine_similarity_toFinalList"]


    axs.plot(range(len(cosine_similarity_toInitialList)),cosine_similarity_toInitialList)
    axs.plot(range(len(cosine_similarity_toFinalList)),cosine_similarity_toFinalList )
    fig.savefig(str(dirPath) + str(plotName))
    pickle.dump(fig, open(str(dirPath) + str(plotName), 'wb'))

    return None

#CLEANED
def plotWeightSignDifferences(dirPath, plotName, set):
    """
    plots the percentage of weight diffences compared to the initial and final weights for trainingSet or testSet 

    saves the plot as png
    saves the plot as pickel( for interactive plot in dataDiscovery)

    parameters:
        dirPath: where to save the plots

        plotName: how is the plot named (name to save plot ) 

        set : chose a data set from "train" , "test", "eval
        
    returns: None
    """
    data = utils.loadData(dirPath +"data.npz")    
    if set == "train":
        percentageWeightSignDifferences_toInitialList = data["trainPercentageWeightSignDifferences_toInitialList"]
        percentageWeightSignDifferences_toFinalList = data["trainPercentageWeightSignDifferences_toFinalList"]
    elif set == "eval":
        percentageWeightSignDifferences_toInitialList = data["evalPercentageWeightSignDifferences_toInitialList"]
        percentageWeightSignDifferences_toFinalList = data["evalPercentageWeightSignDifferences_toFinalList"]
    elif set == "test":
        percentageWeightSignDifferences_toInitialList = data["testPercentageWeightSignDifferences_toInitialList"]
        percentageWeightSignDifferences_toFinalList = data["testPercentageWeightSignDifferences_toFinalList"]


    fig, axs = plt.subplots(nrows=1, ncols=1)

    axs.plot(percentageWeightSignDifferences_toInitialList) 
    axs.plot(percentageWeightSignDifferences_toFinalList) 
    
    fig.savefig(str(dirPath) + str(plotName))    
    pickle.dump(fig, open(str(dirPath) + str(plotName), 'wb'))

    return None

#CLEANED
def plotWeightMagnitude(dirPath, plotName, set):
    """
    plots the weight magnitude(absolute value of gradients) for trainingSet and testSet 

    saves the plot as png
    saves the plot as pickel( for interactive plot in dataDiscovery)

    parameters:
        dirPath: where to save the plots

        plotName: how is the plot named (name to save plot ) 

        set : chose a data set from "train" , "test", "eval
        
    returns: None
    """

    data = utils.loadData(dirPath +"data.npz")
    if set == "train":
        absoluteIterationWeightsList = data["trainAbsoluteIterationWeightsList"]
    elif set == "eval":
        absoluteIterationWeightsList = data["evalAbsoluteIterationWeightsList"]
    elif set == "test":
        absoluteIterationWeightsList = data["testAbsoluteIterationWeightsList"]


    
    fig, axs = plt.subplots(nrows=1, ncols=1)
    axs.plot(absoluteIterationWeightsList)

    fig.savefig(str(dirPath) + str(plotName))    
    pickle.dump(fig, open(str(dirPath) + str(plotName), 'wb'))
    
    return None

#CLEANED
def plotL2Distance(dirPath, plotName, set):
    """
    plots the L2 distance for trainingSet and testSet 

    saves the plot as png
    saves the plot as pickel( for interactive plot in dataDiscovery)

    parameters:
        dirPath: where to save the plots

        plotName: how is the plot named (name to save plot ) 

        set : chose a data set from "train" , "test", "eval
        
    returns: None
    """
    data = utils.loadData(dirPath +"data.npz")
    if set == "train":
        l2Dist_toInitialList = data["trainL2Dist_toInitialList"]
        l2Dist_toFinalList = data["trainL2Dist_toFinalList"]
    elif set == "eval":
        l2Dist_toInitialList = data["evalL2Dist_toInitialList"]
        l2Dist_toFinalList = data["evalL2Dist_toFinalList"]
    elif set == "test":
        l2Dist_toInitialList = data["testL2Dist_toInitialList"]
        l2Dist_toFinalList = data["testL2Dist_toFinalList"]


    fig, axs = plt.subplots(nrows=1, ncols=1)

    axs.plot(l2Dist_toInitialList) 
    axs.set
    axs.plot(l2Dist_toFinalList)
    
    fig.savefig(str(dirPath) + str(plotName))    
    pickle.dump(fig, open(str(dirPath) + str(plotName), 'wb'))

    return None

#CLEANED
def plotWeightTrace(dirPath, plotName, set):
    """
    plots the weight trace for 10Random weights for trainingSet and testSet 

    saves the plot as png
    saves the plot as pickel( for interactive plot in dataDiscovery)

    parameters:
        dirPath: where to save the plots

        plotName: how is the plot named (name to save plot ) 

        set : chose a data set from "train" , "test", "eval
        
    returns: None
    """
    data = utils.loadData(dirPath +"data.npz")
    if set == "train":
        random10WeightsList = data["trainRandom10WeightsList"]
    elif set == "eval":
        random10WeightsList = data["evalRandom10WeightsList"]
    elif set == "test":
        random10WeightsList = data["testRandom10WeightsList"]


        
    fig, axs = plt.subplots(nrows=1, ncols=1)

    for i in random10WeightsList:
        axs.plot(i)
        axs.set_title("weightTrace of 10 Random weights")
        axs.set_xlabel("iteration")
        axs.set_ylabel("weightValue") 

    fig.savefig(str(dirPath) + str(plotName))
    pickle.dump(fig, open(str(dirPath) + str(plotName), 'wb'))

    return None

#CLEANED # also do with set ...? 

def plotTotalGradientMagnitude(total_gradientsList,dirPath, plotName):#, set):
    print("plotting: total gradient magnitude Averaged over number of grads in parameters")
    figGM, axsGM = plt.subplots(nrows=1, ncols=1)
    axsGM.plot(total_gradientsList)
    axsGM.set_xlabel('iteration')
    axsGM.set_ylabel('totalGradientMagnitude')
    axsGM.set_title("gradient magnitude Averaged over number of grads")
    figGM.savefig(str(dirPath) + plotName+ "Averaged")
    pickle.dump(figGM , open(str(dirPath) + str(plotName)+ "Averaged", 'wb'))


def plotGradientMagnitude(dirPath, plotName, set, perFeature):
    """
    plots the cosine_Similarity for trainingSet and testSet 

    saves the plot as png
    saves the plot as pickel( for interactive plot in dataDiscovery)

    parameters:
        dirPath: where to save the plots

        plotName: how is the plot named (name to save plot ) 

        set : chose a data set from "train" , "test", "eval
        
        perFeature : 
                    True: plots for each feature separatly
                    False : averages across the features
    returns: None
    """
    data = utils.loadData(dirPath +"data.npz")

    if set == "train":
        averagedAbsoluteGradientMagnitude =  data["trainAveragedGradientMagnitude"]
        absoluteabsoluteGradientMagnitudePerFeature =  data["trainGradientMagnitudePerFeature"]

    elif set == "eval": 
        averagedAbsoluteGradientMagnitude = data["evalAveragedGradientMagnitude"]
        absoluteabsoluteGradientMagnitudePerFeature = data["evalGradientMagnitudePerFeature"]
    elif set == "test":
        averagedAbsoluteGradientMagnitude = data["testAveragedGradientMagnitude"]
        absoluteabsoluteGradientMagnitudePerFeature = data["testGradientMagnitudePerFeature"]

    
    if not(perFeature):
        print("plotting: GM GradientMagnitude averaged over features")
        figGM, axsGM = plt.subplots(nrows=1, ncols=1)

        axsGM.plot(averagedAbsoluteGradientMagnitude)
        axsGM.set_xlabel('iteration')
        axsGM.set_ylabel('gradientMagnitude')
        axsGM.set_title("GradientMagnitude averaged over features")

        figGM.savefig(str(dirPath) + plotName+ "Averaged")
        pickle.dump(figGM , open(str(dirPath) + str(plotName)+ "Averaged", 'wb'))

    else:  
 
        print("plotting: GM GradientMagnitude PerFeature")
        figGM, axsGM = plt.subplots(nrows=int(len(absoluteabsoluteGradientMagnitudePerFeature)), ncols=1)
        plt.title("GradientMagnitude per features")
        plt.xlabel('iteration') # asxGM
        plt.ylabel('gradientMagnitude')
        for i in range(len(absoluteabsoluteGradientMagnitudePerFeature)):
        
            axsGM[i].plot(absoluteabsoluteGradientMagnitudePerFeature[i])

        
        figGM.savefig(str(dirPath) + plotName)
        pickle.dump(figGM, open(str(dirPath) + str(plotName), 'wb'))

    return None

def plotRulesResults(data):
    
    def calculate_mean_of_lists(list_of_lists):
        means = []
        for sublist in list_of_lists:
            if len(sublist) == 0:
                means.append(-1)
            else:
                sublist_mean = np.mean(sublist)
                means.append(sublist_mean)
        return means


    pathToRulesResults = "./NLP_Results/rulesResults/"
    print("pathToRulesResults  " + pathToRulesResults)
    fig1, axs1 = plt.subplots(nrows=1, ncols=1)
    axs1.plot(calculate_mean_of_lists(data["rulePrecisionList_overIterations"]))
    axs1.set_title("rulePrecisionList_overIterations")
    axs1.set_xlabel("iteration")
    axs1.set_ylabel("precision")
    fig1.savefig(str(pathToRulesResults) + "rulePrecisionList_overIterations")    
    pickle.dump(fig1, open(pathToRulesResults + "rulePrecisionList_overIterations", 'wb'))
    
    fig2, axs2 = plt.subplots(nrows=1, ncols=1)
    axs2.plot(calculate_mean_of_lists(data["ruleSupportList_overIterations"]))
    axs2.set_title("ruleSupportList_overIterations")
    axs2.set_xlabel("iteration")
    axs2.set_ylabel("support")
    fig2.savefig(str(pathToRulesResults) + "ruleSupportList_overIterations")    
    pickle.dump(fig2, open(pathToRulesResults + "ruleSupportList_overIterations", 'wb'))
    
    fig3, axs3 = plt.subplots(nrows=1, ncols=1)
    axs3.plot(calculate_mean_of_lists(data["rulesComplexityList_overIterations"]))
    axs3.set_title("rulesComplexityList_overIterations")
    axs3.set_xlabel("iteration")
    axs3.set_ylabel("complexity")
    fig3.savefig(str(pathToRulesResults) + "rulesComplexityList_overIterations")    
    pickle.dump(fig3, open(pathToRulesResults + "rulesComplexityList_overIterations", 'wb'))
    
    fig4, axs4 = plt.subplots(nrows=1, ncols=1)
    axs4.plot(data["globalCoverageList_overIterations"])
    axs4.set_title("globalCoverageList_overIterations")
    axs4.set_xlabel("iteration")
    axs4.set_ylabel("coverage")
    fig4.savefig(str(pathToRulesResults) + "globalCoverageList_overIterations")    
    pickle.dump(fig4, open(pathToRulesResults + "globalCoverageList_overIterations", 'wb'))
    
    fig5, axs5 = plt.subplots(nrows=1, ncols=1)
    axs5.plot(data["numberOfGeneratedRules_overIterations"])
    axs5.set_title("numberOfGeneratedRules_overIterations")
    axs5.set_xlabel("iteration")
    axs5.set_ylabel("numGeneratedRules")
    fig5.savefig(str(pathToRulesResults) + "numberOfGeneratedRules_overIterations")    
    pickle.dump(fig5, open(pathToRulesResults + "numberOfGeneratedRules_overIterations", 'wb'))
    
    fig6, axs6 = plt.subplots(nrows=1, ncols=1)
    axs6.plot(data["jaccardSimilarity_overIterations"])
    fig6.savefig(str(pathToRulesResults) + "jaccardSimilarity_overIterations") 
    axs6.set_title("jaccardSimilarity_overIterations")
    axs6.set_xlabel("iteration")
    axs6.set_ylabel("similarity")
    pickle.dump(fig6, open(pathToRulesResults + "jaccardSimilarity_overIterations", 'wb'))
    
    fig7, axs7 = plt.subplots(nrows=1, ncols=1)
    axs7.plot(data["cosineSimilarity_overIterations"])
    axs7.set_title("cosineSimilarity_overIterations")
    axs7.set_xlabel("iteration")
    axs7.set_ylabel("similarity")
    fig7.savefig(str(pathToRulesResults) + "cosineSimilarity_overIterations")    
    pickle.dump(fig7, open(pathToRulesResults + "cosineSimilarity_overIterations", 'wb'))
    
    fig8, axs8 = plt.subplots(nrows=1, ncols=1)
    axs8.plot(data["diceSimilarity_overIterations"])
    axs8.set_title("diceSimilarity_overIterations")
    axs8.set_xlabel("iteration")
    axs8.set_ylabel("similarity")
    fig8.savefig(str(pathToRulesResults) + "diceSimilarity_overIterations")    
    pickle.dump(fig8, open(pathToRulesResults + "diceSimilarity_overIterations", 'wb'))
    
    fig9, axs9 = plt.subplots(nrows=1, ncols=1)
    axs9.plot(data["overlapSimilarity_overIterations"])
    axs9.set_title("overlapSimilarity_overIterations")
    axs9.set_xlabel("iteration")
    axs9.set_ylabel("similarity")
    fig9.savefig(str(pathToRulesResults) + "overlapSimilarity_overIterations")    
    pickle.dump(fig9, open(pathToRulesResults + "overlapSimilarity_overIterations", 'wb'))
    
    fig10, axs10 = plt.subplots(nrows=1, ncols=1)
    axs10.plot(data["numberOfGeneratedRulesRAW_overIterations"])
    axs10.set_title("numberOfGeneratedRulesRAW_overIterations")
    axs10.set_xlabel("iteration")
    axs10.set_ylabel("numberOfGeneratedRulesRAW_overIterations")
    fig10.savefig(str(pathToRulesResults) + "numberOfGeneratedRulesRAW_overIterations")    
    pickle.dump(fig10, open(pathToRulesResults + "numberOfGeneratedRulesRAW_overIterations", 'wb'))

    fig11, axs11 = plt.subplots(nrows=1, ncols=1)
    axs11.plot(calculate_mean_of_lists(data["rulePrecisionListPerRule_overIterations"])) 
    axs11.set_title("rulePrecisionListPerRule_overIterations")
    axs11.set_xlabel("iteration")
    axs11.set_ylabel("rulePrecisionListPerRule_overIterations")
    fig11.savefig(str(pathToRulesResults) + "rulePrecisionListPerRule_overIterations")    
    pickle.dump(fig1, open(pathToRulesResults + "rulePrecisionListPerRule_overIterations", 'wb'))

def plotTrainingResults(data, dataPath):
    
    print("plotting trainingResults...")
    print("cosine_similarity")
    plotCosineSimilarity(dataPath, "cosine_simialarity", set="test")
    print("percentageWeightsSignDifference3")
    plotWeightSignDifferences(dataPath, "percentageWeightsSignDifference3" , "test")
    print("weightsMagnitude3")
    plotWeightMagnitude(dataPath, "weightsMagnitude3","test")
    print("L2Distance3")
    plotL2Distance(dataPath, "L2Distance3","test")
    print("weightTrace3")
    plotWeightTrace(dataPath, "weightTrace3","test")    
    print("averageGradientMagnitude3")
    plotGradientMagnitude(dataPath, "averageGradientMagnitude3","test", perFeature=False)
    print("GradientMagnitudePerFeature3")
    plotGradientMagnitude(dataPath, "GradientMagnitudePerFeature3","test", perFeature=True)
    print("total_gradientsList")

    plotTotalGradientMagnitude(data["Total_gradientsList_iteration"],dataPath, "Total_gradientsList_iteration")#,"test") # save in data-npz   
    figAcc, axsAcc = plt.subplots(nrows=1, ncols=1)
    axsAcc.set_title("testAccuracyPerIteration")
    axsAcc.set_xlabel("iteration")
    axsAcc.set_ylabel("accuracy")
    axsAcc.plot(data["testAccPerIterationList"])
    figAcc.savefig(dataPath + "testAccuracyPerIteration")
    pickle.dump(figAcc, open(dataPath + "testAccuracyPerIteration", 'wb'))