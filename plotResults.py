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

    data = utils.loadData(dirPath)
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
    data = utils.loadData(dirPath)
    trainAccPerEpochList =     data["trainAccPerEpochList"]
    trainAccPerIterationList = data["trainAccPerIterationList"]
    trainLossPerEpochList =    data["trainLossPerEpochList"]
    trainLossPerIterationList =data["trainLossPerIterationList"]

    evalAccPerEpochList =     data["evalAccPerEpochList"]
    evalAccPerIterationList = data["evalAccPerIterationList"]
    evalLossPerEpochList =    data["evalLossPerEpochList"]
    evalLossPerIterationList =data["evalLossPerIterationList"]

    testAccPerEpochList =     data["testAccPerEpochList"]
    testAccPerIterationList = data["testAccPerIterationList"]
    testLossPerEpochList =    data["testLossPerEpochList"]
    testLossPerIterationList =data["testLossPerIterationList"]


    stackedInputs = [[trainAccPerEpochList ,trainAccPerIterationList, trainLossPerEpochList,trainLossPerIterationList],
                     [evalAccPerEpochList ,evalAccPerIterationList, evalLossPerEpochList,evalLossPerIterationList],
                     [testAccPerEpochList ,testAccPerIterationList, testLossPerEpochList,testLossPerIterationList],]
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
    data = utils.loadData(dirPath)
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
    data = utils.loadData(dirPath)

    if set == "train":
        cosine_similarity_toInitialList= data["trainCosine_similarity_toInitialList"]    
        cosine_similarity_toFinalList= data["trainCosine_similarity_toFinalList"]
    elif set == "eval":
        cosine_similarity_toInitialList= data["evalCosine_similarity_toInitialList"]    
        cosine_similarity_toFinalList= data["evalCosine_similarity_toFinalList"]
    elif set == "test":
        print("testSet does only get into consideration for Performance measure to be independet of the explanatory set")
        return None


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
    data = utils.loadData(dirPath)
    if set == "train":
        percentageWeightSignDifferences_toInitialList = data["trainPercentageWeightSignDifferences_toInitialList"]
        percentageWeightSignDifferences_toFinalList = data["trainPercentageWeightSignDifferences_toFinalList"]
    elif set == "eval":
        percentageWeightSignDifferences_toInitialList = data["evalPercentageWeightSignDifferences_toInitialList"]
        percentageWeightSignDifferences_toFinalList = data["evalPercentageWeightSignDifferences_toFinalList"]
    elif set == "test":
        print("testSet does only get into consideration for Performance measure to be independet of the explanatory set")
        return None

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

    data = utils.loadData(dirPath)
    if set == "train":
        absoluteIterationWeightsList = data["trainAbsoluteIterationWeightsList"]
    elif set == "eval":
        absoluteIterationWeightsList = data["evalAbsoluteIterationWeightsList"]
    elif set == "test":
        print("testSet does only get into consideration for Performance measure to be independet of the explanatory set")        

        return None
    
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
    data = utils.loadData(dirPath)
    if set == "train":
        l2Dist_toInitialList = data["trainL2Dist_toInitialList"]
        l2Dist_toFinalList = data["trainL2Dist_toFinalList"]
    elif set == "eval":
        l2Dist_toInitialList = data["evalL2Dist_toInitialList"]
        l2Dist_toFinalList = data["evalL2Dist_toFinalList"]
    elif set == "test":
        print("testSet does only get into consideration for Performance measure to be independet of the explanatory set")
        return None

    fig, axs = plt.subplots(nrows=1, ncols=1)

    axs.plot(l2Dist_toInitialList) 
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
    data = utils.loadData(dirPath)
    if set == "train":
        random10WeightsList = data["trainRandom10WeightsList"]
    elif set == "eval":
        random10WeightsList = data["evalRandom10WeightsList"]
    elif set == "test":
        print("testSet does only get into consideration for Performance measure to be independet of the explanatory set")
        return None
        
    fig, axs = plt.subplots(nrows=1, ncols=1)

    for i in random10WeightsList:
        axs.plot(i ) 

    fig.savefig(str(dirPath) + str(plotName))
    pickle.dump(fig, open(str(dirPath) + str(plotName), 'wb'))

    return None

#CLEANED # also do with set ...? 
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
    data = utils.loadData(dirPath)

    if set == "train":
        averagedAbsoluteGradientMagnitude =  data["trainAveragedGradientMagnitude"]
        absoluteabsoluteGradientMagnitudePerFeature =  data["trainGradientMagnitudePerFeature"]

    elif set == "eval": 
        averagedAbsoluteGradientMagnitude = data["evalAveragedGradientMagnitude"]
        absoluteabsoluteGradientMagnitudePerFeature = data["evalGradientMagnitudePerFeature"]
    elif set == "test":
        print("testSet does only get into consideration for Performance measure to be independet of the explanatory set")
        return None
    
    if not(perFeature):
        print("plotting: GM GradientMagnitude averaged over features")
        figGM, axsGM = plt.subplots(nrows=1, ncols=1)

        axsGM.plot(averagedAbsoluteGradientMagnitude)
        axsGM.set_xlabel('gradientMagnitude')
        axsGM.set_ylabel('iteration')
        axsGM.set_title("GradientMagnitude averaged over features")

        figGM.savefig(str(dirPath) + plotName+ "Averaged")
        pickle.dump(figGM , open(str(dirPath) + str(plotName)+ "Averaged", 'wb'))

    else:  
 
        print("plotting: GM GradientMagnitude PerFeature")
        figGM, axsGM = plt.subplots(nrows=int(len(absoluteabsoluteGradientMagnitudePerFeature)), ncols=1)

        for i in range(len(absoluteabsoluteGradientMagnitudePerFeature)):
        
            axsGM[i].plot(absoluteabsoluteGradientMagnitudePerFeature[i])
            axsGM[i].set_xlabel('gradientMagnitude')
            axsGM[i].set_ylabel('iteration')
            axsGM[i].set_title("GradientMagnitude per features")
        
        figGM.savefig(str(dirPath) + plotName+ "GradientMagnitude per feature")
        pickle.dump(figGM, open(str(dirPath) + str(plotName)+ "per feature", 'wb'))

    return None