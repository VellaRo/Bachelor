from matplotlib import pyplot as plt
import pickle #figx = pickle.load(open('FigureObject.fig.pickle', 'rb')) 
              #figx.show() # Show the figure, edit it, etc.!
import sklearn
import seaborn as sns
from torch import no_grad
import numpy as np
import eval
import utils

def plotConfusionMatrix(dirPath, plotName):

    """
    returns: None

    plots the Confusionmatrix for trainingSet and testSet 

    saves the plot as png
    saves the plot as pickel( for interactive plot in dataDiscovery)

    """
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(5,4) )

    data = utils.loadData(dirPath)
    
    y_train = data["y_train"]
    y_eval = data["y_eval"]
    y_test = data["y_test"]
    testPredictions = data["testPredictionList"]
    trainPredictions = data["trainPredictionList"]
    trainAcc = data["trainAccPerEpochList"]
    evalAcc = data["evalAccPerEpochList"]
    testAcc = data["testAccPerEpochList"]

    cmTest = sklearn.metrics.confusion_matrix(y_test,testPredictions)
    cmEval = sklearn.metrics.confusion_matrix(y_eval,testPredictions)
    cmTrain = sklearn.metrics.confusion_matrix(y_train,trainPredictions)


    sns.heatmap(cmTrain, linewidth=0.5,ax=axs[0])
    axs[0].set_xlabel('Actual')
    axs[0].set_ylabel('Predicted')
    axs[0].set_title('Train_Acc: ' + str(trainAcc[-1]))

    sns.heatmap(cmTest, linewidth=0.5 ,ax=axs[1])
    axs[1].set_ylabel('Actual')
    axs[1].set_xlabel('Predicted')
    axs[1].set_title('Eval_Acc: ' + str(evalAcc[-1]))

    sns.heatmap(cmTrain, linewidth=0.5,ax=axs[2])
    axs[2].set_xlabel('Actual')
    axs[2].set_ylabel('Predicted')
    axs[2].set_title('Test_Acc: ' + str(testAcc[-1]))


    fig.savefig(str(dirPath) + plotName)
    pickle.dump(fig, open(str(dirPath) + plotName+'.pickle', 'wb')) 
    #plt.show()
    return None

def plotLoss_Acc(dirPath,plotName):
    """
    returns: None

    plots the Loss and Accuracy over time(epochs/batches) for trainingSet and testSet 

    saves the plot as png
    saves the plot as pickel( for interactive plot in dataDiscovery)

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


    #trainLoss_iteration = torch.tensor(trainLoss_iteration, device = 'cpu')
    #trainLoss_epoch = torch.tensor(trainLoss_epoch, device = 'cpu')
    #evalLoss_iteration = torch.tensor(evalLoss_iteration, device = 'cpu')
    #evalLoss_epoch = torch.tensor(evalLoss_epoch, device = 'cpu')
    #testLoss_iteration = torch.tensor(testLoss_iteration, device = 'cpu')
    #testLoss_epoch = torch.tensor(testLoss_epoch, device = 'cpu')

    #trainingAcc_iteration = torch.tensor(trainAcc_iteration, device = 'cpu')
    #trainingAcc_epoch = torch.tensor(trainingAcc_epoch, device = 'cpu')
    #evalAcc_iteration = torch.tensor(evalLoss_iteration, device = 'cpu')
    #evalAcc_epoch = torch.tensor(training_loss_epoch, device = 'cpu')
    #testAcc_iteration = torch.tensor(testLoss_iteration, device = 'cpu')
    #testAcc_epoch = torch.tensor(test_loss_epoch, device = 'cpu')
    
    
    #fig, axs = plt.subplots(nrows=3, ncols=4)   # train ; epoch / iteration 
                                                # eval  ; epoch / iteration
                                                # test  ; epoch / iteration
    #fig.tight_layout(pad=0.5)

    stackedInputs = [[trainAccPerEpochList ,trainAccPerIterationList, trainLossPerEpochList,trainLossPerIterationList],
                     [evalAccPerEpochList ,evalAccPerIterationList, evalLossPerEpochList,evalLossPerIterationList],
                     [testAccPerEpochList ,testAccPerIterationList, testLossPerEpochList,testLossPerIterationList],]
    separatly = False

    def plotLA():
        if not(separatly):
            fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(3,4) )
            fig.tight_layout(pad=0.5)
            for i in range(len(stackedInputs)):
                for j in range(len(stackedInputs[i])):
                    name = ""
                    if i == 0:
                        name = "train"
                    elif i ==1:
                        name == "eval"
                    elif i==2:
                        name == "test"

                    axs[i][j].plot(range(len(stackedInputs[i][j])),stackedInputs[i][j]) 

                    if j == 0 or j ==1:
                        axs[i][j].set_ylabel('accuracy')
                        name += "Accuracy"

                    elif j == 2 or j == 3:
                        axs[i][j].set_ylabel('loss')
                        name += "Loss"
                    if j == 0 or j == 2:
                        axs[i][j].set_xlabel('epoch')
                        name += "PerEpoch"

                    elif j ==1 or j ==3:
                        axs[i][j].set_xlabel('iteration')
                        name += "PerIteration"
                    axs[i][j].set_title(str(name))
        else:

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

        return None
    plotLA()
"""
    with no_grad():
        if separatly:
            figLA1, axsLA1 = plt.subplots(nrows=1, ncols=1, figsize=(3,4) )
            axsLA1.plot(range(len(trainLossPerEpochList)),trainLossPerEpochList) 
            axsLA1.set_xlabel('epoch')
            axsLA1.set_ylabel('loss')
            axsLA1.set_title("trainLossPerEpoch")
            figLA2, axsLA2 = plt.subplots(nrows=1, ncols=1, figsize=(3,4) )
            axsLA2.plot(range(len(evalLossPerEpochList)),evalLossPerEpochList) 
            axsLA2.set_xlabel('epochs')
            axsLA2.set_ylabel('loss')
            axsLA2.set_title("evalLossPerEpoch")
            figLA3, axsLA3  = plt.subplots(nrows=1, ncols=1, figsize=(3,4) )
            axsLA3.plot(range(len(testLossPerEpochList)),testLossPerEpochList)
            axsLA3.set_xlabel('epochs')
            axsLA3.set_ylabel('loss')
            axsLA3.set_title("testLossPerEpoch")

            figLA4, axsLA4  = plt.subplots(nrows=1, ncols=1, figsize=(3,4) )
            axsLA4.plot(range(len(trainLossPerIterationList)),trainLossPerIterationList) 
            axsLA4.set_xlabel('iteration')
            axsLA4.set_ylabel('loss')   
            axsLA4.set_title("trainLossPerIteration")

            figLA5, axsLA5  = plt.subplots(nrows=1, ncols=1, figsize=(3,4) )
            axsLA5.plot(range(len(evalLossPerIterationList)),evalLossPerIterationList) # change to loss_per_epoch
            axsLA5.set_xlabel('iterations')
            axsLA5.set_ylabel('loss')
            axsLA5.set_title("evalLossPerIteration")

            #figLA6, axsLA6  = plt.subplots(nrows=1, ncols=1, figsize=(3,4) )
            #figLA6, axsLA7  = plt.subplots(nrows=1, ncols=1, figsize=(3,4) )
            #figLA6, axsLA8  = plt.subplots(nrows=1, ncols=1, figsize=(3,4) )
            #figLA6, axsLA9  = plt.subplots(nrows=1, ncols=1, figsize=(3,4) )
            #figLA6, axsLA10 = plt.subplots(nrows=1, ncols=1, figsize=(3,4) )
            #figLA6, axsLA11 = plt.subplots(nrows=1, ncols=1, figsize=(3,4) )
            #figLA6, axsLA12 = plt.subplots(nrows=1, ncols=1, figsize=(3,4) )

        
        #stack like that then loop in METHOD
        #trainE LOSS # trainI LOSS # trainE ACC # trainI ACC
        #evalE      #               #
        #testE      #               #
        axs[0][0].plot(range(len(trainLossPerEpochList)),trainLossPerEpochList) 
        axs[0][0].set_xlabel('epoch')
        axs[0][0].set_ylabel('loss')
        axs[0][0].set_title("trainLossPerEpoch")

        axs[1][0].plot(range(len(evalLossPerEpochList)),evalLossPerEpochList) 
        axs[1][0].set_xlabel('epochs')
        axs[1][0].set_ylabel('loss')
        axs[1][0].set_title("evalLossPerEpoch")

        axs[2][0].plot(range(len(testLossPerEpochList)),testLossPerEpochList)
        axs[2][0].set_xlabel('epochs')
        axs[2][0].set_ylabel('loss')
        axs[2][0].set_title("testLossPerEpoch")

        axs[0][1].plot(range(len(trainLossPerIterationList)),trainLossPerIterationList) 
        axs[0][1].set_xlabel('iteration')
        axs[0][1].set_ylabel('loss')   
        axs[0][1].set_title("trainLossPerIteration")

        axs[1][1].plot(range(len(evalLossPerIterationList)),evalLossPerIterationList) # change to loss_per_epoch
        axs[1][1].set_xlabel('iterations')
        axs[1][1].set_ylabel('loss')
        axs[1][1].set_title("evalLossPerIteration")

        axs[2][1].plot(range(len(testLossPerIterationList)),testLossPerIterationList)
        axs[2][1].set_xlabel('iterations')
        axs[2][1].set_ylabel('loss')
        axs[2][1].set_title("testLossPerIteration")

############# ACC
        axs[0][2].plot(range(len(trainAccPerEpochList)),trainAccPerEpochList) 
        axs[0][2].set_xlabel('epoch')
        axs[0][2].set_ylabel('accuracy')
        axs[0][2].set_title("trainAccPerEpoch")

        axs[1][2].plot(range(len(evalAccPerEpochList)),evalAccPerEpochList)
        axs[1][2].set_xlabel('epoch')
        axs[1][2].set_ylabel('accuracy')
        axs[1][2].set_title("evalAccPerEpoch")

        axs[2][2].plot(range(len(testAccPerEpochList)),testAccPerEpochList)
        axs[2][2].set_xlabel('epoch')
        axs[2][2].set_ylabel('accuracy')
        axs[2][2].set_title("testAccPerEpoch")

        axs[0][3].plot(range(len(trainAccPerIterationList)),trainAccPerIterationList) 
        axs[0][3].set_xlabel('iteration')
        axs[0][3].set_ylabel('accuracy')
        axs[0][3].set_title("trainAccPerIteration")

        axs[1][3].plot(range(len(evalAccPerIterationList)),evalAccPerIterationList)#
        axs[1][3].set_xlabel('iterations')
        axs[1][3].set_ylabel('accuracy')
        axs[1][3].set_title("evalAccPerIteration")

        axs[2][3].plot(range(len(testAccPerIterationList)),testAccPerIterationList)
        axs[2][3].set_xlabel('iterations')
        axs[2][3].set_ylabel('accuracy')
        axs[2][3].set_title("testAccPerIteration")

    print(dirPath)
    fig.savefig(dirPath +plotName)
    pickle.dump(fig, open(dirPath + plotName +'.pickle', 'wb')) 
    #plt.show()

    return None
    """



def plotGradientsPerFeature(dirPath, inputFeatures, featureListALL, plotName):
    """
    returns: None

    plots the gradients for each feature  

    saves the plot as png
    saves the plot as pickel( for interactive plot in dataDiscovery)

    """
    fig, axs = plt.subplots(inputFeatures)

    fig.set_size_inches(10,15)
    fig.tight_layout(pad=2.0)
    #find maximum/minumum 

    maxValue= 0
    minValue= 0
    #calculate metrics
    gradientMean_Features = []
    for i in range(len(featureListALL)):
        gradientMean_Features.append(np.array(featureListALL[i]).mean())

        maxValue = max( maxValue, max( featureListALL[i]))    
        minValue = min( minValue, min(featureListALL[i]))

    #plot for all features
    for i in range(len(axs)):
        axs[i].set_ylim(minValue,maxValue)
        axs[i].set_xlabel("mean:" + str(gradientMean_Features[i]))
        axs[i].plot(featureListALL[i])

    plt.savefig(str(dirPath) +str(plotName) )
    pickle.dump(fig, open(str(dirPath) + str(plotName), 'wb')) 

    plt.tight_layout()  
    plt.show()

    return None


def plotGradientsPerSample(featureListAll, num_epochs,data, dirPath, plotName):
    
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

                #plt.plot(gradientsPerSample[i][j])
                #pickle.dump(fig, open(str(dirPath) + str(plotName), 'wb'))
    howManySamplesToLookAt = 2

    for i in range(0,howManySamplesToLookAt):
        plotSubplots(i)
        ("press a button for the next Sample")
        w = plt.waitforbuttonpress()

def plotCosineSimilarity(dirPath, plotName,model, modelsDirPath):
    
    cosine_similarity_toInitial, cosine_similarity_toFinal =  eval.calcConsineSimilarity(model,modelsDirPath)
    fig, axs = plt.subplots(nrows=1, ncols=1)

    cosine_similarity_toInitial = np.array(cosine_similarity_toInitial).flatten()
    cosine_similarity_toFinal = np.array(cosine_similarity_toFinal).flatten()

    axs.plot(cosine_similarity_toInitial) 
    axs.plot(cosine_similarity_toFinal)
    pickle.dump(fig, open(str(dirPath) + str(plotName), 'wb'))

    plt.show()
    return None

def plotWeightSignDifferences(dirPath, plotName,model, modelsDirPath):
    
    percentageWeightSignDifference=  eval.calcWeightSignDifferences(model,modelsDirPath)
    fig, axs = plt.subplots(nrows=1, ncols=1)

    axs.plot(percentageWeightSignDifference ) 
    pickle.dump(fig, open(str(dirPath) + str(plotName), 'wb'))

    plt.show()
    
    return None

def plotWeightMagnitude(dirPath, plotName,model, modelsDirPath):

    weightsMagnitudeList = eval.calcWeightsMagnitude(model,modelsDirPath)
    fig, axs = plt.subplots(nrows=1, ncols=1)

    axs.plot(weightsMagnitudeList)
    pickle.dump(fig, open(str(dirPath) + str(plotName), 'wb'))

    plt.show()
    
    return None


def plotL2Distance(dirPath, plotName,model, modelsDirPath):
    l2Dist_toInitialList, l2Dist_toFinalList =  eval.calcL2distance(model,modelsDirPath)
    fig, axs = plt.subplots(nrows=1, ncols=1)

    axs.plot(l2Dist_toInitialList) 
    axs.plot(l2Dist_toFinalList)
    pickle.dump(fig, open(str(dirPath) + str(plotName), 'wb'))

    plt.show()
    return None

def plotWeightTrace(dirPath, plotName,model, modelsDirPath):
    weightTraceList =  eval.calcWeightTrace(model,modelsDirPath)
    fig, axs = plt.subplots(nrows=1, ncols=1)

    for i in weightTraceList:
        axs.plot(i ) 

    pickle.dump(fig, open(str(dirPath) + str(plotName), 'wb'))

    plt.show()
    return None

def plotGradientMagnitude(dirPath, plotName,featureListALL,perFeature= False):

    if not(perFeature):
        fig, axs = plt.subplots(nrows=1, ncols=1)
        averageGradientMagnitude =  eval.calcGradientMagnitude(featureListALL)
        axs.plot(averageGradientMagnitude) 

    else:

        gradientMagnitudePerFeature =  eval.calcGradientMagnitude(featureListALL, perFeature =perFeature)
            
        fig, axs = plt.subplots(nrows=int(len(gradientMagnitudePerFeature)), ncols=1) 
 
        for i, feature in enumerate(gradientMagnitudePerFeature):

                axs[i].plot(feature)

    pickle.dump(fig, open(str(dirPath) + str(plotName), 'wb'))

    plt.show()
    return None

"""
#TODO:
"""