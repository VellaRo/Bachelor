from matplotlib import pyplot as plt
#plot the loss function
import pickle #figx = pickle.load(open('FigureObject.fig.pickle', 'rb')) 
              #figx.show() # Show the figure, edit it, etc.!
import sklearn
import seaborn as sns
from torch import no_grad
import numpy as np
import eval

def plot_CM(y_train,y_test,  train_predictions, test_predictions, dirPath):

    """
    returns: None

    plots the Confusionmatrix for trainingSet and testSet 

    saves the plot as png
    saves the plot as pickel( for interactive plot in dataDiscovery)

    """
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(5,3) )


    cmTest = sklearn.metrics.confusion_matrix(y_test,test_predictions)
    cmTrain = sklearn.metrics.confusion_matrix(y_train,train_predictions)

    sns.heatmap(cmTest, linewidth=0.5 ,ax=axs[0])
    axs[0].set_ylabel('Actual')
    axs[0].set_xlabel('Predict')
    axs[0].set_title('Test_Acc: ' + str(sklearn.metrics.accuracy_score(y_test , test_predictions)))

    sns.heatmap(cmTrain, linewidth=0.5,ax=axs[1])
    axs[1].set_xlabel('Actual')
    axs[1].set_ylabel('Predicted')
    axs[1].set_title('LastTrain_Acc: ' + str(sklearn.metrics.accuracy_score(y_train , train_predictions)))



    fig.savefig(str(dirPath) + "_TrueFalseHeatmap")
    pickle.dump(fig, open(str(dirPath) + '_TrueFalseHeatmap.pickle', 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`

    return None

import torch

def plot_Loss_Acc(dirPath,model,modelsDirPath, trainloader,evalloader,testloader, device, loss_function, num_epochs, y_train, y_eval, y_test):
    """
    returns: None

    plots the Loss and Accuracy over time(epochs/batches) for trainingSet and testSet 

    saves the plot as png
    saves the plot as pickel( for interactive plot in dataDiscovery)

    """
    trainAcc_epoch, trainAcc_iteration, trainLoss_epoch, trainLoss_iteration = eval.calcAccLoss(model, modelsDirPath, trainloader,"train", device, loss_function, num_epochs, y_train)
    evalAcc_epoch, evalAcc_iteration, evalLoss_epoch, evalLoss_iteration =     eval.calcAccLoss(model, modelsDirPath, evalloader, "eval" ,device, loss_function, num_epochs, y_eval)
    testAcc_epoch, testAcc_iteration, testLoss_epoch, testLoss_iteration =     eval.calcAccLoss(model, modelsDirPath, testloader,"test", device, loss_function, num_epochs, y_test)

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
    
    
    fig, axs = plt.subplots(nrows=3, ncols=4)   # train ; epoch / iteration 
                                                # eval  ; epoch / iteration
                                                # test  ; epoch / iteration
    #fig.tight_layout(pad=2.0)

    with no_grad():
        axs[0][0].plot(range(len(trainLoss_epoch)),trainLoss_epoch) 
        axs[0][0].set_xlabel('epoch')
        axs[0][0].set_ylabel('trainLoss_epoch')


        axs[1][0].set_xlabel('epochs')
        axs[1][0].set_ylabel('evalLoss_epoch')
        axs[1][0].plot(range(len(evalLoss_epoch)),evalLoss_epoch) 

        axs[2][0].set_xlabel('epochs')
        axs[2][0].set_ylabel('testLoss_epoch')
        axs[2][0].plot(range(len(testLoss_epoch)),testLoss_epoch)

        axs[0][1].plot(range(len(trainLoss_iteration)),trainLoss_iteration) 
        axs[0][1].set_xlabel('iteration')
        axs[0][1].set_ylabel('trainLoss_iteration')

        
        axs[1][1].set_xlabel('iterations')
        axs[1][1].set_ylabel('evalLoss_iteration')
        axs[1][1].plot(range(len(evalLoss_iteration)),evalLoss_iteration) # change to loss_per_epoch

        axs[2][1].set_xlabel('iterations')
        axs[2][1].set_ylabel('testLoss_iteration')
        axs[2][1].plot(range(len(testLoss_iteration)),testLoss_iteration)

############# ACC
        axs[0][2].plot(range(len(trainAcc_epoch)),trainAcc_epoch) 
        axs[0][2].set_xlabel('epoch')
        axs[0][2].set_ylabel('trainAcc_epoch')


        axs[1][2].set_xlabel('epoch')
        axs[1][2].set_ylabel('evalAcc_epoch')
        axs[1][2].plot(range(len(evalAcc_epoch)),evalAcc_epoch) 

        axs[2][2].set_xlabel('epoch')
        axs[2][2].set_ylabel('testAcc_epoch')
        axs[2][2].plot(range(len(testAcc_epoch)),testAcc_epoch)

        axs[0][3].plot(range(len(trainAcc_iteration)),trainAcc_iteration) 
        axs[0][3].set_xlabel('iteration')
        axs[0][3].set_ylabel('trainAcc_iteration')


        axs[1][3].set_xlabel('iterations')
        axs[1][3].set_ylabel('evalAcc_iteration')
        axs[1][3].plot(range(len(evalAcc_iteration)),evalAcc_iteration) # change to loss_per_epoch

        axs[2][3].set_xlabel('iterations')
        axs[2][3].set_ylabel('testAcc_iteration')
        axs[2][3].plot(range(len(testAcc_iteration)),testAcc_iteration)
        
        
    plt.show()
    fig.savefig(str(dirPath) + "_loss")
    pickle.dump(fig, open(str(dirPath) + '_loss.pickle', 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`

    return None


def plot_features(dirPath, inputFeatures, featureListALL, plotName):
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
    pickle.dump(fig, open(str(dirPath) + str(plotName), 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`

    plt.tight_layout()  
    plt.show()

    return None



def plotGradientsPerSample(featureListAll, num_epochs,data, dirPath, plotName):
    
    print("SHUFFLE NEEDS TO BE FALSE AT THE MOMENT") 

    print(np.shape(np.array(featureListAll)))
    #d1, d2 = np.shape(np.array(featureListAll))
    #print(type(featureListAll))
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

        for i in range(len(featureListAll)):#len(featureListAllPerSample)):

            axs1[i][0].plot(gradientsPerSample[i][index])
            axs1[i][1].plot(gradientsPerSample[i][index+1])

                #plt.plot(gradientsPerSample[i][j])
                #pickle.dump(fig, open(str(dirPath) + str(plotName), 'wb'))
    howManySamplesToLookAt = 2

    for i in range(0,howManySamplesToLookAt):
        plotSubplots(i)
        w = plt.waitforbuttonpress()

    
             
    """
        w = plt.waitforbuttonpress()
    print(np.shape(gradientsPerSample))
    def calculateSignChanges(begin, end , gradientsPerSample):

        gradientsPerSample=  np.array(gradientsPerSample)
        earlyStageList = gradientsPerSample[:,:,begin:end] # features , samples , epochs 
        print(np.shape(earlyStageList))
        biggerThenZeroBefore = False 
        signChangeCounterList =[]
        for i in range(len(earlyStageList)): # number features 
            signChangeCounterListTemp= []
            for j in range(len(earlyStageList[i])): # nuber of samples 
                signChangeCounter = 0
                for k in range(len(earlyStageList[i][j])):
                    if earlyStageList[i][j][k] < 0:#
                        biggerThenZero = False
                    else:
                        biggerThenZero = True

                    if biggerThenZero != biggerThenZeroBefore: 
                        signChangeCounter = signChangeCounter +1
                        biggerThenZeroBefore = not(biggerThenZeroBefore)
                signChangeCounterListTemp.append(signChangeCounter)
            signChangeCounterList.append(signChangeCounterListTemp)
 
        return signChangeCounterList
    
    for i in range(0,num_epochs,10):
        begin = i
        end = begin +10
        signChangeCounterList = calculateSignChanges(begin, end, gradientsPerSample)


    return signChangeCounterList
"""
def plot_cosine_similarity(dirPath, plotName,model, modelsDirPath):
    
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

def plotWeightsMagnitude(dirPath, plotName,model, modelsDirPath):

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
# plot the gradients for every image over time 
"""