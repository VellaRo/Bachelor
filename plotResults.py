from matplotlib import pyplot as plt
#plot the loss function
import pickle #figx = pickle.load(open('FigureObject.fig.pickle', 'rb')) 
              #figx.show() # Show the figure, edit it, etc.!
import sklearn
import seaborn as sns
from torch import no_grad
import numpy as np
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

def plot_Loss_Acc(dirPath, training_loss_batch, training_loss_epoch, training_acc,test_loss_batch, test_loss_epoch, test_acc):
    """
    returns: None

    plots the Loss and Accuracy over time(epochs/batches) for trainingSet and testSet 

    saves the plot as png
    saves the plot as pickel( for interactive plot in dataDiscovery)

    """
    
    training_loss_batch = torch.tensor(training_loss_batch, device = 'cpu')
    training_loss_epoch = torch.tensor(training_loss_epoch, device = 'cpu')
    test_loss_batch = torch.tensor(test_loss_batch, device = 'cpu')
    test_loss_epoch = torch.tensor(test_loss_epoch, device = 'cpu')


    fig, axs = plt.subplots(nrows=3, ncols=2)
    fig.tight_layout(pad=2.0)

    with no_grad():
        axs[0][0].plot(range(len(training_loss_batch)),training_loss_batch) #change to loss_per_batch
        axs[0][0].set_xlabel('batch')
        axs[0][0].set_ylabel('training_loss')


        axs[1][0].set_xlabel('epochs')
        axs[1][0].set_ylabel('training_loss_epoch')
        axs[1][0].plot(range(len(training_loss_epoch)),training_loss_epoch) # change to loss_per_epoch

        axs[2][0].set_xlabel('epochs')
        axs[2][0].set_ylabel('training_acc')
        axs[2][0].plot(range(len(training_acc)),training_acc)

        axs[0][1].plot(range(len(test_loss_batch)),test_loss_batch) #change to loss_per_batch
        axs[0][1].set_xlabel('batch')
        axs[0][1].set_ylabel('test_loss')


        axs[1][1].set_xlabel('epochs')
        axs[1][1].set_ylabel('test_loss_epoch')
        axs[1][1].plot(range(len(test_loss_epoch)),test_loss_epoch) # change to loss_per_epoch

        axs[2][1].set_xlabel('epochs')
        axs[2][1].set_ylabel('test_acc')
        axs[2][1].plot(range(len(test_acc)),test_acc)

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

import eval
def plot_cosine_similarity(dirPath, plotName,model, modelsDirPath):
    
    cosine_similarity_toInitial, cosine_similarity_toFinal =  eval.calcConsineSimilarity(model,modelsDirPath)
    fig, axs = plt.subplots(nrows=1, ncols=1)

    #print(cosine_similarity_toFinal)
    cosine_similarity_toInitial = np.array(cosine_similarity_toInitial).flatten()
    cosine_similarity_toFinal = np.array(cosine_similarity_toFinal).flatten()

    axs.plot(cosine_similarity_toInitial ) #np.flip # for better plot
    axs.plot(cosine_similarity_toFinal)
    pickle.dump(fig, open(str(dirPath) + str(plotName), 'wb'))

    plt.show()
    return None

def plotGradientsPerSample(featureListAll, num_epochs,data, dirPath, plotName):
    
    print("SHUFFLE NEEDS TO BE FALSE AT THE MOMENT") #THIS IS WROOOOONG REDO ON PAPER
    ##
    ##pro feature
    ##

    


    def doItForFeature(index):
        samplesPerFeature = []

        for i in range(len(data)):
 #           print(index)
        #print(str(i/8)+ "%")
            tempSamples = []
            #test = []
            for j in range(i, num_epochs*len(data),len(data)):
                
                tempSamples.append(featureListAll[index][j])
                #test.append(j)
            #print(test)
            samplesPerFeature.append(tempSamples)
 #       print(len(samplesPerFeature))
        return samplesPerFeature
    gradientsPerSample = []
    
    for k in range(len(featureListAll)):
 #       print(k)
        
        gradientsPerSample.append(doItForFeature(k))

       # gradientsPerSample.append(tempSamplesPerFeature)
    
   
    
    #fig.set_size_inches(10,15)
    #ig.tight_layout(pad=4.0)
    #print(len(gradientsPerSample))

    def plotSubplots(index):
        fig, axs = plt.subplots(nrows=len(featureListAll), ncols=2 )

        for i in range(len(featureListAll)):#len(featureListAllPerSample)):
            #for j in range(index,index+2):
                #print(str(i) +" "+ str(j))
                axs[i][0].plot(gradientsPerSample[i][index])
                axs[i][1].plot(gradientsPerSample[i][index+1])


                #plt.plot(gradientsPerSample[i][j])
                #pickle.dump(fig, open(str(dirPath) + str(plotName), 'wb'))
    howManySamplesToLookAt = 2
    #print(howManySamplesToLookAt)
    for i in range(0,howManySamplesToLookAt):
        plotSubplots(i)
        
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
                    #tempList.append(signChangeCounter)
                signChangeCounterListTemp.append(signChangeCounter)
            signChangeCounterList.append(signChangeCounterListTemp)
 
        return signChangeCounterList
    
    for i in range(0,num_epochs,10):
        begin = i
        end = begin +10
        signChangeCounterList = calculateSignChanges(begin, end, gradientsPerSample)
        #print(np.average(signChangeCounterList))
    

    return signChangeCounterList
    


"""
#TODO:
# plot the gradients for every image over time 
"""