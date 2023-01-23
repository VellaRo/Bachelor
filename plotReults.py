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

def plot_Loss_Acc(dirPath, training_loss_batch, training_loss_epoch, training_acc,test_loss_batch, test_loss_epoch, test_acc):
    """
    returns: None

    plots the Loss and Accuracy over time(epochs/batches) for trainingSet and testSet 

    saves the plot as png
    saves the plot as pickel( for interactive plot in dataDiscovery)

    """
    
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


def plot_features(dirPath, inputFeatures, featureListALL):
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

    plt.savefig(str(dirPath) +"featureListALL" )
    pickle.dump(fig, open(str(dirPath) + 'featureListALL.pickle', 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`

    plt.tight_layout()  
    plt.show()

    return None

"""
#TODO:
# plot the gradients for every image over time 
"""