import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import Sampler
import random
from sklearn.preprocessing import LabelEncoder

class StratifiedBatchSampler:
    """
       Stratified batch sampling
       
       Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        print(n_batches)
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)  
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)
    

#def dropFeatures(droplist, X, inputFeatures):
#    """
#    drops features which are part of the dataSet but we dont want to considerate into training
#
#    parameters:
#            droplist: a list of features(strings of the featureNames) these features will be droped from the dataSet
#
#            X: data
#
#            inputFeatures: number of input features  
#    """
#    for i in droplist:
#        inputFeatures -= 1
#        X = X.drop([i], axis=1)
#    return X, inputFeatures

def preProcessingData(X, y, batch_size= 4, test_size =0.2 ,datasetType="numerical"): #, path= None 
    """
    splits , normalizes, uses Stratisfied sample( to distribute all label categories even in small batch size like 4)
    
    parameters:
            X:  inputs
            y: labels
            batch_size: batch size
            test_size: size of the data for evaluation and testSet
            droplist: list of features(stings) to drop

    returns trainloader ,evalloader, testloader ,X_train ,X_eval, X_test ,  y_train, y_eval, y_test
numerical
    """
    #inputFeatures= 0

    # split into train , eval test
    #X_train, X_rem, y_train, y_rem = train_test_split(X,y, test_size =test_size,random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =test_size,random_state=1)
    #X_eval, X_test, y_eval, y_test = train_test_split(X_rem,y_rem, test_size=0.5, random_state=1)

    #convert them to tensors
    X_train=torch.FloatTensor(X_train.values)
    #X_eval=torch.FloatTensor(X_eval.values)
    X_test=torch.FloatTensor(X_test.values)
    y_train=torch.LongTensor(y_train.values)
    #y_eval=torch.LongTensor(y_eval.values)
    y_test=torch.LongTensor(y_test.values)
    #X , inputFeatures= dropFeatures(droplist=droplist, X=X, inputFeatures=inputFeatures)

    def normalize(inputToNormalize, dim):
       #normalize 

        mean = torch.mean(inputToNormalize, dim =dim ,keepdim= True)
        variance = torch.var(inputToNormalize, dim =dim ,keepdim= True)
        #print(variance)
        eps=1e-5
        normalizeed1 = (inputToNormalize - mean) / torch.sqrt(variance+ eps)
 
        return normalizeed1
    
    
    if datasetType == "numerical":
        X_train = normalize(X_train, dim=1)
        #X_eval = normalize(X_eval, dim=1)
        X_test = normalize(X_test, dim=1)
        print(X_test)

    # stratisfied sampler
    ##print(batch_size)
    #train_sampler = StratifiedBatchSampler(y_train, batch_size=batch_size, shuffle=True)
    #eval_sampler = StratifiedBatchSampler(y_eval, batch_size=batch_size, shuffle=True)
    #test_sampler = StratifiedBatchSampler(y_test, batch_size=batch_size, shuffle=False)
    ##print(batch_size)
    trainset = list(zip(X_train , y_train))      
    
    random_indices_test =  data_list = random.sample(range(len(X_test)), len(X_test))
    #print(random_indices_test)
    random_indices_train =  data_list = random.sample(range(len(X_train)), len(X_train))

    
    class OrderedListSampler(Sampler):
        def __init__(self, data_list):
            self.data_list = data_list

        def __iter__(self):
            return iter(self.data_list)

        def __len__(self):
            return len(self.data_list)
    
    sampler_test = OrderedListSampler(random_indices_test)
    sampler_train =OrderedListSampler(random_indices_train)
    #trainloader = torch.utils.data.DataLoader(trainset, shuffle=False, batch_sampler=train_sampler , num_workers=2)# shuffle false
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=False, batch_size= batch_size , num_workers=2, sampler= sampler_train)
    print("train:shuffel = False")
    #evalset = list(zip(X_eval , y_eval))      
    #evalloader =  torch.utils.data.DataLoader(evalset, shuffle=False, batch_sampler=eval_sampler , num_workers=2)
    print("eval:shuffel = False")
    
    testset = list(zip(X_test , y_test))
    #testloader = torch.utils.data.DataLoader(testset,shuffle=False, batch_sampler=test_sampler , num_workers=2)
    testloader = torch.utils.data.DataLoader(testset,shuffle=False, batch_size=batch_size , num_workers=2 ,sampler = sampler_test)


    #return trainloader ,evalloader, testloader ,X_train ,X_eval, X_test ,  y_train, y_eval, y_test
    return trainloader , testloader ,X_train , X_test ,  y_train, y_test , random_indices_train, random_indices_test




# DATASETS
def load_kaggle_diabetes_dataset( batch_size= 4, test_size =0.2):
    """

    path: (string) path to the dataset(.csv)
    batch_size: (int) 
    test_size: (double 0 - 1) ratio for testSet
    droplist: (list of strings) a list of string of features to drop ( not consider for training/testing) 

    returns trainloader , testloader ,X_train ,X_test ,  y_train , y_test, inputFeatures according to specifications
    """
        
    path = "./Diabetes/Data/diabetes.csv"
    
    datasetType ="numerical"
    datasetName ="KaggleDiabetesALL"
    data = pd.read_csv(path) #macht ohne r' problme ?

    inputFeatures = len(data.columns) -1 
    #print(inputFeatures)

    outputFeatures =2 
  
    features_names =  data.columns[:-1]

    X = data.drop("Outcome" , axis = 1) #independent Feature
    #print("MINIMAL DATALOADER WITH 100 datapoints")
    #X = X[:100]

    ### droping all features that i dont want in my Dataloader
    #X , inputFeatures= dropFeatures( X=X, inputFeatures=inputFeatures)
    
    y = data["Outcome"]
    #y = y[:100]

    trainloader , testloader ,X_train , X_test ,  y_train, y_test , random_indices_train, random_indices_test= preProcessingData(X=X, y=y,batch_size=batch_size, test_size=test_size, datasetType=datasetType)
    #return trainloader ,evalloader, testloader ,X_train, X_eval, X_test, y_train, y_eval, y_test , inputFeatures, outputFeatures, datasetName   trainloader ,evalloader, testloader ,X_train ,X_eval, X_test ,  y_train, y_eval, y_test= preProcessingData(X=X, y=y,batch_size=batch_size, features_names
    #trainloader , testloader ,X_train , X_test ,  y_train, y_test ,random_indices_train, random_indices_test, = preProcessingData(X=X, y=y,batch_size=batch_size, test_size=test_size,droplist=droplist)
    return trainloader ,random_indices_train, testloader,random_indices_test,X_train , X_test,  y_train , y_test, inputFeatures, outputFeatures, datasetName, features_names, datasetType


def BreastCancerUCI(batch_size= 4, test_size =0.2 ): # 0.2 tset,size  0.2 macght probleme ???ß
    """
    path: (string) path to the dataset(.csv)
    batch_size: (int) 
    test_size: (double 0 - 1) ratio for testSet
    droplist: (list of strings) a list of string of features to drop ( not consider for training/testing) 

    returns trainloader , testloader ,X_train ,X_test ,  y_train , y_test, inputFeatures according to specifications
    """
    datasetType = "numerical"
    datasetName = "BreastCancerUCI"
    data= load_breast_cancer(as_frame=True)
    features_names =  data.feature_names
    #print(features_names)
    inputFeatures = 30
    outputFeatures =2
    data = data.frame
    #print(type(data))
    X = data.drop('target', axis = 1) 
    #print("joo")
    #print(type(X))
   # X = dropFeatures(droplist, X, inputFeatures=inputFeatures)
    #print("joo")
    #print(type(X))

    #X = X.values
    
    y= data["target"]#.values    
    print(np.shape(y))
    print(np.shape(X))

    #X, inputFeatures = dropFeatures( X=X, inputFeatures=inputFeatures)
    trainloader , testloader ,X_train , X_test ,  y_train, y_test , random_indices_train, random_indices_test = preProcessingData(X=X, y=y,batch_size=batch_size, test_size=test_size, datasetType=datasetType)
    return  trainloader ,random_indices_train, testloader,random_indices_test,X_train , X_test,  y_train , y_test, inputFeatures, outputFeatures, datasetName, features_names, datasetType

def dryBeanUCI(batch_size= 4, test_size =0.2 ):
    """
    returns trainloader , testloader ,X_train ,X_test ,  y_train , y_test, inputFeatures according to specifications

    path: (string) path to the dataset(.csv)
    batch_size: (int) 
    test_size: (double 0 - 1) ratio for testSet
    droplist: (list of strings) a list of string of features to drop ( not consider for training/testing) 
    
    """
    path = "./DryBeans/Dry_Bean_Dataset.csv"
    
    datasetType = "numerical"
    datasetName = "dryBeanUCI"
    inputFeatures = 16
    outputFeatures = 2# 7# 
    data = pd.read_csv(path) 
    features_names =  data.feature_names
    
    X = data.drop('Class' , axis = 1) #independent Feature
    #print(X)
    X = X[:3349]
    #print(X)
    X = X.drop('IntClass' , axis = 1) #independent Feature
    #print(X)
    X = X.drop('IntClass1' , axis = 1) #independent Feature

    
    ### droping all features that i dont want in my Dataloader

    #X , inputFeatures= dropFeatures( X=X, inputFeatures=inputFeatures)
    y = data["IntClass"]
    y = y[:3349]
    trainloader , testloader ,X_train , X_test ,  y_train, y_test , random_indices_train, random_indices_test = preProcessingData(X=X, y=y,batch_size=batch_size, test_size=test_size , datasetType=datasetType)

    return trainloader ,random_indices_train, testloader,random_indices_test,X_train , X_test,  y_train , y_test, inputFeatures, outputFeatures, datasetName, features_names, datasetType
def loadAdult(batch_size= 4, test_size =0.2):
    data = pd.read_csv("./Adult/all.csv")
    
    datasetType = "categorical"
    datasetName = "Adult"
   
    outputFeatures = 2


    data = data[~data.isin(['?']).any(axis=1)]
    data = data[:500]
    data['target'] = data['target'].replace('>50K.', '>50k')
    data['target'] = data['target'].replace('<=50K.', '<=50k') 

    label_encoder = LabelEncoder()
    for i in data.columns:

        data[i] = label_encoder.fit_transform(data[i])
    X_DF = data.drop("target",axis=1)
    featureNames =  X_DF.columns
    inputFeatures = len(featureNames)
    
    y_DF = data["target"]
    print(y_DF)
    trainloader , testloader ,X_train , X_test ,  y_train, y_test , random_indices_train, random_indices_test = preProcessingData(X=X_DF, y=y_DF ,batch_size=batch_size, test_size=test_size, datasetType=datasetType)

    return trainloader ,random_indices_train, testloader,random_indices_test,X_train , X_test,  y_train , y_test, inputFeatures, outputFeatures, datasetName, featureNames, datasetType

