import pandas as pd
import torch
#import torch.utils.data.Sampler 
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import StratifiedKFold

class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
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
def dropFeatures(droplist, X, inputFeatures):
    for i in droplist:
        inputFeatures -= 1
        X = X.drop([i], axis=1)
    return X, inputFeatures

def preProcessingData(X, y, path= None , batch_size= 4, test_size =0.2 ,droplist= [],):
        
   # X_train,X_test,y_train,y_test = train_test_split(X,y , test_size =test_size,random_state=1)# 

    X_train, X_rem, y_train, y_rem = train_test_split(X,y, test_size =test_size,random_state=1)

    X_eval, X_test, y_eval, y_test = train_test_split(X_rem,y_rem, test_size=0.5, random_state=1)

    #convert them to tensors
    X_train=torch.FloatTensor(X_train.values)
    X_eval=torch.FloatTensor(X_eval.values)
    X_test=torch.FloatTensor(X_test.values)


    y_train=torch.LongTensor(y_train.values)
    y_eval=torch.LongTensor(y_eval.values)
    y_test=torch.LongTensor(y_test.values)
    #X , inputFeatures= dropFeatures(droplist=droplist, X=X, inputFeatures=inputFeatures)

    def normalize(inputToNormalize, dim):
        from sklearn import preprocessing

       #normalize 
        inputToNormalize_min, _ = torch.min(inputToNormalize, dim=dim, keepdim=True)
        inputToNormalize_max, _ = torch.max(inputToNormalize, dim=dim, keepdim=True)
        mean = torch.mean(inputToNormalize, dim =dim ,keepdim= True)
        variance = torch.var(inputToNormalize, dim =dim ,keepdim= True)
        #print(variance)
        #normalizeed = (inputToNormalize - inputToNormalize_min) / (inputToNormalize_max - inputToNormalize_min) # Broadcasting rules apply
        
        #normalizeed = (inputToNormalize - inputToNormalize_min) / variance
        eps=1e-5
        normalizeed1 = (inputToNormalize - mean) / torch.sqrt(variance+ eps)
        #normalizeed2 = preprocessing.normalize(inputToNormalize, "l1") 
        print(normalizeed1[0])
        #print(normalizeed2[0])
        #print(type(normalizeed))
        #print(normalizeed)
        return normalizeed1

    X_train = normalize(X_train, dim=1)
    X_test = normalize(X_test, dim=1)
    #y_train = normalize(y_train, dim=0)
    #y_test = normalize(y_test, dim=0)
 
    #noramlize
    #X_train = torch.nn.functional.normalize(X_train, p=18.0, dim = 0)
    #X_test = torch.nn.functional.normalize(X_test, p=18.0, dim = 0)
    train_sampler = StratifiedBatchSampler(y_train, batch_size=batch_size, shuffle=True)
    eval_sampler = StratifiedBatchSampler(y_eval, batch_size=batch_size, shuffle=True)
    test_sampler = StratifiedBatchSampler(y_test, batch_size=batch_size, shuffle=False)

    trainset = list(zip(X_train , y_train))      
    #shuffle=True,
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=False, batch_sampler=train_sampler , num_workers=2)# shuffle false
    #trainloader = torch.utils.data.DataLoader(trainset, shuffle=False, batch_size=batch_size , num_workers=2)# shuffle false
    
    evalset = list(zip(X_eval , y_eval))      
    evalloader =  torch.utils.data.DataLoader(evalset, shuffle=False, batch_sampler=eval_sampler , num_workers=2)
    #print("shuffel = False")
    #print(next(trainloader))
    
    testset = list(zip(X_test , y_test))
    testloader = torch.utils.data.DataLoader(testset,shuffle=False, batch_sampler=test_sampler , num_workers=2)
    #testloader = torch.utils.data.DataLoader(testset,shuffle=False, batch_size=batch_size , num_workers=2)


    return trainloader ,evalloader, testloader ,X_train ,X_test ,  y_train , y_test
#data = pd.read_csv(r"/home/rosario/explainable/Bachelor/Diabetes/Data/diabetes.csv")

def load_kaggle_diabetes_dataset( batch_size= 4, test_size =0.2 ,droplist= []):
    path = "/home/rosario/explainable/Bachelor/Diabetes/Data/diabetes.csv"
    
    """
    returns trainloader , testloader ,X_train ,X_test ,  y_train , y_test, inputFeatures according to specifications

    path: (string) path to the dataset(.csv)
    batch_size: (int) 
    test_size: (double 0 - 1) ratio for testSet
    droplist: (list of strings) a list of string of features to drop ( not consider for training/testing) 
    
    """
    datasetName ="KaggleDiabetesALL"
    data = pd.read_csv(path) #macht ohne r' problme ?
    inputFeatures = len(data.columns) -1 
    print(inputFeatures)

    outputFeatures =2 
  
    features_names =  data.columns[:-1]

    X = data.drop("Outcome" , axis = 1) #independent Feature
    
    ### droping all features that i dont want in my Dataloader

    X , inputFeatures= dropFeatures(droplist=droplist, X=X, inputFeatures=inputFeatures)
    
    y = data["Outcome"]
    trainloader ,evalloader, testloader ,X_train ,X_test ,  y_train , y_test= preProcessingData(X=X, y=y, path=path ,batch_size=batch_size, test_size=test_size,droplist=droplist)
    print(len(y_train))
    return trainloader ,evalloader, testloader ,X_train ,X_test ,  y_train , y_test , inputFeatures, outputFeatures, datasetName, features_names



def BreastCancerUCI(batch_size= 4, test_size =0.2 ,droplist= []): # 0.2 tset,size  0.2 macght probleme ???ß

    datasetName = "BreastCancerUCI"
    data= load_breast_cancer(as_frame=True)
    features_names =  data.feature_names
    print(features_names)
    inputFeatures = 30
    outputFeatures =2
    data = data.frame


    #print(data.target)
    X = data.drop('target', axis = 1) 

    #X = dropFeatures(droplist=droplist, X=X, inputFeatures=input)
    #print(X)
    y= data["target"]    

    X, inputFeatures = dropFeatures(droplist= droplist, X=X, inputFeatures=inputFeatures)
    trainloader ,evalloader, testloader ,X_train ,X_test ,  y_train , y_test = preProcessingData(X=X, y=y,batch_size=batch_size, test_size=test_size,droplist=droplist)
    print(y_train)
    return trainloader ,evalloader, testloader ,X_train ,X_test ,  y_train , y_test , inputFeatures, outputFeatures, datasetName, features_names
#DiabetesUCI(batch_size= 4, test_size =0.2 ,droplist= [])

def dryBeanUCI(batch_size= 4, test_size =0.2 ,droplist= []):
    """
    returns trainloader , testloader ,X_train ,X_test ,  y_train , y_test, inputFeatures according to specifications

    path: (string) path to the dataset(.csv)
    batch_size: (int) 
    test_size: (double 0 - 1) ratio for testSet
    droplist: (list of strings) a list of string of features to drop ( not consider for training/testing) 
    
    """
    path = "/home/rosario/explainable/Bachelor/Diabetes/Data/DryBeanDataset/Dry_Bean_Dataset.csv"

    datasetName = "dryBeanUCI"
    inputFeatures = 16
    outputFeatures = 2# 7# 
    data = pd.read_csv(path) #macht ohne r' problme ?

    
    X = data.drop('Class' , axis = 1) #independent Feature
    #print(X)
    X = X[:3349]
    print(X)
    X = X.drop('IntClass' , axis = 1) #independent Feature
    #print(X)
    X = X.drop('IntClass1' , axis = 1) #independent Feature

    
    ### droping all features that i dont want in my Dataloader

    X , inputFeatures= dropFeatures(droplist=droplist, X=X, inputFeatures=inputFeatures)
    y = data["IntClass"]
    y = y[:3349]
    trainloader ,evalloader, testloader ,X_train ,X_test ,  y_train , y_test = preProcessingData(X=X, y=y, path=path ,batch_size=batch_size, test_size=test_size,droplist=droplist)

    return trainloader ,evalloader, testloader ,X_train ,X_test ,  y_train , y_test , inputFeatures, outputFeatures, datasetName