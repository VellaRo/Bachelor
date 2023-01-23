import pandas as pd
import torch 
import sklearn




#data = pd.read_csv(r"/home/rosario/explainable/Bachelor/Diabetes/Data/diabetes.csv")

def load_kaggle_diabetes_dataset(path , batch_size= 4, test_size =0.2 ,droplist= [],):
    """
    returns trainloader , testloader ,X_train ,X_test ,  y_train , y_test, inputFeatures according to specifications

    path: (string) path to the dataset(.csv)
    batch_size: (int) 
    test_size: (double 0 - 1) ratio for testSet
    droplist: (list of strings) a list of string of features to drop ( not consider for training/testing) 
    
    """
    
    inputFeatures = 8

    data = pd.read_csv(path) #macht ohne r' problme ?
    X = data.drop('Outcome' , axis = 1) #independent Feature
    ### droping all features that i dont want in my Dataloader
    for i in droplist:
        inputFeatures -= 1
        X = X.drop([i], axis=1)
    #X = X.drop('Pregnancies' , axis = 1)
    #X = X.drop('BloodPressure' , axis = 1)
    #X = X.drop('Age' , axis = 1)
    #X = X.drop('SkinThickness' , axis = 1)
    #X = X.drop('Glucose' , axis = 1)
    #X = X.drop('Insulin' , axis = 1)
    y = data["Outcome"]
    import numpy as np
 
    
    X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y , test_size =test_size,random_state=0)# 
    print("y_test")
    print(np.sum(y_test))
    print(len(y_test))
    print(np.sum(y_test)/len(y_test))
    print(1- np.sum(y_test)/len(y_test))

    print("y_train")
    print(np.sum(y_train))
    print(len(y_train))
    print(np.sum(y_train)/len(y_train))
    print(1- np.sum(y_train)/len(y_train))

    #convert them to tensors
    X_train=torch.FloatTensor(X_train.values)
    X_test=torch.FloatTensor(X_test.values)
    y_train=torch.LongTensor(y_train.values)
    y_test=torch.LongTensor(y_test.values)
    

    #noramlize
    X_train = torch.nn.functional.normalize(X_train, p=1.0, dim = 0)
    X_test = torch.nn.functional.normalize(X_test, p=1.0, dim = 0)


    trainset = list(zip(X_train , y_train))      


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True , num_workers=2)

    testset = list(zip(X_test , y_test))

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

    return trainloader , testloader ,X_train ,X_test ,  y_train , y_test, inputFeatures