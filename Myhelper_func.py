import pandas as pd
from multiprocessing import Queue
import numpy as np

intervals_dict = {}
pos_queue = Queue()
neg_queue = Queue()


def plot_roc(clf, X, Y):
    import matplotlib.pyplot as plt
    from sklearn.metrics import plot_roc_curve, roc_curve, auc

    plot_roc_curve(clf, X, Y)
    plt.show()


def find_Optimal_Cutoff(fpr, tpr, threshold):
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return list(roc_t['threshold'])


def compute_intervals(intervals_dict, X_train, num_bins=5):
    names = X_train.columns
    for name in names:
        unique_values = X_train[name].unique()
        if len(unique_values) > 2 or max(unique_values) != 1 or min(unique_values) != 0:
            intervals = pd.cut(X_train[name], num_bins)
            intervals_dict[name] = intervals


def get_relevant_features(zipped_data):
    global pos_queue
    global neg_queue

    shap_value, feature_value, feature_name, shap_threshold = zipped_data

    if shap_value != 0:
        if feature_value == 0:
            shap_value = -(shap_value)

        if shap_value > shap_threshold:
            name = format_name(feature_name, feature_value)
            pos_queue.put(name)
        elif shap_value < (-shap_threshold):
            name = format_name(feature_name, feature_value)
            neg_queue.put(name)


def append_to_encoded_vals(class_queue, itemset, encoded_vals):
    labels = {}
    rowset = set()

    while class_queue.qsize() > 0:
        rowset.add(class_queue.get())

    uncommons = list(itemset - rowset)
    commons = list(itemset.intersection(rowset))

    for uc in uncommons:
        labels[uc] = 0
    for com in commons:
        labels[com] = 1
    encoded_vals.append(labels)


def format_name(name, feature_value):
    global intervals_dict

    if name in intervals_dict:
        intervals = intervals_dict[name]
        for interval in intervals:
            if interval != interval: continue
            if feature_value in interval:
                left = interval.left
                right = interval.right
                name = f'{left}<{name}<={right}'
                break
    new_name = str(name).replace('less', '<').replace('greater', '>')
    return new_name


def clean_name(feature_name):
    if feature_name.split(' ')[0].strip().replace('.', '').isdigit():
        feature_name = feature_name.split(' ')[2].strip()
    else:
        feature_name = feature_name.split(' ')[0].strip()
    return feature_name

import torch 
def modelProbability(model, instance, device):

    with torch.no_grad():
        
        #for inputs,labels in testloader:
        
        #    inputs = inputs.to(device)
        #    labels = labels.to(device)  
    
            outputs = model(instance)
                
            __, preds = torch.max(outputs, 1)
            prob = torch.nn.functional.softmax(outputs, dim=1)

            #print(outputs)
    return(prob)
            #test_running_corrects += torch.sum(t_preds == t_labels.data)
            #test_loss = loss_function(outputs, labels)

import random
def calculateShaplyValue(model,instance,device, data,indx,featurePosition):
    j = featurePosition
    M = 1000 # number of iterations
    n_features = len(data)
    marginal_contributions = []
    feature_idxs = list(range(n_features))
    feature_idxs.remove(j)
    for _ in range(M):
        z = data[indx]
        x_idx = random.sample(feature_idxs, min(max(int(0.2*n_features), random.choice(feature_idxs)), int(0.8*n_features)))
        z_idx = [idx for idx in feature_idxs if idx not in x_idx]
    
        # construct two new instances
        x_plus_j = np.array([data[i] if i in x_idx + [j] else z[i] for i in range(n_features)])
        x_minus_j = np.array([z[i] if i in z_idx + [j] else data[i] for i in range(n_features)])
    
        # calculate marginal contribution
        marginal_contribution = modelProbability(model, instance,device)[0][1] - modelProbability(model, instance,device)[0][1]
        marginal_contributions.append(marginal_contribution)
    
    shaplyValue = sum(marginal_contributions) / len(marginal_contributions)  # our shaply value
    print(f"Shaply value for feature j: {shaplyValue:.5}")
    return(shaplyValue)
