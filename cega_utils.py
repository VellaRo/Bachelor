from tqdm import tqdm
import pickle
import os
import shutil
import pandas as pd
from multiprocessing import Pool, cpu_count
from helper_func import *
from mlxtend.frequent_patterns import apriori, association_rules
import re
import utils
from datetime import datetime
##
import time
from torchtext.data.utils import get_tokenizer 
from datasetsNLP import _get_vocab
import numpy as np
def loadOHE_Rules(iterationNumber):
    """
    """
    path  = f"./OHEresults/ohe@Iteration_{iterationNumber}.pkl"
    with open(path, 'rb') as f:
        ohe_df =  pickle.load(f)

    return ohe_df

#def jaccard_similarity(set1, set2):
#
#    intersection = len(set1.intersection(set2))
#    union = len(set1.union(set2))
#    similarity = intersection / union
#
#    return similarity

def jaccard_similarity(list1, list2):
    set1 = set(tuple(element) for element in list1)
    set2 = set(tuple(element) for element in list2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    similarity = intersection / union if union != 0 else 0
    
    return similarity

#from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity(list1, list2):
    try:
        set1 = set(tuple(element) for element in list1)
        set2 = set(tuple(element) for element in list2)
        intersection = set1.intersection(set2)
        similarity = len(intersection) / (len(set1) * len(set2)) ** 0.5
        return similarity
    except ZeroDivisionError:
        return 0.0

def dice_similarity(list1, list2):

    intersection = len( set(tuple(element) for element in list1).intersection( set(tuple(element) for element in list1)))
    try:
        dice = (2.0 * intersection) / (len(list1) + len(list2))
    except:
        dice = 0
    return dice

def overlap_coefficient(list1, list2):
    set1 = set(tuple(element) for element in list1)
    set2 = set(tuple(element) for element in list2)
    intersection = set1.intersection(set2)
    try:

        return len(intersection) / min(len(set1), len(set2))
    except:
        return 0

def tversky_index(list1, list2, alpha=0.5, beta=0.5):
    set1 = set(tuple(element) for element in list1)
    set2 = set(tuple(element) for element in list2)
    intersection = set1.intersection(set2)
    numerator = len(intersection)
    denominator = (alpha * len(set1)) + (beta * len(set2)) + ((1 - alpha - beta) * len(intersection))
    return numerator / denominator



def categoricalToOHE(data):

    # drop target
    
    
    oldFeatureNames = data.columns
    newFeatureNames = []
    
    for column in data.columns:
        tempUniqueInColumn = data[column].unique()
        tempUniqueInColumn = [str(column)+ str(item)  for item in tempUniqueInColumn]
        newFeatureNames.extend(tempUniqueInColumn)
    
    OHE = np.zeros((len(data[column]), len(newFeatureNames)))
    OHE_DF =  pd.DataFrame(OHE, columns=newFeatureNames)
    # fill OHE_DF
    for c, column in enumerate(oldFeatureNames):
        for i,item in enumerate(data[column]):
            if str(item) ==  newFeatureNames[c][len(oldFeatureNames[c]):]:
                OHE_DF.iloc[i,c] = 1

    return OHE_DF, newFeatureNames

def vocabToOHE(data ,vocab):
    #def get_vocab_dictionary: 
    #tokenizer = get_tokenizer('basic_english')
    #vocab = _get_vocab 
    #padding_val =vocab['<pad>']

    #def text_pipeline(input):
    #    return vocab(tokenizer(input))

    vocab_dictionary = []
    for x in range(len(data)):

        #processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text = data.iloc[x].values
        #processed_text = text_pipeline(text)
        #processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        processed_text_int = []
        #print(processed_text)
        for i in text:
            processed_text_int.append(i)
        vocab_dictionary.extend(processed_text_int)

    
    vocab_dictionary =  set(vocab_dictionary)

    featureNames = np.sort(list(vocab_dictionary))

# Create a DataFrame with all zeros
    OHE_DF = pd.DataFrame(np.zeros((len(data), len(featureNames))), columns=featureNames)

# Iterate over featureNames and check if each one is in the corresponding row of 'data'
    for j, feature in enumerate(featureNames):
        OHE_DF[feature] = np.where(data.apply(lambda row: feature in row.values, axis=1),1, 0)# str(feature)+ " : 1", str(feature)+ " : 0") #



    #OHE = np.zeros((len(data), len(vocab_dictionary)))
    #OHE_DF =  pd.DataFrame(OHE, columns=featureNames)
    #for i in range(len(data)):
     #   for j in range(len(featureNames)):
      #      if featureNames[j] in data.iloc[i].values:
       #         OHE_DF.iloc[i,j] = 1
        #    else:
        #        OHE_DF.iloc[i,j] = 0

	    #return set(vocab_directory)
    
    return OHE_DF, featureNames

#only for binary 
                  #X_test
def calculateAndSaveOHE_Rules(data, featureNames,trainedModelPrediction_Test, grads, datasetType ,debug=False, vocab = None):

    """
    explainationGrads should have shape of [epochs * iterationsPerEpoch , testDatasetSize, featuresize ]


    """
    data_df = pd.DataFrame(data, columns=featureNames)

    pos_label = '1'
    neg_label = '0'

    itemset = set()
    encoded_vals = []
    summed_values = {}
    num_features = data.shape[1]

    shap_threshold = 0 # 0 0.0001
    num_cores = cpu_count()

    if datasetType == "NLP": #"numerical":
        if len(intervals_dict) == 0:
            compute_intervals(intervals_dict, data_df, 1) #1
    else:
        if len(intervals_dict) == 0:
            compute_intervals(intervals_dict, data_df, 5) #5 

    p = Pool(num_cores)


    if debug:
        print("this saves to a dummy folder which is beeing replaced")
        output_directory = './DEBUG/OHEresults/'
    else:
        output_directory = './OHEresults/'
    output_base_filename = 'ohe@Iteration'
    iterationCounter = 0

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    else :
        # Remove the directory and its contents
        shutil.rmtree(output_directory)
        os.makedirs(output_directory)

    if datasetType == "numerical":
        #print(data_df)
        #print("DATA")
        #print(data_df)
        for feature in data_df.columns.to_list(): # for NLP this must be the whole vocab 
            if feature in intervals_dict:
                intervals = intervals_dict[feature]
                for interval in intervals:
                    if interval != interval: continue
                    left = interval.left
                    right = interval.right
                    name = f'{left}<{feature}<={right}'
                    itemset.add(name)

            else:
                itemset.add(feature)
        #print(itemset)


    else: 
        #data_df, featureNames = categoricalToOHE(data_df)
        #print("NLP")
        data_df, featureNames = vocabToOHE(data_df, vocab)
        #print("DATA")
        #print(data_df)
        #print(data_df.öoll)
        #print(featureNames)
        #print(data_df)
        #for feature in data_df.columns.to_list(): # for NLP this must be the whole vocab 
        #    if feature in intervals_dict:
        #        intervals = intervals_dict[feature]
        #        for interval in intervals:
        #            if interval != interval: continue
        #            left = interval.left
        #            right = interval.right
        #            name = f'{left}<{feature}<={right}'
        #            itemset.add(name)

        #    else:
        #        itemset.add(feature)
        #print(itemset)

        for feature in data_df.columns.to_list():
            #print(feature)
            itemset.add(str(feature))
            #print(itemset)
        #print(itemset)
    itemset.add(pos_label)
    itemset.add(neg_label)
    #print(len(itemset))


    def CEGA(gradsPerIteration): 
        ### for every model iteration the gradients of whole test_dataset is calculated
        for indx in tqdm (range(len(gradsPerIteration))):#(len(trainedModelPrediction_Test)): #  len test dataset
            #print(indx)
            pos_queue.put(pos_label)
            neg_queue.put(neg_label)
            exp = gradsPerIteration[indx]#[item[indx] for item in sample] #normalize featureListALL ?
            #print("exp")
            #print(exp)

            instance_features = data_df.iloc[[indx]].to_dict(orient='records')[0] 
            #print("print(instance_features)")
            #print(instance_features)
            
            feature_vals = [instance_features[name] for name in featureNames] #put here grads#   feature values ?? 
            #print(feature_vals)
            #print.essa
            zipped = zip(exp, feature_vals,
                         featureNames, [shap_threshold]*len(featureNames))


            p.map(get_relevant_features, zipped)
            #print(encoded_vals)
            append_to_encoded_vals(pos_queue, itemset, encoded_vals)
            append_to_encoded_vals(neg_queue, itemset, encoded_vals)

            ohe_df = pd.DataFrame(encoded_vals)
            #print(ohe_df)
            #print.pes   
        return ohe_df #ohe_dfList.append(ohe_df)
    
    for i in tqdm (range(len(grads))):
        gradsPerIteration  = grads[i]
        #trainedModelPrediction_TestPerIteration = trainedModelPrediction_Test[i]
        output_filename = f'{output_directory}{output_base_filename}_{iterationCounter}.pkl'
        #try:
        print(i)
        #if i ==0 or i ==50 or i ==100 or i ==199:
        with open(output_filename, 'wb') as f:
            ohe_df = CEGA(gradsPerIteration)#,trainedModelPrediction_TestPerIteration)
            pickle.dump(ohe_df, f)
        iterationCounter += 1

        #except FileExistsError:
            # If the file already exists, increment the counter and try again
        #    iterationCounter += 1
        #    pickle.dump(ohe_df, f)
        #    print("exception")
        encoded_vals = []



        #gradsPerIteration  = grads[i]


    # TAKES ~30 sec for 154 samples  
    return featureNames
import gely

import myAssociationRules

def runApriori(ohe_df,testDataLength, pos_label ,neg_label): # min thrshold add  to def input ?
    
    def gelyOutputToDF(gelyOutput):
        # Create a dictionary to hold the data in the required format
        data_dict = {"support": [], "itemsets": []}

        # Extract the itemset and support values from the input and populate the dictionary
        for itemsets, support in gelyOutput:
            #print(type(itemsets))
            itemsets = {str(item) for item in itemsets} # convert to string
            #print(itemsets)
            data_dict["itemsets"].append(frozenset(itemsets))
            #mapping = {idx: item for idx, item in enumerate(["support", "itemsets"])}

           # data_dict['itemsets'] = data_dict["itemsets"].apply(lambda x: frozenset([mapping[i] for i in x]))
            data_dict["support"].append(0.5)#support / testDataLength)


        # Create a pandas DataFrame using the dictionary
        freq_items = pd.DataFrame(data_dict)
        # Create a new column to store the length of each frozenset
        freq_items['itemset_length'] = freq_items['itemsets'].apply(lambda x: len(x))

        # Sort the DataFrame based on the "itemset_length" column
        freq_items = freq_items.sort_values(by='itemset_length')

        # Drop the temporary "itemset_length" column if you don't need it in the final DataFrame
        freq_items = freq_items.drop(columns='itemset_length')
        
        return freq_items


                                            # 10/ len(pred)10/testDataLength*5
    freq_items1 = apriori(ohe_df, min_support=(0.00000000000000001), use_colnames=True, max_len=3, low_memory=True) #[array[0.5, frozenset]]
    print(freq_items1["itemsets"])
    print(freq_items1["itemsets"][0])
    print(type(freq_items1["itemsets"][0]))
    #print((freq_items1["itemsets"][0]).dtype)

    all_rules = myAssociationRules.association_rules(freq_items1, metric="confidence", min_threshold=0.0 ,support_only=False) # 0.7 support_only=False

    freq_items = gely.gely(ohe_df.values, 3) 
    #print(freq_items1)
    #print(freq_items1["antecedent"])
    #print("??")
    #print(freq_items)
    #print(type(freq_items1["itemsets"][0]))
    freq_items = gelyOutputToDF(freq_items)
    print("---")
    print(freq_items["itemsets"])
    print(freq_items["itemsets"][0])
    print(type(freq_items["itemsets"][0]))
    #print((freq_items["itemsets"][0]).dtype)
    #print(freq_items)
    #print("-------")
    #print(freq_items1)     
    #association_rules
    all_rules = myAssociationRules.association_rules(freq_items, metric="confidence", min_threshold=0.0 ,support_only=False) # 0.7 support_only=False

    #exit()                                    # 10/ len(pred)10/testDataLength*5
    #freq_items1 = apriori(ohe_df.loc[ohe_df[pos_label] == 1], min_support=(0.00000000000000001), use_colnames=True, max_len=3 , low_memory=True) # max len 3
    freq_items = gely.gely(ohe_df.loc[ohe_df[pos_label] == 1].values, 3) 
    freq_items = gelyOutputToDF(freq_items)

    pos_rules = myAssociationRules.association_rules(freq_items, metric="confidence", min_threshold=0.0, support_only=False) # 0.6 support_only=False
    #print(ohe_df.loc[ohe_df[neg_label] == 1])
    #print.epss
    freq_items = gely.gely(ohe_df.loc[ohe_df[neg_label] == 1].values, 3) 
    freq_items = gelyOutputToDF(freq_items)
                                                                    # 10/ len(pred)10/testDataLength*5
    #freq_items1 = apriori(ohe_df.loc[ohe_df[neg_label] == 1], min_support=(0.00000000000000001), use_colnames=True, max_len=3, low_memory=True) # max len 3 
    neg_rules = myAssociationRules.association_rules(freq_items, metric="confidence", min_threshold=0.0 , support_only=False) # 0.6 support_only=False

    #np.savez("./test.npz",all_rules , )
    #utils.appendToNPZ("./test.npz")
    return all_rules, pos_rules , neg_rules # pickle this ?

def getDiscriminativeRules(all_rules, pos_label, neg_label ):

    def filterPosRules(all_rules, pos_label):
        positive = all_rules[all_rules['consequents'] == {pos_label}]
        positive = positive[positive['confidence'] >= 0.1] # confidence == 1
        positive = positive.sort_values(['confidence', 'support'], ascending=[False, False])

        seen = set()
        dropped = set()
        indexes_to_drop = []

        positive = positive.reset_index(drop=True)
        for i in positive.index:
            new_rule = positive.loc[[i]]['antecedents'].values[0]

            for seen_rule in seen:
                if seen_rule.issubset(new_rule):#new_rule.issubset(seen_rule) or seen_rule.issubset(new_rule):
                    indexes_to_drop.append(i)
                    break
            else:
                seen.add(new_rule)

        positive.drop(positive.index[indexes_to_drop], inplace=True )
        positiveRules = positive
        return positiveRules

    def filterNegRules(all_rules, neg_label):
        negative = all_rules[all_rules['consequents'] == {neg_label}]
        negative = negative[negative['confidence'] >= 0.1] # confidence == 1

        negative = negative.sort_values(['confidence', 'support'], ascending=[False, False])

        seen = set()
        dropped = set()
        indexes_to_drop = []

        negative = negative.reset_index(drop=True)
        for i in negative.index:
            new_rule = negative.loc[[i]]['antecedents'].values[0]

            for seen_rule in seen:
                if seen_rule.issubset(new_rule):#new_rule.issubset(seen_rule) or seen_rule.issubset(new_rule):
                    indexes_to_drop.append(i)
                    break
            else:
                seen.add(new_rule)

        negative.drop(negative.index[indexes_to_drop], inplace=True )
        negativeRules = negative
        return negativeRules
    
    positive = filterPosRules(all_rules, pos_label)
    negative = filterNegRules(all_rules, neg_label)
    
    positive['num-items'] = positive['antecedents'].map(lambda x: len(x))
    negative['num-items'] = negative['antecedents'].map(lambda x: len(x))
    positive['consequents'] = positive['consequents'].map(lambda x: pos_label)
    negative['consequents'] = negative['consequents'].map(lambda x: neg_label)

    both = pd.concat([positive, negative], ignore_index=True)

    #both = positive.append(negative, ignore_index=True)

    discr_rules = both[['antecedents', 'consequents', 'num-items', 'support', 'confidence', 'antecedent support']].sort_values(
        ['support', 'confidence', 'num-items'], ascending=[False, False, False])

    discr_rules = discr_rules.rename(columns={"antecedents": "itemset", "consequents": "label"})

    return discr_rules

def getCharasteristicRules(pos_rules, pos_label, neg_rules,neg_label ):

    def getRev_postive(pos_rules, pos_label):
        rev_positive = pos_rules[pos_rules['antecedents'] == {pos_label}]
        #rev_positive = pos_rules[pos_rules['consequents'] == {pos_label}] 
        rev_positive = rev_positive[rev_positive['confidence'] >= 0.7]
        rev_positive = rev_positive.sort_values(['confidence', 'support'], ascending=[False, False])

        seen = set()
        dropped = set()
        indexes_to_drop = []

        rev_positive = rev_positive.reset_index(drop=True)
        for i in rev_positive.index:
            new_rule = rev_positive.loc[[i]]['consequents'].values[0]

            for seen_rule, indx in seen:
                if seen_rule.issubset(new_rule):
                    indexes_to_drop.append(i)
                    break
            else:
                seen.add((new_rule, i))

        rev_positive.drop(rev_positive.index[indexes_to_drop], inplace=True )

        return rev_positive

    def getRev_negative(neg_rules,neg_label):

        rev_negative = neg_rules[neg_rules['antecedents'] == {neg_label}]
        #rev_negative = neg_rules[neg_rules['consequents'] == {neg_label}]
        rev_negative = rev_negative[rev_negative['confidence'] >= 0.7]
        rev_negative = rev_negative.sort_values(['confidence', 'support'], ascending=[False, False])

        seen = set()
        dropped = set()
        indexes_to_drop = []

        rev_negative = rev_negative.reset_index(drop=True)
        for i in rev_negative.index:
            new_rule = rev_negative.loc[[i]]['consequents'].values[0]

            for seen_rule, indx in seen:
                if seen_rule.issubset(new_rule):
                    indexes_to_drop.append(i)
                    break
            else:
                seen.add((new_rule, i))

        rev_negative.drop(rev_negative.index[indexes_to_drop], inplace=True )

        return rev_negative
    #
    rev_positive = getRev_postive(pos_rules, pos_label)
    rev_negative = getRev_negative(neg_rules,neg_label)

    rev_positive['num-items'] = rev_positive['consequents'].map(lambda x: len(x))
    rev_negative['num-items'] = rev_negative['consequents'].map(lambda x: len(x))
    rev_positive['antecedents'] = rev_positive['antecedents'].map(lambda x: pos_label)
    rev_negative['antecedents'] = rev_negative['antecedents'].map(lambda x: neg_label)

    #rev_both = rev_positive.append(rev_negative, ignore_index=True)
    rev_both = pd.concat([rev_positive, rev_negative], ignore_index=True)

    chr_rules = rev_both[['antecedents', 'consequents', 'num-items', 'support', 
                              'confidence', 'consequent support']].sort_values(
        ['support', 'confidence', 'num-items'], ascending=[False, False, False])

    chr_rules = chr_rules.rename(columns={"antecedents": "label", "consequents": "itemset"})

    chr_rules
    return  chr_rules

                                                            #TODO: predictions to predictionsList over iteration
def calculateRulesMetrics(rules_DF,featureDict, dataloader, predictions):#, dirPath , debug = False):
    """
    
    featureDict example: featureDict= {'Pregnancies':0, 'Glucose':1, 'BloodPressure':2, 'SkinThickness':3, 'Insulin':4,
                                        'BMI':5, 'DiabetesPedigreeFunction':6, 'Age':7}

    """
    X_List = []
    y_List = [] 
    for inputs, lables in dataloader:
        X_List.extend(inputs.detach())
        y_List.extend(lables.detach())
    
    def extractNlpRules_df(rules_DF):
        #print(rules_DF)
        #print(rules_DF)
        rulesList =rules_DF["itemset"].to_list()
        rulesList = [set(frozenset) for frozenset in rulesList]
        #print("ruleesss RAAAWWWWWW")
        #print(rulesList)
        labelList_rules = rules_DF["label"].apply(lambda x: ', '.join(list(x))).astype("unicode")
        labelList_rules = [set(frozenset).pop() for frozenset in labelList_rules]
        
        return rulesList, labelList_rules, rulesList # dont filter so RAW and "filtered" are same
     
    def extractRules_df(rules_DF):
        #print(rules_DF)
        #print(rules_DF)
        rulesList =rules_DF["itemset"].to_list()
        rulesList = [set(frozenset) for frozenset in rulesList]
        #print("ruleesss RAAAWWWWWW")
        #print(rulesList)
        labelList_rules = rules_DF["label"].apply(lambda x: ', '.join(list(x))).astype("unicode")
        labelList_rules = [set(frozenset).pop() for frozenset in labelList_rules]

        # Regular expression pattern
        #pattern = r"([+-]?\d+\.\d+)(<\w+<=)([+-]?\d+\.\d+)"

        #pattern = r"([+-]?\d+\.\d+)(<\w+<=)([+-]?\d+\.\d+)"
        pattern = r"([+-]?\d+\.\d+)(<\w+(\s*\w*)*<=)([+-]?\d+\.\d+)"


        # Filter out upper bound, lower bound, and category for each set
        rules_list = []
        for set_item in rulesList:
            set_filtered = []
            for item in set_item:
                matches = re.findall(pattern, item)
                if matches:

                    lower_bound, feature,_, upper_bound, = matches[0]
                    set_filtered.append((lower_bound, feature[1:-2], upper_bound))
            rules_list.append(set_filtered)

        #print(len(rules_list))
        return rules_list, labelList_rules, rulesList

    def applyNlpRulesOnData(X,predictions, rules, labelList_rules, featureDict):

        predictionComparisonList = []  

        for j in range(len(X)):
            tempPredictionList = []  
            rulesComplexityList = []
            for i in range(len(rules)): 
                # for all rules per sample # num samples * num rules
                                                # only works for one 

                conditionsAreMet = False
                rulesComplexityList.append(len(rules[i]))
                for k in range(len(rules[i])): # if rule consists of more than one tupel (3< insulin <=5 , 2< glucose <=7)
                    feature = list(rules[i])[k]
                    #print("feature")
                    #print(X[j][featureDict[int(feature)]].item)
                    #print(type(featureZ))

                    if  X[j][featureDict[int(feature)]] == 1:#
                        #print("a condition is met")
                        conditionsAreMet = True
                        # Rule is apicable
                    else:
                        # rules is not aplicable      
                        tempPredictionList.append(-1)
                        conditionsAreMet = False
                        break # if one condition of the rule is not aplicable break
                #print("lpp")
                #print(predictions[j])
                #print("as")
                #print(int(labelList_rules[i]))
                #print()
                if conditionsAreMet == True:
                    #print("ALL condition ARE met")
                    if predictions[j] ==   int(labelList_rules[i]):
                        # explaination model prediction is the same as trained Model prediction
                        tempPredictionList.append(1)
                    elif predictions[j] != int(labelList_rules[i]):
                        # explaination model prediction not the same as trained Model prediction
                        tempPredictionList.append(0)

            predictionComparisonList.append(tempPredictionList)
            return predictionComparisonList, rulesComplexityList
        
        
    def applyRulesOnData(X,predictions, rules, labelList_rules, featureDict):
        """
        X: List
        predictions: List
        rules : List 
        rulesLables : List

        """
        predictionComparisonList = []  

        for j in range(len(X)):
            tempPredictionList = []  
            rulesComplexityList = []
            for i in range(len(rules)): 
                # for all rules per sample # num samples * num rules
                                                # only works for one 

                conditionsAreMet = False
                rulesComplexityList.append(len(rules[i]))
                for k in range(len(rules[i])): # if rule consists of more than one tupel (3< insulin <=5 , 2< glucose <=7)
                    lowerBound, feature, upperBound = rules[i][k]

                    if float(lowerBound) < X[j][featureDict[feature]] <= float(upperBound):
                        conditionsAreMet = True
                        # Rule is apicable
                    else:
                        # rules is not aplicable      
                        tempPredictionList.append(-1)
                        conditionsAreMet = False
                        break # if one condition of the rule is not aplicable break

                if conditionsAreMet == True:
                    if predictions[j] ==  int(labelList_rules[i]):
                        # explaination model prediction is the same as trained Model prediction
                        tempPredictionList.append(1)
                    elif predictions[j] != int(labelList_rules[i]):
                        # explaination model prediction not the same as trained Model prediction
                        tempPredictionList.append(0)

            predictionComparisonList.append(tempPredictionList)
            return predictionComparisonList, rulesComplexityList

    #predictionList, rulesComplexityList = applyRulesOnData(X_List ,predictions, labelList_rules,  featureDict)
    #predictionList

    def globalCoverage(predictionComparisonList):
        isCoverdList = []
        correctPredictedList = [] 
        for i in predictionComparisonList:
            isCovered = False
            for j in i:
                if j == 0 or j ==1:
                    isCovered = True
                    isCoverdList.append(1) # wird von einer Rule abgedeckt
                    break
            if not isCovered:
               isCoverdList.append(-1)
        coverageList = isCoverdList.count(1) /len(isCoverdList) # instances covered by global explanation set
        return coverageList
    
    #globalCoverage = globalCoverage(predictionList)

    def rulePrecisionAndSupport(predictionComparisonList): 
        # recision of an applicable rule on test instances(precision)
        # AND how many instances does a rule cover (Support)

        predictionList_transposed =  np.array(predictionComparisonList).transpose()
        rulePrecisionList = [] 
        ruleSupportList = [] 

        for i in predictionList_transposed:
            tempCorrectClassified = list(i).count(1)
            tempFalseClassified =   list(i).count(0)
            

            #tempNotAplicable =     list(i).count(0) 
            try:
                rulePrecisionList.append(tempCorrectClassified/ len(predictionList_transposed[i]) )
            except ZeroDivisionError:
                rulePrecisionList.append(0)
            try:
                ruleSupportList.append((tempCorrectClassified + tempFalseClassified)/len(predictionList_transposed[i]) )
            except ZeroDivisionError:
                rulePrecisionList.append(0)


        return rulePrecisionList, ruleSupportList
    
    
    rules_list, labelList_rules, raw_rules = extractNlpRules_df(rules_DF)
    
    #rules_list, labelList_rules, raw_rules= extractRules_df(rules_DF)
    predictionComparisonList, rulesComplexityList = applyNlpRulesOnData(X_List,predictions, rules_list, labelList_rules, featureDict)
    #predictionComparisonList, rulesComplexityList = applyRulesOnData(X_List,predictions, rules_list, labelList_rules, featureDict)
    
    rulePrecisionList, ruleSupportList = rulePrecisionAndSupport(predictionComparisonList)
    coverageList = globalCoverage(predictionComparisonList)
    #rulePrecisionList, ruleSupportList = rulePrecisionAndSupport(predictionComparisonList)
    numberOfGeneratedRules = (len(rules_list))
    return rules_list, labelList_rules, rulePrecisionList, predictionComparisonList, rulesComplexityList , coverageList,  ruleSupportList,   numberOfGeneratedRules, raw_rules

    ## Get the current date and time
    #now = datetime.now()
#
    ## Format the date and time as a string
    #date_time_string = now.strftime("%Y-%m-%d %H:%M:%S")
#
    #
#
    #pathToNPZ =  dirPath + f"/rulesdata_{date_time_string}.npz"
    #np.savez(pathToNPZ ,rules_list = rules_list) 
    #utils.appendToNPZ(pathToNPZ, "labelList_rules", labelList_rules)
    #utils.appendToNPZ(pathToNPZ, "rulePrecisionList", rulePrecisionList)
    #utils.appendToNPZ(pathToNPZ, "predictionComparisonList", predictionComparisonList)
    #utils.appendToNPZ(pathToNPZ, "rulesComplexityList", rulesComplexityList)
    #utils.appendToNPZ(pathToNPZ, "coverageList", coverageList)
    #utils.appendToNPZ(pathToNPZ, "rulePrecisionList", rulePrecisionList)
    #utils.appendToNPZ(pathToNPZ, "ruleSupportList", ruleSupportList)
    #utils.appendToNPZ(pathToNPZ, "numberOfGeneratedRules", numberOfGeneratedRules)
    #utils.appendToNPZ(pathToNPZ, "rulePrecisionList", rulePrecisionList)



    # think does it work with NLP no problems ?? / maybe just run on other pipeline results  
    #   + if not fix for NLP 
    

def trackRulesList(rules_list_overIterations):
    """
        rules list   -> 
        [[1 1 1 0 0]
      ^ [1 0 0 1 0]
      | [0 1 1 0 1]]
    iterations

    """
    unique_items = set(item for sublist in rules_list_overIterations for item in sublist)

    item_to_index = {item: index for index, item in enumerate(unique_items)}

    num_items = len(unique_items)

    one_hot_matrix = []

    for sublist in rules_list_overIterations:
        sublist_encoded = [0] * num_items
    
        for tup in sublist:
            item, _ = tup
        
            index = item_to_index[item]
            sublist_encoded[index] = 1
    
        one_hot_matrix.append(sublist_encoded)
    one_hot_matrix = np.array(one_hot_matrix)

    print(one_hot_matrix)

###TODO:
#for two: overlay jaccardsimilarity and testing accuracy  

#for 3 : need to asses "positive influence" *** 

#for 3a: go back to set without the "stable" rule 
#and compare results with the last set without the rule
#with the ruleset with the rules"
#    *** ==> Net correct calssified : from changed predictions (now correct classified - wrong classified)
#
#fidelity / AUC check cega and glocalX paper
