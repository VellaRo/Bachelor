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
import torch
def loadOHE_Rules(iterationNumber):
    """
    """
    path  = f"./OHEresults/ohe@Iteration_{iterationNumber}.pkl"
    with open(path, 'rb') as f:
        ohe_df =  pickle.load(f)

    return ohe_df

def jaccard_similarity(list1, list2):
    #print(list1)
    #print(list2)
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
    
    return OHE_DF, featureNames

#only for binary 
                  #X_test
def calculateAndSaveOHE_Rules(data, featureNames,trainedModelPrediction_Test, grads, datasetType ,debug=False, vocab = None):

    """
    explainationGrads should have shape of [epochs * iterationsPerEpoch , testDatasetSize, featuresize ]


    """
    data_df = pd.DataFrame(data, columns=featureNames)

    #pos_label = '1'#"1"
    #neg_label = '0'#"0"

    pos_label = 0
    neg_label = 1   #'0'

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
        print("numerical")
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


    else: 
        #data_df, featureNames = categoricalToOHE(data_df)
        data_df, featureNames = vocabToOHE(data_df, vocab)


        for feature in data_df.columns.to_list():
            itemset.add(str(feature))
    itemset.add(pos_label)
    itemset.add(neg_label)


    def CEGA(gradsPerIteration): 
        ### for every model iteration the gradients of whole test_dataset is calculated
        for indx in tqdm (range(len(gradsPerIteration))):#(len(trainedModelPrediction_Test)): #  len test dataset
            pos_queue.put(pos_label)
            neg_queue.put(neg_label)
            exp = gradsPerIteration[indx]#[item[indx] for item in sample] #normalize featureListALL ?

            instance_features = data_df.iloc[[indx]].to_dict(orient='records')[0] 
    
            feature_vals = [instance_features[name] for name in featureNames] #put here grads#   feature values ?? 
       
            zipped = zip(exp, feature_vals,
                         featureNames, [shap_threshold]*len(featureNames))


            p.map(get_relevant_features, zipped)
            append_to_encoded_vals(pos_queue, itemset, encoded_vals)
            append_to_encoded_vals(neg_queue, itemset, encoded_vals)

            ohe_df = pd.DataFrame(encoded_vals)

        return ohe_df #ohe_dfList.append(ohe_df)
    
    for i in tqdm (range(len(grads))):
        gradsPerIteration  = grads[i]
        output_filename = f'{output_directory}{output_base_filename}_{iterationCounter}.pkl'
        
        print(i)
        with open(output_filename, 'wb') as f:
            ohe_df = CEGA(gradsPerIteration)#,trainedModelPrediction_TestPerIteration)
            pickle.dump(ohe_df, f)
        iterationCounter += 1
        encoded_vals = []

    return featureNames
import gely

import myAssociationRules

def runApriori(ohe_df,testDataLength, pos_label ,neg_label): # min thrshold add  to def input ?
    
    def gelyOutputToDF(gelyOutput):
        # Create a dictionary to hold the data in the required format
        data_dict = {"support": [], "itemsets": []}

        # Extract the itemset and support values from the input and populate the dictionary
        for itemsets, support in gelyOutput:
            itemsets = {str(item) for item in itemsets} # convert to string
            data_dict["itemsets"].append(frozenset(itemsets))
            #mapping = {idx: item for idx, item in enumerate(["support", "itemsets"])}

           # data_dict['itemsets'] = data_dict["itemsets"].apply(lambda x: frozenset([mapping[i] for i in x]))
            data_dict["support"].append(0.5)#support / testDataLength)


        # Create a pandas DataFrame using the dictionary
        freq_items = pd.DataFrame(data_dict)
        ## Create a new column to store the length of each frozenset
        #freq_items['itemset_length'] = freq_items['itemsets'].apply(lambda x: len(x))
#
        ## Sort the DataFrame based on the "itemset_length" column
        #freq_items = freq_items.sort_values(by='itemset_length')
#
        ## Drop the temporary "itemset_length" column if you don't need it in the final DataFrame
        #freq_items = freq_items.drop(columns='itemset_length')
        #freq_items.reset_index(drop=True, inplace=True)
        return freq_items


                                            # 10/ len(pred)10/testDataLength*5
    freq_items1 = apriori(ohe_df, min_support=(0.00000000000000001), use_colnames=True, max_len=3, low_memory=True) #[array[0.5, frozenset]]

    #all_rules = myAssociationRules.association_rules(freq_items1, metric="confidence", min_threshold=0.0 ,support_only=True) # 0.7 support_only=False

    #from sympy import symbols, solve, Eq 
#
#
    #x = symbols('x')
    #expr1 = Eq(x / len(ohe_df) -0.1)
    #thresholdGely1 =solve(expr1)[0]
    #expr1 = Eq(x / len(ohe_df.loc[ohe_df[pos_label] == 1]) -0.1)
    #thresholdGely2 =solve(expr1)[0]
    #expr1 = Eq(x / len(ohe_df.loc[ohe_df[neg_label] == 1]) -0.1)
    #thresholdGely3 =solve(expr1)[0]

    #freq_items = gely.gely(ohe_df.values, thresholdGely1)#remove_copmlete_transactions=False) 
    #freq_items = gelyOutputToDF(freq_items)

    all_rules = myAssociationRules.association_rules(freq_items1, metric="confidence", min_threshold=0.0 ,support_only=False) # 0.7 support_only=False
    
                              # 10/ len(pred)10/testDataLength*5
    freq_items1 = apriori(ohe_df.loc[ohe_df[pos_label] == 1], min_support=(0.00000000000000001), use_colnames=True, max_len=3 , low_memory=True) # max len 3
    #freq_items = gely.gely(ohe_df.loc[ohe_df[pos_label] == 1].values, thresholdGely2 )#,remove_copmlete_transactions=False) 
    #freq_items = gelyOutputToDF(freq_items)

    pos_rules = myAssociationRules.association_rules(freq_items1, metric="confidence", min_threshold=0.0, support_only=False) # 0.6 support_only=False
    #freq_items = gely.gely(ohe_df.loc[ohe_df[neg_label] == 1].values, thresholdGely3)#,remove_copmlete_transactions=False) 
    #freq_items = gelyOutputToDF(freq_items)
                                                                    # 10/ len(pred)10/testDataLength*5
    freq_items1 = apriori(ohe_df.loc[ohe_df[neg_label] == 1], min_support=(0.00000000000000001), use_colnames=True, max_len=3, low_memory=True) # max len 3 
    neg_rules = myAssociationRules.association_rules(freq_items1, metric="confidence", min_threshold=0.0 , support_only=False) # 0.6 support_only=False
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
def calculateRulesMetrics(rules_DF,featureDict, dataloader, predictions, datasetType):#, dirPath , debug = False):
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
        rulesList =rules_DF["itemset"].to_list()
        rulesList = [set(frozenset) for frozenset in rulesList]

        labelList_rules =  rules_DF["label"].to_list()# rules_DF["label"].apply(lambda x: ', '.join(list(x))).astype("unicode")

        labelList_rules = [frozenset for frozenset in labelList_rules] #[set(frozenset).pop() for frozenset in labelList_rules]
        
        #print(labelList_rules)
        return rulesList, labelList_rules, rulesList # dont filter so RAW and "filtered" are same
     
    def extractRules_df(rules_DF):
        rulesList =rules_DF["itemset"].to_list()
        rulesList = [set(frozenset) for frozenset in rulesList]

        labelList_rules =  rules_DF["label"].to_list()# rules_DF["label"].apply(lambda x: ', '.join(list(x))).astype("unicode")

        labelList_rules = [frozenset for frozenset in labelList_rules] #[set(frozenset).pop() for frozenset in labelList_rules]
        
        #rulesList =rules_DF["itemset"].to_list()
        #rulesList = [set(frozenset) for frozenset in rulesList]
        #labelList_rules = rules_DF["label"].apply(lambda x: ', '.join(list(x))).astype("unicode")
        #labelList_rules = [set(frozenset).pop() for frozenset in labelList_rules]

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

        return rules_list, labelList_rules, rulesList

    def applyNlpRulesOnData(X,predictions, rules, labelList_rules, featureDict):

        predictionComparisonList = []  

        for j in range(len(X)):

            tempPredictionList = []  
            rulesComplexityList = []
            for i in range(len(rules)): 

                conditionsAreMet = False
                rulesComplexityList.append(len(rules[i]))
                for k in range(len(rules[i])): # if rule consists of more than one tupel (3< insulin <=5 , 2< glucose <=7)
                    feature = list(rules[i])[k]


                    if  X[j][featureDict[int(feature)]] == 1:#
                        #print("a condition is met")
                        conditionsAreMet = True
                        # Rule is apicable
                    else:
                        # rules is not aplicable      
                        tempPredictionList.append(-1)
                        conditionsAreMet = False
                        break # if one condition of the rule is not aplicable break
                if conditionsAreMet == True:
                    
                    if predictions[j] ==   int(labelList_rules[i]):
                        # explaination model prediction is the same as trained Model prediction
                        tempPredictionList.append(1)
                    elif predictions[j] != int(labelList_rules[i]):
                        # explaination model prediction not the same as trained Model prediction
                        tempPredictionList.append(0)

            predictionComparisonList.append(tempPredictionList)
            #print("len(predictionComparisonList)")
            #print(len(predictionComparisonList))

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

    def getGlobalCoverage(predictionComparisonList):
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
        globalCoverage = isCoverdList.count(1) /len(isCoverdList) # instances covered by global explanation set
        return globalCoverage
    
    #globalCoverage = globalCoverage(predictionList)

    def rulePrecisionAndSupport(predictionComparisonList, rules_list, labelList_rules): 
        # recision of an applicable rule on test instances(precision)
        # AND how many instances does a rule cover (Support)
        import sys

        epsilon = sys.float_info.epsilon
        predictionComparisonList_transposed =  np.array(predictionComparisonList).transpose()

        rulePrecisionListPerRule = []

        ruleSupportListPerRule =[]
        newRulesList = []
        newLabelList = []

        rulePrecisionListPerRule_NOTFILTERED =[]
        ruleSupportListPerRule_NOTFILTERED =[]
        for indx,i in enumerate(predictionComparisonList_transposed): #

            ruleTempCorrectClassified = list(i).count(1)
            ruleTempFalseClassified =   list(i).count(0)
            ruleTempNotAppliable =   list(i).count(-1)

            accuracyOfAppliableRules = (ruleTempCorrectClassified / ((ruleTempCorrectClassified + ruleTempFalseClassified) +epsilon))
            supportOfRule = (ruleTempCorrectClassified + ruleTempFalseClassified)/ ((ruleTempCorrectClassified +ruleTempFalseClassified +ruleTempNotAppliable) + epsilon) 

            import math
            if accuracyOfAppliableRules * math.sqrt(supportOfRule) >= 0.0:#0.5:
                rulePrecisionListPerRule.append(accuracyOfAppliableRules)
                ruleSupportListPerRule.append(supportOfRule)
                newRulesList.append(rules_list[indx])
                newLabelList.append(labelList_rules[indx])

            rulePrecisionListPerRule_NOTFILTERED.append(accuracyOfAppliableRules)
            ruleSupportListPerRule_NOTFILTERED.append(supportOfRule)
        print("len(rulePrecisionListPerRule)")
        print(len(rulePrecisionListPerRule))
        print(len(ruleSupportListPerRule))    

        return ruleSupportListPerRule ,rulePrecisionListPerRule , newRulesList , newLabelList, rulePrecisionListPerRule_NOTFILTERED ,ruleSupportListPerRule_NOTFILTERED  #newPredictionComparisonList


    
    
    

    #rules_list =rules_list[0:10]
    #labelList_rules =labelList_rules[0:10]

    #rules_list, labelList_rules, raw_rules= extractRules_df(rules_DF)
    if datasetType == "NLP":
        rules_list, labelList_rules, raw_rules = extractNlpRules_df(rules_DF)
        predictionComparisonList, rulesComplexityList = applyNlpRulesOnData(X_List,predictions, rules_list, labelList_rules, featureDict)
    else:
        rules_list, labelList_rules, raw_rules = extractRules_df(rules_DF)
        predictionComparisonList, rulesComplexityList = applyRulesOnData(X_List,predictions, rules_list, labelList_rules, featureDict)
        #predictionComparisonList, rulesComplexityList = applyRulesOnData(X_List,predictions, rules_list, labelList_rules, featureDict)   
    
    # globalRulePrecisionList, globalRuleSupportList ,rulePrecisionListPerRule , rules_list , labelList_rules,predictionComparisonList  = rulePrecisionAndSupport (predictionComparisonList, 0.9, rules_list, labelList_rules, datasetType)
    ruleSupportListPerRule ,rulePrecisionListPerRule , newRulesList , newLabelList, rulePrecisionListPerRule_NOTFILTERED ,ruleSupportListPerRule_NOTFILTERED   = rulePrecisionAndSupport(predictionComparisonList, rules_list, labelList_rules)
    if datasetType == "NLP":

        predictionComparisonList, rulesComplexityList = applyNlpRulesOnData(X_List,predictions, newRulesList, newLabelList, featureDict)
    else:
        predictionComparisonList, rulesComplexityList = applyRulesOnData(X_List,predictions, newRulesList, newLabelList, featureDict)
        #globalRulePrecisionList, globalRuleSupportList, rulePrecisionListPerRule = rulePrecisionAndSupport(predictionComparisonList)
    globalCoverage = getGlobalCoverage(predictionComparisonList) # before


    #print(rulePrecisionListPerRule)
    #print(len(rulePrecisionListPerRule))
    #print(globalCoverage)
    #print("----")
    #rules_list, labelList_rules = filterPrecision(1.0, rules_list, labelList_rules,rulePrecisionListPerRule, predictionComparisonList)
    #print("after Filter")
    #predictionComparisonList, rulesComplexityList = applyNlpRulesOnData(X_List,predictions, rules_list, labelList_rules, featureDict)
    #and filter
    #globalRulePrecisionList, globalRuleSupportList ,rulePrecisionListPerRule , rules_list , labelList_rules = rulePrecisionAndSupport (predictionComparisonList, 0.9, rules_list, labelList_rules,rulePrecisionListPerRule)
    #globalCoverage = getGlobalCoverage(predictionComparisonList) # after:4
    #print(rulePrecisionListPerRule)
    #print(len(rulePrecisionListPerRule))
    
    #filteredRulesList,filteredRulesLableList = filterPrecision(0.9, rules_list, labelList_rules,rulePrecisionListPerRule, predictionComparisonList)

    #print(globalCoverage)
    #rulePrecisionList, ruleSupportList = rulePrecisionAndSupport(predictionComparisonList)
    numberOfGeneratedRules = (len(newRulesList))

    #print.pes
    #return rules_list, labelList_rules, globalRulePrecisionList, predictionComparisonList, rulesComplexityList , globalCoverage,  globalRuleSupportList,   numberOfGeneratedRules, raw_rules, rulePrecisionListPerRule
    return newRulesList, newLabelList, predictionComparisonList, rulesComplexityList , globalCoverage,  ruleSupportListPerRule, numberOfGeneratedRules, raw_rules, rulePrecisionListPerRule, rulePrecisionListPerRule_NOTFILTERED ,ruleSupportListPerRule_NOTFILTERED
     
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
    

def trackRulesList(rules_list_overIterations, preciscionList):
    """
        rules list   -> 
        [[1 1 1 0 0]
      ^ [1 0 0 1 0]
      | [0 1 1 0 1]]
    iterations

    """
    #print(rules_list_overIterations)
    
    rulesTemp = [rules for rules in rules_list_overIterations]
    
    #print(rulesTemp)
    #print(type(rulesTemp))  
    
    #for i in rulesTemp:
    #    print(i)
    #unique_rules = set().union(*[set.union(*lst) for lst in rulesTemp])
    #unique_rules = (set(rule) for rule in rulesTemp)
        #item for rule in 
    #print(unique_rules)
    # Initialize an empty set to store the tuples
    unique_rules = set()

    # Iterate through each element in the nested lists
    for rule in rulesTemp:
        for s in rule:
            # Convert the set to a tuple and add it to the result set
            unique_rules.add(tuple(s))
    #print(unique_rules)
    item_to_index = {item: index for index, item in enumerate(unique_rules)}

    num_uniqueRules = len(unique_rules)

    one_hot_matrix = []
    precsicionDict = [None] * num_uniqueRules 
    for i,rules in enumerate(rules_list_overIterations): # rulesSet for each iteration 
        #print(type(rules))
        #print(rules)
        rule_encoded = [0] * num_uniqueRules
        counter = 0
        for rule in rules:
            #print(rule)
            #print(type(rule))
            index = item_to_index[tuple(rule)]
            rule_encoded[index] = 1
            print("rule_encoded[index]") 
            print(rule_encoded[index])
            if rule_encoded[index] == 1:
                print("index")
                print(index)
                precsicionDict[index] = preciscionList[i][counter]
                print("counter")
                print(counter)
                counter +=1
        one_hot_matrix.append(rule_encoded)

    one_hot_matrix = np.array(one_hot_matrix)

    print(one_hot_matrix)
    print(precsicionDict)

    trackedRules_OHE = one_hot_matrix
    return trackedRules_OHE , precsicionDict

def runCEGA(dirPath, modelsDirPath, model, X_test, device, data,date_time_string, test_set , datasetType,vocab=None ):

    # get predictionsOver iterations
    # filter out any special models
    r = re.compile("^[0-9]*$")
    modelsDirFiltered = list(filter(r.match, os.listdir(modelsDirPath))) # remove any special models

    
    trainedModelPrediction_Test_overIterations = []
    for modelNumber,filename in enumerate(np.sort(list(eval(i) for i in modelsDirFiltered))): #(os.listdir(modelsDirPath)))): # iterations time 
        model.load_state_dict(torch.load(modelsDirPath + "/" +str(filename)))
        model.eval()
        tempTrainedModelPrediction_Test = model.predict(X_test.to(device))
        trainedModelPrediction_Test_overIterations.append(tempTrainedModelPrediction_Test)

    featureNames = []
    #for i in range(len(data["testGradientsPerSamplePerFeature"][-1])): #vocab_size
    gradsTemp = data["testGradientsPerSamplePerFeature"]
    print(np.shape(gradsTemp))
    for i in range(len(gradsTemp[1][-1])):
    
        featureNames.append(str(i))
        #print(featureNames)
    #print.print()
    featureNames = calculateAndSaveOHE_Rules(X_test, featureNames,trainedModelPrediction_Test_overIterations[-1], data["testGradientsPerSamplePerFeature_iteration"], datasetType,debug= False, vocab=vocab) #OHEresults


    import warnings
    warnings.filterwarnings('ignore')
    #     frequent_itemsets = apriori(basket_sets.astype('bool'), min_support=0.07, use_colnames=True) https://stackoverflow.com/questions/74114745/how-to-fix-deprecationwarning-dataframes-with-non-bool-types-result-in-worse-c
    debug = False


    pos_label = 0
    neg_label = 1#'0'

    featureDict = {feature: index  for index, feature in enumerate(featureNames)} #{feature: int(feature)  for feature in featureNames}#{feature: index  for index, feature in enumerate(featureNames)}



    discriminative_rules_overIterations = []
    characteristic_rules_overIterations = []
    rules_list_overIterations   = []
    labelList_rules_overIterations = []
    rulePrecisionList_overIterations =[]
    predictionComparisonList_overIterations = []
    rulesComplexityList_overIterations = []
    globalCoverageList_overIterations = []    
    ruleSupportList_overIterations = []
    numberOfGeneratedRules_overIterations = []
    jaccardSimilarity_overIterations = []
    cosineSimilarity_overIterations = []
    diceSimilarity_overIterations = []
    overlapSimilarity_overIterations = []
    raw_rules_overIterations = []
    numberOfGeneratedRulesRAW_overIterations =[]
    rulePrecisionListPerRule_overIterations = []
    rulePrecisionListPerRule_overIterations_NOTFILTERED =[]
    ruleSupportList_overIterations_NOTFILTERED = []
    tempRules_list = None
    
    from tqdm import tqdm
    for i in tqdm(range(len(os.listdir("./OHEresults/")))):
        ohe_df = loadOHE_Rules(i)

        all_rules, pos_rules , neg_rules =  runApriori(ohe_df,len(X_test), pos_label ,neg_label)
        discriminative_rules = getDiscriminativeRules(all_rules, pos_label, neg_label )
        characteristic_rules = getCharasteristicRules(pos_rules, pos_label, neg_rules,neg_label )

        resultName = "discriminative_rules"

        rules_list, labelList_rules, predictionComparisonList, rulesComplexityList , globalCoverage,  ruleSupportListPerRule, numberOfGeneratedRules, raw_rules, rulePrecisionListPerRule, rulePrecisionListPerRule_NOTFILTERED ,ruleSupportListPerRule_NOTFILTERED= calculateRulesMetrics(discriminative_rules, featureDict, test_set, trainedModelPrediction_Test_overIterations[i], datasetType)
        discriminative_rules_overIterations.append(discriminative_rules)
        characteristic_rules_overIterations.append(characteristic_rules) 


        rules_list_overIterations.append(rules_list)
        labelList_rules_overIterations.append(labelList_rules)
        rulePrecisionList_overIterations.append(rulePrecisionListPerRule)

        predictionComparisonList_overIterations.append(predictionComparisonList)

        rulesComplexityList_overIterations.append(rulesComplexityList)
        globalCoverageList_overIterations.append(globalCoverage)
        ruleSupportList_overIterations.append(ruleSupportListPerRule)
        numberOfGeneratedRules_overIterations.append(numberOfGeneratedRules)
        raw_rules_overIterations.append(raw_rules)
        numberOfGeneratedRulesRAW_overIterations.append(len(raw_rules))
        rulePrecisionListPerRule_overIterations.append(rulePrecisionListPerRule)
        rulePrecisionListPerRule_overIterations_NOTFILTERED.append(rulePrecisionListPerRule_NOTFILTERED)
        ruleSupportList_overIterations_NOTFILTERED.append(ruleSupportListPerRule_NOTFILTERED)


        if tempRules_list is not None:
            #print("not Jaccard")
            jaccardSimilarity_overIterations.append(jaccard_similarity(rules_list , tempRules_list))
            #cosineSimilarity_overIterations.append(cosine_similarity(rules_list , tempRules_list))
            diceSimilarity_overIterations.append(dice_similarity(rules_list , tempRules_list))
            overlapSimilarity_overIterations.append(overlap_coefficient(rules_list , tempRules_list))
            #jaccardSimilarity_overIterations.append(cosine_similarity(rules_list , tempRules_list))

        tempRules_list = rules_list

    if debug:
        pathToNPZ =  dirPath + f"DEBUG.npz"
    else:    
        pathToNPZ =  dirPath +"NLP_Results/rulesResults/" f"{resultName}/_{date_time_string}.npz"

    np.savez(pathToNPZ ,rules_list_overIterations = rules_list_overIterations) 
    utils.appendToNPZ(pathToNPZ, "labelList_rules_overIterations", labelList_rules_overIterations)
    utils.appendToNPZ(pathToNPZ, "rulePrecisionList_overIterations", rulePrecisionList_overIterations)
    utils.appendToNPZ(pathToNPZ, "predictionComparisonList_overIterations", predictionComparisonList_overIterations)
    utils.appendToNPZ(pathToNPZ, "rulesComplexityList_overIterations", rulesComplexityList_overIterations)
    utils.appendToNPZ(pathToNPZ, "globalCoverageList_overIterations", globalCoverageList_overIterations)
    utils.appendToNPZ(pathToNPZ, "ruleSupportList_overIterations", ruleSupportList_overIterations)
    utils.appendToNPZ(pathToNPZ, "numberOfGeneratedRules_overIterations", numberOfGeneratedRules_overIterations)
    utils.appendToNPZ(pathToNPZ, "jaccardSimilarity_overIterations", jaccardSimilarity_overIterations)
    utils.appendToNPZ(pathToNPZ, "cosineSimilarity_overIterations", cosineSimilarity_overIterations)
    utils.appendToNPZ(pathToNPZ, "overlapSimilarity_overIterations", overlapSimilarity_overIterations)
    utils.appendToNPZ(pathToNPZ, "diceSimilarity_overIterations", diceSimilarity_overIterations)
    utils.appendToNPZ(pathToNPZ, "raw_rules_overIterations", raw_rules_overIterations)
    utils.appendToNPZ(pathToNPZ, "numberOfGeneratedRulesRAW_overIterations", numberOfGeneratedRulesRAW_overIterations) 
    utils.appendToNPZ(pathToNPZ, "rulePrecisionListPerRule_overIterations", rulePrecisionListPerRule_overIterations)
    utils.appendToNPZ(pathToNPZ, "rulePrecisionListPerRule_overIterations_NOTFILTERED", rulePrecisionListPerRule_overIterations_NOTFILTERED)
    utils.appendToNPZ(pathToNPZ, "ruleSupportList_overIterations_NOTFILTERED", ruleSupportList_overIterations_NOTFILTERED)

    #rulePrecisionListPerRule_overIterations_NOTFILTERED =[]
    #ruleSupportList_overIterations_NOTFILTERED = []
    return pathToNPZ
###TODO:
#  
#for 3 : need to asses "positive influence" *** 

#for 3a: go back to set without the "stable" rule 
#and compare results with the last set without the rule
#with the ruleset with the rules"
#    *** ==> Net correct calssified : from changed predictions (now correct classified - wrong classified)
#
#fidelity / AUC check cega and glocalX paper
