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

def loadOHE_Rules(iterationNumber):
    """
    """
    path  = f"./OHEresults/ohe@Iteration_{iterationNumber}.pkl"
    with open(path, 'rb') as f:
        ohe_df =  pickle.load(f)

    return ohe_df

#only for binary 
                  #X_test
def calculateAndSaveOHE_Rules(data, featureNames,trainedModelPrediction_Test, grads, debug=False):

    """
    explainationGrads should have shape of [epochs * iterationsPerEpoch , testDatasetSize, featuresize ]


    ute: sonntag 25.jun  not this -> ( 23:13),  20:12 (essen?), 22:55  

    """
    data_df = pd.DataFrame(data, columns=featureNames)

    pos_label = '1'
    neg_label = '0'

    itemset = set()
    encoded_vals = []
    #summed_values = {}
    #num_features = data.shape[1]

    shap_threshold = 0.001
    num_cores = cpu_count()
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

    itemset.add(pos_label)
    itemset.add(neg_label)


    def CEGA(gradsPerIteration): 
        ### for every model iteration the gradients of whole test_dataset is calculated
        for indx in range(len(trainedModelPrediction_Test)): #  len test dataset
            pos_queue.put(pos_label)
            neg_queue.put(neg_label)
            exp = gradsPerIteration[indx]#[item[indx] for item in sample] #normalize featureListALL ?
            #print("exp")
            #print(np.shape(exp))
            #print(exp)

            instance_features = data_df.iloc[[indx]].to_dict(orient='records')[0]
            feature_vals = [instance_features[name] for name in featureNames] #put here grads# feature values ?? 

            # GRADS AS LOCAL EXPLAINATION #
            # 
            #print("eh")

            zipped = zip(exp, feature_vals,
                         featureNames, [shap_threshold]*len(featureNames))


            p.map(get_relevant_features, zipped)
            append_to_encoded_vals(pos_queue, itemset, encoded_vals)
            append_to_encoded_vals(neg_queue, itemset, encoded_vals)

            ohe_df = pd.DataFrame(encoded_vals)
            #print(ohe_df)
            #exit()
        return ohe_df #ohe_dfList.append(ohe_df)
    
    for i in tqdm (range(len(grads))):

        gradsPerIteration  = grads[i]
        output_filename = f'{output_directory}{output_base_filename}_{iterationCounter}.pkl'
        try:
            with open(output_filename, 'xb') as f:
                ohe_df = CEGA(gradsPerIteration)
                pickle.dump(ohe_df, f)
                iterationCounter += 1
        except FileExistsError:
            # If the file already exists, increment the counter and try again
            iterationCounter += 1
    # TAKES ~30 sec for 154 samples  



def runApriori(ohe_df,testDataLength, pos_label ,neg_label): # min thrshold add  to def input ?
                                            # 10/ len(pred)
    freq_items = apriori(ohe_df, min_support=(1/testDataLength), use_colnames=True, max_len=3)
    #print(len(freq_items))
    #print(freq_items)
    all_rules = association_rules(freq_items, metric="confidence", min_threshold=0.02, support_only=False) # 0.7 support_only=False
    #print(len(all_rules))
    #print(all_rules)                                                   # 10/ len(pred)
    freq_items = apriori(ohe_df.loc[ohe_df[pos_label] == 1], min_support=(1/testDataLength), use_colnames=True, max_len=10) # max len 3
    pos_rules = association_rules(freq_items, metric="confidence", min_threshold=0.02, support_only=False) # 0.6 support_only=False
                                                                        # 10/ len(pred)
    freq_items = apriori(ohe_df.loc[ohe_df[neg_label] == 1], min_support=(1/testDataLength), use_colnames=True, max_len=10) # max len 3 
    neg_rules = association_rules(freq_items, metric="confidence", min_threshold=0.02, support_only=False) # 0.6 support_only=False

    return all_rules, pos_rules , neg_rules





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


def calculateRulesMetrics(rules_DF,featureDict, dataloader, predictions):#, dirPath , debug = False):
    """
    
    featureDict example: featureDict= {'Pregnancies':0, 'Glucose':1, 'BloodPressure':2, 'SkinThickness':3, 'Insulin':4,
                                        'BMI':5, 'DiabetesPedigreeFunction':6, 'Age':7}

    """
    X_List = []
    y_List = [] 
    #print(testData[1])
    for inputs, lables in dataloader:
        X_List.extend(inputs)
        y_List.extend(lables)
    
    def extractRules_df(rules_DF):
        rulesList =rules_DF["itemset"].to_list()
        rulesList = [set(frozenset) for frozenset in rulesList]

        labelList_rules = rules_DF["label"].apply(lambda x: ', '.join(list(x))).astype("unicode")
        labelList_rules = [set(frozenset).pop() for frozenset in labelList_rules]

        # Regular expression pattern
        pattern = r"([+-]?\d+\.\d+)(<\w+<=)([+-]?\d+\.\d+)"


        # Filter out upper bound, lower bound, and category for each set
        rules_list = []
        for set_item in rulesList:
            set_filtered = []
            for item in set_item:
                matches = re.findall(pattern, item)
                if matches:
                    lower_bound, feature, upper_bound = matches[0]
                    set_filtered.append((lower_bound, feature[1:-2], upper_bound))
            rules_list.append(set_filtered)

        return rules_list, labelList_rules

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
                rulePrecisionList.append(len(predictionList_transposed) / tempCorrectClassified)
            except ZeroDivisionError:
                #print("ZeroDivisionError")
                rulePrecisionList.append(0)
            try:
                ruleSupportList.append(len(predictionList_transposed)/ (tempCorrectClassified + tempFalseClassified))
            except ZeroDivisionError:
                #print("ZeroDivisionError")
                rulePrecisionList.append(0)


        return rulePrecisionList, ruleSupportList
    
    


    rules_list, labelList_rules =  extractRules_df(rules_DF)
    predictionComparisonList, rulesComplexityList = applyRulesOnData(X_List,predictions, rules_list, labelList_rules, featureDict)

    rulePrecisionList, ruleSupportList = rulePrecisionAndSupport(predictionComparisonList)
    coverageList = globalCoverage(predictionComparisonList)
    rulePrecisionList, ruleSupportList = rulePrecisionAndSupport(predictionComparisonList)
    numberOfGeneratedRules = (len(rules_list))

    return rules_list, labelList_rules, rulePrecisionList, predictionComparisonList, rulesComplexityList , coverageList,  ruleSupportList,   numberOfGeneratedRules,

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


    # save to pickle k (not tested)
    # check if works 
    # do it in a for loop for multiple global rulesSets
    #jaccard similarity
    # think does it work with NLP no problems ?? / maybe just run on other pipeline results  
    #   + if not fix for NLP 
    #print(predictionList)
    
