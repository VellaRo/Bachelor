from time import time

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

from datasetsNLP import get_agnews
from modelsNLP import SentenceCNN, BiLSTMClassif
import evalModel
import plotResults
from matplotlib import pyplot as plt
import pickle   
def _get_outputs(inference_fn, data, model, device, batch_size=256):

    _data = DataLoader(TensorDataset(data), shuffle=False, batch_size=batch_size)

    try:
        _y_out = []
        for x in _data:
            _y = inference_fn(x[0].to(device))
            _y_out.append(_y.cpu())
        return torch.vstack(_y_out)
    except RuntimeError as re:
        if "CUDA out of memory" in str(re):
            model.to('cpu')
            outputs = _get_outputs(inference_fn, data, model, 'cpu')
            model.to('cuda')
            return outputs
        else:
            raise re


def _get_predictions(inference_fn, data, model, device):
    return torch.argmax(_get_outputs(inference_fn, data, model, device), dim=1)


def validate(inference_fn, model, X, Y):

    if inference_fn is None:
        inference_fn = model

    model.eval()
    device = next(model.parameters()).device

    _y_pred = _get_predictions(inference_fn, X, model, device)
    model.train()

    acc = torch.mean((Y == _y_pred).to(torch.float)).detach().cpu().item()  # mean expects float, not bool (or int)
    return acc

def train_loop(model, optim, loss_fn, tr_data: DataLoader, te_data: tuple, inference_fn=None, \
               n_batches_max=10, device='cuda'):
    
#    print(device)
    model.to(device)
    acc_val = []
    losses = []
    n_batches = 0
    _epochs, i_max = 0, 0
    accs = []
    
    # save model
    import os
    import shutil
    modelsdirPath = "./NLP_Models"

    if os.path.exists(modelsdirPath) and os.path.isdir(modelsdirPath):
        shutil.rmtree(modelsdirPath)

    os.mkdir(modelsdirPath)
    epoch_counter = 0
    iterationCounter = 0
    total_gradientsList = []
    while n_batches <= n_batches_max:
        for i, (text, labels) in enumerate(tr_data, 0):

            #break
            acc = validate(inference_fn, model, *te_data)
            accs.append(acc)
            if i % 100 == 0:
                print(f"test acc @ batch {i+_epochs*i_max}/{n_batches_max}: {acc:.4f}")
            text = text.to(device)
            labels = labels.to(device)
            out = model(text)
            loss = loss_fn(out, labels)
            optim.zero_grad()
            loss.backward()
            # Sum up the gradients of the weights in the neural network
            
            total_gradients = 0.0   
            # no softmax
            for param in model.parameters():                
                if param.grad is not None:
                    #print(param.grad.numel())
                    #print(param.grad)

                    total_gradients += (torch.abs(param.grad).sum() / param.grad.numel())
                    #print(total_gradients)
            total_gradientsList.append(total_gradients.cpu())


            # save model
            torch.save(model.state_dict(), modelsdirPath +"/"+str(iterationCounter))
            iterationCounter += 1
            #
            optim.step()
            losses.append(loss.item())
            
            n_batches += 1
            if n_batches > n_batches_max:
                break
        i_max = i
    #copy and name special Models 
    shutil.copyfile(modelsdirPath +"/"+str(0), modelsdirPath +"/initialModel")
    shutil.copy(modelsdirPath +"/"+str(iterationCounter -1), modelsdirPath +"/finalModel") # rename macht probleme ???

    print("NOTE: THESE SAVED MODELS ARE BEEING OVERWRITTEN ON NEXT RUN")
    ##
    acc_val.append(validate(inference_fn, model, *te_data))
    print("accuracies over test set")
    print(acc_val)
    return model, losses, accs , total_gradientsList


if __name__ == '__main__':
    
    #SETUP

    size_train_batch = 64
    size_test_batch = 20 # 3h apriori
    n_batches = 6
    embedding_dim = 128
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # get dataset and dataloader
    train_set, test_set, size_vocab, n_classes, vocab = get_agnews(random_state=42, batch_sizes=(size_train_batch, size_test_batch))

    X_test, Y_test = next(iter(test_set))  # only use first batch as a test set
    

    #Y_test_distr = torch.bincount(Y_test, minlength=n_classes)/size_test_batch
    #print(f"class distribution in test set: {Y_test_distr}")  # this should roughly be uniformly distributed

    #model = BiLSTMClassif(n_classes=n_classes, embed_dim=embedding_dim, vocab_size=size_vocab, hid_size=64)
    model = SentenceCNN(n_classes=n_classes, embed_dim=embedding_dim, vocab_size=size_vocab) # learns faster then LSTM
    optimizer = Adam(model.parameters())
    loss_fun = torch.nn.CrossEntropyLoss()

    _t_start = time()

    model, loss, test_accuracies , total_gradientsList= \
        train_loop(model, optimizer, loss_fun, train_set, (X_test, Y_test),
                   inference_fn=model.forward_softmax, device=device, n_batches_max=n_batches)
    
    dirPath ="./"
    modelsDirPath = dirPath + "NLP_Models"

    loaderList = [test_set] # testLoader
    nameList = ["test"]
    yList = [Y_test]

    inputFeatures = size_vocab  
    num_epochs = n_batches # just for tracking progress

    print("eval")

    evalModel.doALLeval(model, modelsDirPath,dirPath, loaderList, device,optimizer, loss_fun, num_epochs, nameList, yList, inputFeatures, NLP=True)

    dataPath= dirPath+ "NLP_Results/Trainingresults/"
    
    print("plotting...")
    import utils 
    import numpy as np

    print("cosine_similarity")
    plotResults.plotCosineSimilarity(dataPath, "cosine_simialarity", set="test")
    print("percentageWeightsSignDifference3")
    plotResults.plotWeightSignDifferences(dataPath, "percentageWeightsSignDifference3" , "test")
    print("weightsMagnitude3")
    plotResults.plotWeightMagnitude(dataPath, "weightsMagnitude3","test")
    print("L2Distance3")
    plotResults.plotL2Distance(dataPath, "L2Distance3","test")
    print("weightTrace3")
    plotResults.plotWeightTrace(dataPath, "weightTrace3","test")    
    print("averageGradientMagnitude3")
    plotResults.plotGradientMagnitude(dataPath, "averageGradientMagnitude3","test", perFeature=False)
    print("GradientMagnitudePerFeature3")
    plotResults.plotGradientMagnitude(dataPath, "GradientMagnitudePerFeature3","test", perFeature=True)
    
    print("total_gradientsList")
    plotResults.plotTotalGradientMagnitude(total_gradientsList,dataPath, "total_gradientsList","test")

    figAcc, axsAcc = plt.subplots(nrows=1, ncols=1)

    data = utils.loadData(dataPath+ "data.npz")

    axsAcc.set_title("testAccuracyPerIteration")
    axsAcc.set_xlabel("iteration")
    axsAcc.set_ylabel("accuracy")
    axsAcc.plot(data["testAccPerIterationList"])
    figAcc.savefig(dataPath + "testAccuracyPerIteration")
    pickle.dump(figAcc, open(dataPath + "testAccuracyPerIteration", 'wb'))
#### FUNTIONIERT

#### CEGA ??

    import os
    import cega_utils
    import re 

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

    datasetType = "categorical"
    featureNames = []
    #for i in range(len(data["testGradientsPerSamplePerFeature"][-1])): #vocab_size
    gradsTemp = data["testGradientsPerSamplePerFeature"]
    for i in range(len(gradsTemp[1][-1])):

        featureNames.append(str(i))

    featureNames = cega_utils.calculateAndSaveOHE_Rules(X_test, featureNames,trainedModelPrediction_Test_overIterations[-1], data["testGradientsPerSamplePerFeature_iteration"], datasetType,debug= False, vocab=vocab) #OHEresults


    import warnings
    warnings.filterwarnings('ignore')
    #     frequent_itemsets = apriori(basket_sets.astype('bool'), min_support=0.07, use_colnames=True) https://stackoverflow.com/questions/74114745/how-to-fix-deprecationwarning-dataframes-with-non-bool-types-result-in-worse-c
    debug = False

    import os 
    from datetime import datetime

    pos_label = 0
    neg_label = 1#'0'



    #rulesResultDataPath = dirPath + "rulesResultData/" 
    # eigentlich ohne dict
    featureDict = {feature: index  for index, feature in enumerate(featureNames)} #{feature: int(feature)  for feature in featureNames}#{feature: index  for index, feature in enumerate(featureNames)}
    #featureDict[pos_label] = int(pos_label)
    #featureDict[neg_label] = int(neg_label)

    #print(featureDict)
    #print(featureDict)
    #print.sasa
    now = datetime.now()
    date_time_string = now.strftime("%Y-%m-%d %H:%M:%S")
    # Replace space with underscore
    date_time_string = date_time_string.replace(" ", "_")

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

    tempRules_list = None
    
    from tqdm import tqdm
    for i in tqdm(range(len(os.listdir("./OHEresults/")))):
        ohe_df = cega_utils.loadOHE_Rules(i)

        all_rules, pos_rules , neg_rules =  cega_utils.runApriori(ohe_df,len(X_test), pos_label ,neg_label)
        discriminative_rules = cega_utils.getDiscriminativeRules(all_rules, pos_label, neg_label )
        characteristic_rules = cega_utils.getCharasteristicRules(pos_rules, pos_label, neg_rules,neg_label )

        resultName = "discriminative_rules"
        #resultName = "characteristic_rules"
        #rules_list, labelList_rules, rulePrecisionList, predictionComparisonList, rulesComplexityList , globalCoverage,  ruleSupportList,   numberOfGeneratedRules,  =cega_utils.calculateRulesMetrics(discriminative_rules, resultName ,featureDict, testloader, trainedModelPrediction_Test, rulesResultDataPath)
        #try:    
        rules_list, labelList_rules, rulePrecisionList, predictionComparisonList, rulesComplexityList , globalCoverage,  ruleSupportList,   numberOfGeneratedRules, raw_rules, rulePrecisionListPerRule  =cega_utils.calculateRulesMetrics(discriminative_rules, featureDict, test_set, trainedModelPrediction_Test_overIterations[i])
        #resultName = "characteristic_rules"
        #rules_list, labelList_rules, rulePrecisionList, predictionComparisonList, rulesComplexityList , globalCoverage,  ruleSupportList,  = numberOfGeneratedRules,  =cega_utils.calculateRulesMetrics(characteristic_rules, resultName ,featureDict, testloader, trainedModelPrediction_Test, rulesResultDataPath, debug=True )
        discriminative_rules_overIterations.append(discriminative_rules)
        characteristic_rules_overIterations.append(characteristic_rules) 
        #
        #print(rules_list)
        #except:
        #       print("rules df empty")
        #       rules_list = []
        #       labelList_rules = []
        #       rulePrecisionList =[]
        #       predictionComparisonList = []
        #       rulesComplexityList = []
        #       globalCoverage = 0
        #       ruleSupportList = []
        #       numberOfGeneratedRules = 0

        rules_list_overIterations.append(rules_list)
        labelList_rules_overIterations.append(labelList_rules)
        rulePrecisionList_overIterations.append(rulePrecisionList)
        predictionComparisonList_overIterations.append(predictionComparisonList)
        rulesComplexityList_overIterations.append(rulesComplexityList)
        globalCoverageList_overIterations.append(globalCoverage)
        ruleSupportList_overIterations.append(ruleSupportList)
        numberOfGeneratedRules_overIterations.append(numberOfGeneratedRules)
        raw_rules_overIterations.append(raw_rules)
        numberOfGeneratedRulesRAW_overIterations.append(len(raw_rules))
        rulePrecisionListPerRule_overIterations.append(rulePrecisionListPerRule)

        if tempRules_list is not None:
            print("not Jaccard")
            jaccardSimilarity_overIterations.append(cega_utils.jaccard_similarity(rules_list , tempRules_list))
            cosineSimilarity_overIterations.append(cega_utils.cosine_similarity(rules_list , tempRules_list))
            diceSimilarity_overIterations.append(cega_utils.dice_similarity(rules_list , tempRules_list))
            overlapSimilarity_overIterations.append(cega_utils.overlap_coefficient(rules_list , tempRules_list))
            #jaccardSimilarity_overIterations.append(cega_utils.cosine_similarity(rules_list , tempRules_list))

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


    rules_data = np.load(pathToNPZ , allow_pickle=True)

    import statistics
    def calculate_mean_of_lists(list_of_lists):
        means = []
        for sublist in list_of_lists:
            if len(sublist) == 0:
                means.append(-1)
            else:
                sublist_mean = np.mean(sublist)
                means.append(sublist_mean)
        return means


    pathToDiscriminative_rules = "./NLP_Results/rulesResults/discriminative_rules/"
    pathToCharacteristic_rules = "./NLP_Results/rulesResults/characteristic_rules"
    resultPaths_dicriminative_rules = os.listdir(pathToDiscriminative_rules)
    resultPaths_characteristic_rules = os.listdir(pathToCharacteristic_rules)
    resultPaths_dicriminative_rules= np.sort(resultPaths_dicriminative_rules)

    #get last generated rule
    mostRecentResultPaths_discriminative = pathToDiscriminative_rules + (resultPaths_dicriminative_rules[-1])

    data = utils.loadData(mostRecentResultPaths_discriminative)

    trackedRules_OHE = cega_utils.trackRulesList(data["rules_list_overIterations"])
    
    utils.appendToNPZ(pathToNPZ, "trackedRules_OHE", trackedRules_OHE)


    pathToRulesResults = "./NLP_Results/rulesResults/"

    fig1, axs1 = plt.subplots(nrows=1, ncols=1)
    axs1.plot(calculate_mean_of_lists(data["rulePrecisionList_overIterations"]))
    axs1.set_title("rulePrecisionList_overIterations")
    axs1.set_xlabel("iteration")
    axs1.set_ylabel("precision")
    fig1.savefig(str(pathToRulesResults) + "rulePrecisionList_overIterations")    
    pickle.dump(fig1, open(pathToRulesResults + "rulePrecisionList_overIterations", 'wb'))

    fig2, axs2 = plt.subplots(nrows=1, ncols=1)
    axs2.plot(calculate_mean_of_lists(data["ruleSupportList_overIterations"]))
    axs2.set_title("ruleSupportList_overIterations")
    axs2.set_xlabel("iteration")
    axs2.set_ylabel("support")
    fig2.savefig(str(pathToRulesResults) + "ruleSupportList_overIterations")    
    pickle.dump(fig2, open(pathToRulesResults + "ruleSupportList_overIterations", 'wb'))

    fig3, axs3 = plt.subplots(nrows=1, ncols=1)
    axs3.plot(calculate_mean_of_lists(data["rulesComplexityList_overIterations"]))
    axs3.set_title("rulesComplexityList_overIterations")
    axs3.set_xlabel("iteration")
    axs3.set_ylabel("complexity")
    fig3.savefig(str(pathToRulesResults) + "rulesComplexityList_overIterations")    
    pickle.dump(fig3, open(pathToRulesResults + "rulesComplexityList_overIterations", 'wb'))


    fig4, axs4 = plt.subplots(nrows=1, ncols=1)
    axs4.plot(data["globalCoverageList_overIterations"])
    axs4.set_title("globalCoverageList_overIterations")
    axs4.set_xlabel("iteration")
    axs4.set_ylabel("coverage")
    fig4.savefig(str(pathToRulesResults) + "globalCoverageList_overIterations")    
    pickle.dump(fig4, open(pathToRulesResults + "globalCoverageList_overIterations", 'wb'))

    fig5, axs5 = plt.subplots(nrows=1, ncols=1)
    axs5.plot(data["numberOfGeneratedRules_overIterations"])
    axs5.set_title("numberOfGeneratedRules_overIterations")
    axs5.set_xlabel("iteration")
    axs5.set_ylabel("numGeneratedRules")
    fig5.savefig(str(pathToRulesResults) + "numberOfGeneratedRules_overIterations")    
    pickle.dump(fig5, open(pathToRulesResults + "numberOfGeneratedRules_overIterations", 'wb'))

    fig6, axs6 = plt.subplots(nrows=1, ncols=1)
    axs6.plot(data["jaccardSimilarity_overIterations"])
    fig6.savefig(str(pathToRulesResults) + "jaccardSimilarity_overIterations") 
    axs6.set_title("jaccardSimilarity_overIterations")
    axs6.set_xlabel("iteration")
    axs6.set_ylabel("similarity")
    pickle.dump(fig6, open(pathToRulesResults + "jaccardSimilarity_overIterations", 'wb'))

    fig7, axs7 = plt.subplots(nrows=1, ncols=1)
    axs7.plot(data["cosineSimilarity_overIterations"])
    axs7.set_title("cosineSimilarity_overIterations")
    axs7.set_xlabel("iteration")
    axs7.set_ylabel("similarity")
    fig7.savefig(str(pathToRulesResults) + "cosineSimilarity_overIterations")    
    pickle.dump(fig7, open(pathToRulesResults + "cosineSimilarity_overIterations", 'wb'))

    fig8, axs8 = plt.subplots(nrows=1, ncols=1)
    axs8.plot(data["diceSimilarity_overIterations"])
    axs8.set_title("diceSimilarity_overIterations")
    axs8.set_xlabel("iteration")
    axs8.set_ylabel("similarity")
    fig8.savefig(str(pathToRulesResults) + "diceSimilarity_overIterations")    
    pickle.dump(fig8, open(pathToRulesResults + "diceSimilarity_overIterations", 'wb'))


    fig9, axs9 = plt.subplots(nrows=1, ncols=1)
    axs9.plot(data["overlapSimilarity_overIterations"])
    axs9.set_title("overlapSimilarity_overIterations")
    axs9.set_xlabel("iteration")
    axs9.set_ylabel("similarity")
    fig9.savefig(str(pathToRulesResults) + "overlapSimilarity_overIterations")    
    pickle.dump(fig9, open(pathToRulesResults + "overlapSimilarity_overIterations", 'wb'))

    fig10, axs10 = plt.subplots(nrows=1, ncols=1)
    axs10.plot(data["numberOfGeneratedRulesRAW_overIterations"])
    axs10.set_title("numberOfGeneratedRulesRAW_overIterations")
    axs10.set_xlabel("iteration")
    axs10.set_ylabel("numberOfGeneratedRulesRAW_overIterations")
    fig10.savefig(str(pathToRulesResults) + "numberOfGeneratedRulesRAW_overIterations")    
    pickle.dump(fig10, open(pathToRulesResults + "numberOfGeneratedRulesRAW_overIterations", 'wb'))
    
    fig1, axs1 = plt.subplots(nrows=1, ncols=1)
    axs1.plot(calculate_mean_of_lists(data["rulePrecisionListPerRule_overIterations"])) 
    axs1.set_title("rulePrecisionListPerRule_overIterations")
    axs1.set_xlabel("iteration")
    axs1.set_ylabel("rulePrecisionListPerRule_overIterations")
    fig1.savefig(str(pathToRulesResults) + "rulePrecisionListPerRule_overIterations")    
    pickle.dump(fig1, open(pathToRulesResults + "rulePrecisionListPerRule_overIterations", 'wb'))

    #print(data["numberOfGeneratedRulesRAW_overIterations"])
    _t_end = time()
    print(f"Training finished in {int(_t_end - _t_start)} s")


    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('#batches')
    ax1.set_ylim(0, 1.)
    ax1.set_ylabel('test accuracy', color=color)
    ax1.plot(test_accuracies,  color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:orange'
    ax2.set_ylabel('losses', color=color)  # we already handled the x-label with ax1
    ax2.set_ylim(min(0, min(loss)), max(loss))
    ax2.plot(loss, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
