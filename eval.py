import torch
import os 
import re 
import numpy as np

import utils


from sklearn.metrics.pairwise import cosine_similarity


def calcConsineSimilarity(model,modelsDirPath):

    
    initialModel = model
    

    #initialModel.load_state_dict(torch.load(modelsDirPath + "/" +str(0)))
    initialModel.load_state_dict(torch.load(modelsDirPath +"/initialModel"))
    initialModel.eval()
    initalWeights = utils.getWeights(initialModel)

    finalModel = model

    #finalModel.load_state_dict(torch.load(modelsDirPath + "/" +str(len(modelsDirPath))))
    finalModel.load_state_dict(torch.load(modelsDirPath +"/finalModel"))
    finalModel.eval()
    finalWeights = utils.getWeights(finalModel)

    #print(initialModel == finalModel)
    model = model
    #print(initalWeights[255:265])
    #print(finalWeights[255:265])
    # remove initialModel and finalModel 
    r = re.compile("^[0-9]*$")
    modelsDirFiltered = list(filter(r.match, os.listdir(modelsDirPath)))
    #print(modelsDirPath)
    cosine_similarity_toInitial = []
    cosine_similarity_toFinal = []
    for filename in np.sort(list(eval(i) for i in modelsDirFiltered)): #(os.listdir(modelsDirPath)))): #
        #print(filename)
        #if filename == "initialModel" or filename == "finalModel":
        #    print("pass")
        #    pass

        model.load_state_dict(torch.load(modelsDirPath + "/" +str(filename)))
        model.eval()
        #get weights of model at iteration
        iterationWeights = utils.getWeights(model)

        
 
        #print(initalWeights)
        #for i in iterationWeights:
        #    print(i)
        cosine_similarity_toInitial.append(cosine_similarity([initalWeights],[iterationWeights]))
        cosine_similarity_toFinal.append(cosine_similarity([finalWeights], [iterationWeights]))
        #print(cosine_similarity([fixed_weights], [i]))
    
    return cosine_similarity_toInitial , cosine_similarity_toFinal