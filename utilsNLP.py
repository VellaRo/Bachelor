from torch.utils.data.sampler import Sampler
"""
def getVocabSize():

    return None 

trainset = list(zip(X_train , y_train))      
    
random_indices_test =  data_list = random.sample(range(len(X_test)), len(X_test))
#print(random_indices_test)
random_indices_train =  data_list = random.sample(range(len(X_train)), len(X_train))

""" 
class OrderedListSampler(Sampler):
    def __init__(self, data_list):
        self.data_list = data_list

    def __iter__(self):
        return iter(self.data_list)

    def __len__(self):
        return len(self.data_list)
"""    
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
"""
