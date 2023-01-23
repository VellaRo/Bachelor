import torch
import torch.nn as nn
# train + eval ,method 

#import eval
def train_eval(trainloader, testloader, model, num_epochs, device, y_train,y_test, lr):
    """returns 
    training_loss_batch =[] 
    grads = [] 
    training_loss_epoch = []
    test_loss_epoch = []
    test_loss_batch = []
    training_acc = []
    test_acc = []
    loss = []
    """
    # Backward Propergation - loss and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr) # 0.001 works # 0.0001 works good slows down # 

    #results
    grads = [] 
    training_loss_batch =[] 
    training_loss_epoch = []
    training_acc = []

    test_loss_batch = [] 
    test_loss_epoch = [] 
    test_acc = []
    #temp var
    loss = []

    epoch_counter = 0

    for epoch in range(num_epochs):

        running_corrects = 0

        for inputs, labels in trainloader:

            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs.requires_grad = True
            #inputs.retain_grad= True
            model.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            #outputs = model.forward(inputs) # ist das nicht das gleiche wie model(inputs)

            loss = loss_function(outputs, labels)
            training_loss_batch.append(loss.cpu())
       
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)
            
            # backward
            loss.backward(retain_graph=True) # the grads are from now on no longer none #   same ? grad = torch.autograd.grad(torch.unbind(_outputs), inputs)[0]

            """
            -> torch.unbind erwartet einen Vektor. Wenn du y den shap=(batch_size,
            n_classes) hat und n_classes>1, dann musst du noch die genaue Variable
            festlegen nach der abgeleitet werden soll, was leider etwas umstaendlich
            ist:
            """
            _outputs_max_idx = torch.argmax(outputs, dim=1) # indexthat contains maximal value per row (prediction per sample in batch)
            _outputs = torch.gather(outputs, dim=1, index= _outputs_max_idx.unsqueeze(1)) # gather sammelt outputs aus y entlang der Axe 
                                                                                         # dim die in index spezifiziert sind, 
                                                                                         # wobei index einen tensor von shape(batch_size, 1)
                                                                                         # erwartet (->unsqueeze(1))
            grad = torch.autograd.grad(torch.unbind(_outputs), inputs)[0]
            grads.append(grad)

            optimizer.step() 
            

    ##eval

        test_running_corrects = 0
        for inputs,labels in testloader:
        
            inputs = inputs.to(device)
            labels = labels.to(device)  
    
            with torch.no_grad():
    
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                test_running_corrects += torch.sum(preds == labels.data)
                test_loss = loss_function(outputs, labels)
                test_loss_batch.append(test_loss.cpu())
 
    
    ##evalEnd
        training_acc.append(running_corrects.item() /len(y_train))
        test_acc.append(test_running_corrects.item() /len(y_test)) 
        training_loss_epoch.append(loss.cpu())
        test_loss_epoch.append(test_loss.cpu())

        print("Epoch: " + str(epoch_counter) +" " + str(running_corrects.item() /len(y_train)))
        print("Epoch: Testing_acc" + str(epoch_counter) +" " + str(test_running_corrects.item() /len(y_test)))

        epoch_counter +=1 
    return grads, training_loss_batch, training_loss_epoch, training_acc, test_loss_batch, test_loss_epoch, test_acc

