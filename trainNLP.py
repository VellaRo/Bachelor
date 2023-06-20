from time import time

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

from datasetsNLP import get_agnews
from modelsNLP import SentenceCNN, BiLSTMClassif
import evalModel

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
    print(device)
    model.to(device)
    acc_val = []
    losses = []
    n_batches = 0
    _epochs, i_max = 0, 0
    accs = []
    while n_batches <= n_batches_max:
        for i, (text, labels) in enumerate(tr_data, 0):
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
            optim.step()
            losses.append(loss.item())

            n_batches += 1
            if n_batches > n_batches_max:
                break
        i_max = i

    acc_val.append(validate(inference_fn, model, *te_data))
    print("accuracies over test set")
    print(acc_val)
    return model, losses, accs


if __name__ == '__main__':

    size_train_batch = 64
    size_test_batch = 1024
    n_batches = 100
    embedding_dim = 128
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_set, test_set, size_vocab, n_classes = get_agnews(random_state=42, batch_sizes=(size_train_batch, size_test_batch))
    
    #X_test, Y_test = next(iter(test_set))  # only use first batch as a test set
    #X_test, Y_test = next(test_set)
    print(test_set)
    for X,y in test_set:
        X_test, Y_test =X,y
        break
    print(len(Y_test))
    
    Y_test_distr = torch.bincount(Y_test, minlength=n_classes)/size_test_batch
    print(f"class distribution in test set: {Y_test_distr}")  # this should roughly be uniformly distributed

    model = BiLSTMClassif(n_classes=n_classes, embed_dim=embedding_dim, vocab_size=size_vocab, hid_size=64)
    #model = SentenceCNN(n_classes=n_classes, embed_dim=embedding_dim, vocab_size=size_vocab)
    optimizer = Adam(model.parameters())
    loss_fun = torch.nn.CrossEntropyLoss()

    _t_start = time()

    model, loss, test_accuracies = \
        train_loop(model, optimizer, loss_fun, train_set, (X_test, Y_test),
                   inference_fn=model.forward_softmax, device=device, n_batches_max=n_batches)
    
    dirPath ="./"
    modelsDirPath = dirPath + "NLP_Models"

    loaderList = [test_set] # testLoader
    nameList = ["test"]
    yList = [Y_test]
    inputFeatures = []  
    num_epochs = n_batches # just for tracking progress


    evalModel.doALLeval(model, modelsDirPath,dirPath, loaderList, device,optimizer, loss_fun, num_epochs, nameList, yList, inputFeatures, random_indices_test)

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
