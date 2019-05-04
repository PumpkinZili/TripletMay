import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.preprocessing import MultiLabelBinarizer

def my_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def get_confuse_matrix(output, target):
    chain = itertools.chain(*target)
    target = list(chain)

    chain = itertools.chain(*output)
    output = list(chain)

    with torch.no_grad():
        cm = confusion_matrix(target, output)

    np.save('confuse_matrix', cm)
    return cm

