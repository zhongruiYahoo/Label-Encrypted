import random
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import load_iris
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
import torch
import copy

np.set_printoptions(suppress=True)

# neural network from scratch
class Neural_Net:
    def __init__(self, layers: list):
        self.W = []
        self.Layers = []
        self.Bais = []
        self.GredientW = []
        self.GredientBais = []
        self.LayerOutput = []
        self.InitWeight = []

        for l in range(len(layers) - 1):
            if l + 1 != len(layers) - 1:
                self.Layers.append((layers[l], layers[l + 1], True, 'sigmoid'))
            else:
                self.Layers.append((layers[l], layers[l + 1], False, 'logSoftmax'))

        for l in range(len(layers) - 1):
            w = self.xavier_uniform(layers[l], layers[l + 1])
            self.W.append(w)
        self.InitWeight = copy.deepcopy(self.W)
        for l in self.Layers:
            if not isinstance(l, str):
                self.Bais.append(np.zeros((1, l[1]))) if l[2] else self.Bais.append('None')

    def xavier_uniform(self, layer_in, layer_out):
        limit = np.sqrt(6.0 / (layer_in + layer_out))
        return np.random.uniform(-limit, limit, size=[layer_in, layer_out])

    def sigmoid(self, x, derivation=False):
        if derivation:
            return np.exp(-x) / (1 + np.exp(-x)) ** 2
        else:
            return 1.0 / (1 + np.exp(-x))

    def logSoftmax(self, x):
        x = np.exp(x)
        sum_ = np.sum(x, axis=1).reshape((-1, 1))
        x = x / sum_
        return np.log(x), x

    def forward(self, x):
        x_der = None
        for c, l in enumerate(self.Layers):
            temp_put = []
            x = np.dot(x, self.W[c]) + self.Bais[c] if l[2] else np.dot(x, self.W[c])
            temp_put.append(x)
            match l[-1]:
                case "sigmoid":
                    x = self.sigmoid(x)
                    temp_put.append(x)
                case "logSoftmax":
                    x, x_der = self.logSoftmax(x)
                    temp_put.append(x)
                case _:
                    assert False
            self.LayerOutput.append(temp_put)
        return x, x_der

    def cross_entropy(self, target, pred, der):
        loss = -np.sum(target * pred)
        derivation = np.mean(der - target, axis=0)
        return loss, derivation

    def backward(self, feature, loss_derivation):
        out_derivation = loss_derivation
        temp_gradient = []
        for l_num in range(len(self.Layers) - 1, -1, -1):
            l = self.Layers[l_num]
            temp = []
            if l_num == len(self.Layers) - 1:
                input = self.LayerOutput[l_num - 1][1][0]
                gradient = np.zeros((l[0], l[1]))
                for c, i in enumerate(input):
                    gradient[c, :] = i
                gradient *= out_derivation
                temp.append(gradient)
            else:
                before_activate = self.LayerOutput[l_num][0][0]
                match l[-1]:
                    case "sigmoid":
                        active_derivation = self.sigmoid(before_activate, derivation=True)
                    case _:
                        assert False
                out_derivation = active_derivation * out_derivation
                input = feature if l_num == 0 else self.LayerOutput[l_num - 1][1][0]
                gradient = np.zeros((l[0], l[1]))
                for c, i in enumerate(input):
                    gradient[c, :] = i
                gradient *= out_derivation
                temp.append(gradient)
                gradient_bias = out_derivation if l[2] else 'None'
                temp.append(gradient_bias)
            if l_num != 0:
                out_derivation = np.dot(self.W[l_num], out_derivation.reshape(-1, 1)).reshape(1, -1)
            temp_gradient.append(temp)
        return temp_gradient

    def gradients(self, loss_derivation, feature):
        gradient = self.backward(loss_derivation=loss_derivation, feature=feature)
        gradient.reverse()
        self.LayerOutput = []
        return gradient

    def step(self, gradient, lr, weight_decay=0):
        for c, layer in enumerate(self.Layers):
            self.W[c] = (1 - weight_decay * lr) * self.W[c] - lr * gradient[c][0]
            if layer[2]:
                self.Bais[c] = (1 - weight_decay * lr) * self.Bais[c] - lr * gradient[c][1]

    def train(self, feature, label, lr, weight_decay, epoch, batch_size, indices):
        for _ in range(epoch):
            for ff, ll in data_iter(feature, label, batch_size, indices):
                Gradient = None
                for f, l in zip(ff, ll):
                    out, out_der = self.forward(f)
                    loss, loss_derivation = self.cross_entropy(l, out, out_der)
                    gradient = self.gradients(loss_derivation=loss_derivation, feature=f)
                    if Gradient is None:
                        Gradient = copy.deepcopy(gradient)
                    else:
                        Gradient_tmp = []
                        for i in range(len(gradient)):
                            g1 = Gradient[i]
                            g2 = gradient[i]
                            total = [g1[0] + g2[0]]
                            if len(g1) == 2:
                                total_bias = np.atleast_2d(g1[1] + g2[1])
                                total.append(total_bias)
                            Gradient_tmp.append(total)
                        Gradient = copy.deepcopy(Gradient_tmp)
                Gradient_tmp = []
                for g in Gradient:
                    avg_g = [g[0] / len(ff)]
                    if len(g) == 2:
                        avg_bias = g[1] / len(ff)
                        avg_g.append(avg_bias)
                    Gradient_tmp.append(avg_g)
                self.step(gradient=Gradient_tmp, lr=lr, weight_decay=weight_decay)

    def accuracy(self, feature, label):
        _, out = self.forward(feature)
        index = np.argmax(out, axis=1)
        label_index = np.argmax(label, axis=1)
        acc = sum(index == label_index) / len(label)
        return acc, out

# neural network from PyTorch
class NN(nn.Module):
    def __init__(self, init_weight, layers):
        super(NN, self).__init__()
        self.net = nn.Sequential()

        for i in range(len(layers) - 1):
            if i == len(layers) - 2:
                self.net.add_module(f'l{i}-l{i + 1}', nn.Linear(layers[i], layers[i + 1], bias=False))
            else:
                self.net.add_module(f'l{i}-l{i + 1}', nn.Linear(layers[i], layers[i + 1]))
                self.net.add_module(f'l{i + 1}Fun', nn.Sigmoid())
        num = 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = torch.FloatTensor(init_weight[num].T)
                if m.bias != None:
                    m.bias.data.zero_()
                num += 1

    def forward(self, x):
        x = self.net(x)
        out = F.log_softmax(x, dim=1)
        return out

    def call_loss(self, target, pred):
        loss = -torch.sum(pred * target) / len(pred)
        return loss

# training model of PyTorch
def torchTrain(feature, label, w, hidden, lr, weight_decay, epoch, batch_size, indices):
    net = NN(init_weight=w, layers=hidden)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=0)
    net.train()
    for _ in range(epoch):
        for f, l in data_iter(feature, label, batch_size, indices):
            f = torch.FloatTensor(np.atleast_2d(f))
            l = torch.LongTensor(np.atleast_2d(l))
            optimizer.zero_grad()
            out = net(f)
            loss = net.call_loss(target=l, pred=out)
            loss.backward()
            optimizer.step()
    return net


# data generation
def data_iter(feature, label, batch_size, sequence):
    indices = sequence
    for i in range(0, len(feature), batch_size):
        batch_indices = indices[i:min(i+batch_size, len(feature))]
        yield feature[batch_indices], label[batch_indices]

# ROC curvy
def roc_curvy(pred, label, n_class):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_class):
        fpr[i], tpr[i], _ = roc_curve(label[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_class):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, roc_auc

if __name__ == '__main__':

    # load data
    np.random.seed(0)
    print('Dateset: Iris')
    iris = load_iris()
    data = iris['data']
    target = np.atleast_2d(iris['target']).T
    input_dim = data.shape[1]
    output_dim = len(set(target[:, 0]))

    # encode label
    enc = OneHotEncoder()
    enc.fit(target)
    target = enc.transform(target).toarray()

    # data split
    train_feature, test_feature, train_label, test_label = train_test_split(data, target, train_size=0.7,
                                                                            random_state=0)

    # initiate architecture
    layers = [4, 4]
    layers.insert(0, input_dim)
    layers.append(output_dim)
    lr = 0.1
    weight_decay = 1e-2
    epoch = 100
    batch_size = 16
    sequence = list(range(len(train_feature)))
    random.shuffle(sequence)

    # training processing of nn from scratch
    nn_ = Neural_Net(layers)
    nn_.train(feature=train_feature,
              label=train_label,
              lr=lr,
              weight_decay=weight_decay,
              epoch=epoch,
              batch_size=batch_size,
              indices=sequence)
    numpy_acc, out = nn_.accuracy(test_feature, test_label)
    np_fpr, np_tpr, np_roc_auc = roc_curvy(pred=out, label=test_label, n_class=output_dim)

    # training processing of nn from PyTorch
    net = torchTrain(feature=train_feature,
                     label=train_label,
                     w=nn_.InitWeight,
                     hidden=layers,
                     lr=lr,
                     weight_decay=weight_decay,
                     epoch=epoch,
                     batch_size=batch_size,
                     indices=sequence)
    net.eval()
    feature = torch.FloatTensor(test_feature)
    with torch.no_grad():
        out = net(feature)
        pred_exp = np.exp(out.data.numpy().squeeze())
        pred = np.argmax(pred_exp, axis=1)
        target = np.argmax(test_label, axis=1)
        torch_accuracy = sum(pred == target) / len(test_feature)
    torch_fpr, torch_tpr, torch_roc_auc = roc_curvy(pred=pred_exp, label=test_label, n_class=output_dim)

    # save results
    print('*'*60)
    print('From scratch weights and bias')
    with open('./others/from_scratch_weights.txt', 'w') as f:
        for w in nn_.W:
            f.write(str(w.T))
            f.write('\n')
            print(np.around(w.T, 4))
        for b in nn_.Bais:
            if not isinstance(b, str):
                f.write(str(b.T))
                f.write('\n')
                print(np.around(b, 4))

    print('*' * 60)
    print('Torch weights and bias')
    with open('./others/torch_weights.txt', 'w') as f:

        for m in net.modules():
            if isinstance(m, nn.Linear):
                print(m.weight.data)
                f.write(str(m.weight.data))
                f.write('\n')
                if m.bias is not None:
                    f.write(str(m.bias.data))
                    f.write('\n')
                    print(m.bias.data)

    print('numpy accuracy', numpy_acc)
    print('torch accuracy', torch_accuracy)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    plt.figure()
    for i, color in zip(range(output_dim), colors):
        plt.plot(np_fpr[i], np_tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, np_roc_auc[i]))
    plt.plot(np_fpr["macro"], np_tpr["macro"],
             label='Average ROC curve (area = {0:0.2f})'
                   ''.format(np_roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.legend()
    plt.xlim([-0.005, 1.0])
    plt.ylim([0.00, 1.05])
    plt.xlabel('False Positive Rate')
    plt.savefig('./others/figure_6_np_roc.png')

    plt.figure()
    for i, color in zip(range(output_dim), colors):
        plt.plot(torch_fpr[i], torch_tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, torch_roc_auc[i]))
    plt.plot(torch_fpr["macro"], torch_tpr["macro"],
             label='Average ROC curve (area = {0:0.2f})'
                   ''.format(torch_roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.legend()
    plt.xlim([-0.005, 1.0])
    plt.ylim([0.00, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('./others/figure_6_torch_roc.png')
