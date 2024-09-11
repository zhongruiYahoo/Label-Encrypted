import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from network import Neural_Net, torch_train
import torch

"""
    more data more accuracy
"""

print('Dataset: Abrupto')

# load data and stan
file = pd.read_csv('data/abrupto/mixed_1010_abrupto_1.csv').to_numpy()
mean, std = np.mean(file[:, :-1], axis=0), np.std(file[:, :-1], axis=0)
file_copy = (file[:, :-1] - mean) / std
file_copy = np.hstack((file_copy, file[:, -1].reshape(-1, 1)))

# create test data and train data
cls1_idx = np.random.choice(np.where(file_copy[:, -1] == 0.0)[0], 200, replace=False).tolist()
cls2_idx = np.random.choice(np.where(file_copy[:, -1] == 1.0)[0], 200, replace=False).tolist()
remain_idx = list(set(range(len(file_copy))) - set(cls2_idx + cls1_idx))
train = file_copy[remain_idx]
test = file_copy[cls1_idx + cls2_idx]

# create micro data
cls1_idx = np.random.choice(np.where(train[:, -1] == 0.0)[0], 96, replace=False).tolist()
cls2_idx = np.random.choice(np.where(train[:, -1] == 1.0)[0], 864, replace=False).tolist()
micro_train = train[cls1_idx + cls2_idx]

# encode target to onehot
enc = OneHotEncoder()
enc.fit(train[:, -1].reshape(-1, 1))
train_feature, train_label = train[:, :-1], enc.transform(train[:, -1].reshape(-1, 1)).toarray()
micro_feature, micro_label = micro_train[:, :-1], enc.transform(micro_train[:, -1].reshape(-1, 1)).toarray()
test_feature, test_label = test[:, :-1], enc.transform(test[:, -1].reshape(-1, 1)).toarray()

# initiate architecture of nn
layers = [4]
layers.insert(0, 4)
layers.append(2)

res_micro = []
res_large = []
for i in range(10):
    np.random.seed(50+i)

    # same initialization of weights
    nn1 = Neural_Net(layers)
    sequence = list(range(micro_feature.shape[0]))
    np.random.shuffle(sequence)

    # train for M1
    net1 = torch_train(feature=micro_feature,
                       label=micro_label,
                       w=nn1.InitWeight,
                       hidden=layers,
                       lr=0.2,
                       weight_decay=0.01,
                       epoch=100,
                       batch_size=128,
                       indices=sequence)
    net1.eval()
    feature = torch.FloatTensor(test_feature)
    with torch.no_grad():
        out = net1(feature)
        pred_exp = np.exp(out.data.numpy().squeeze())
        pred = np.argmax(pred_exp, axis=1)
        target = np.argmax(test_label, axis=1)
        accuracy = sum(pred == target) / len(test_feature)
    res_micro.append(accuracy)

    # train for M2
    net2 = torch_train(feature=train_feature,
                       label=train_label,
                       w=nn1.InitWeight,
                       hidden=layers,
                       lr=0.1,
                       weight_decay=0.01,
                       epoch=100,
                       batch_size=512,
                       indices=sequence)
    net2.eval()
    feature = torch.FloatTensor(test_feature)
    with torch.no_grad():
        out = net2(feature)
        pred_exp = np.exp(out.data.numpy().squeeze())
        pred = np.argmax(pred_exp, axis=1)
        target = np.argmax(test_label, axis=1)
        acc = sum(pred == target) / len(test_feature)
    res_large.append(acc)

print("Accuracy M1:", res_micro)
print("Accuracy M2:", res_large)

with open('./others/table_3_single_test.txt', 'w') as f:
    f.write(repr(res_micro))
    f.write('\n')
    f.write(repr(res_large))
