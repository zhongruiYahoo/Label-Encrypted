import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from fhe import FHE
from network import Neural_Net, torch_train
import time
import os
import sys

# Repetition
round = 10

def run(name, total_epsilon):
    """
        Main experiment function for table 4 and table 8

    :param name: dataset
    :param total_epsilon: total epsilon, a scalar
    """

    # load data
    match name:
        case 'iris':
            file = pd.read_csv('./data/iris/iris.data', header=None).values
            trait = file[:, :-1].astype(float)
            label = file[:, -1].reshape(-1, 1)
        case 'seeds':
            file = pd.read_table('./data/seeds/seeds_dataset.txt', sep='\\s+', header=None).values
            trait = file[:, :-1].astype(float)
            label = file[:, -1].reshape(-1, 1)
        case 'wine':
            file = pd.read_csv('./data/wine/wine.data', header=None).values
            trait = file[:, 1:]
            label = file[:, 0].reshape(-1, 1)
        case 'abrupto':
            file = pd.read_csv('./data/abrupto/mixed_1010_abrupto_1.csv').values
            trait = file[:, :-1]
            label = file[:, -1].reshape(-1, 1)
        case 'drebin':
            file = np.load('./data/drebin/vec.npy')
            trait = file[:, :-1]
            label = file[:, -1].reshape(-1, 1)
        case _:
            file = trait = label = None
            assert print('not exist dataset')

    # initiate parameters of nn
    if name in ['iris', 'seeds', 'wine', 'abrupto', 'drebin']:
        nn_neuron = 20
        epoch = 50
        l2 = 0.01
        lr = 0.1
        batch = 256
    else:
        nn_neuron = 128
        epoch = 70
        l2 = 0.001
        lr = 0.01
        batch = 32

    # normalization mu std
    trait = (trait - trait.mean(axis=0)) / trait.std(axis=0)
    input_dim = trait.shape[1]
    output_dim = len(set(label[:, -1]))

    # encode label
    enc = OneHotEncoder()
    enc.fit(label)
    label = enc.transform(label).toarray()

    # initiate fhe parameters
    FHE.LABEL_SHAPE = (1, output_dim)
    FHE.OUTPUT_SHAPE = (nn_neuron, output_dim)
    FHE.NOISE_SHAPE = (nn_neuron, output_dim)
    FHE.N_SIGMA_SHAPE = (nn_neuron, output_dim)

    # initiate neural networks
    layers = [nn_neuron]
    layers.insert(0, input_dim)
    layers.append(output_dim)

    # load t list of sensitivity
    delta_list = pd.read_csv(f'./TListDPNoise/{name}/delta_list.csv', header=None, index_col=0).values[1:]
    dp_noise = np.load(f'./TListDPNoise/{name}/{total_epsilon}_dp_list.npy')
    epsilon = total_epsilon / np.sqrt(epoch)

    res_part = np.zeros((round, 8))
    for ii in tqdm(range(round)):

        # sample test/holdout from universal
        test_idx = np.random.choice(range(trait.shape[0]), int(trait.shape[0] * 0.3), replace=False)
        test_trait, test_label = trait[test_idx], label[test_idx]

        # remainder is D2 + D1 (train M2)
        remain_idx = list(set(range(trait.shape[0])) - set(test_idx))
        large_trait, large_label = trait[remain_idx], label[remain_idx]

        # sample D1 (train M1) from D2 + D1
        micro_idx = np.random.choice(range(large_trait.shape[0]), int(trait.shape[0] * 0.1), replace=False)
        micro_trait, micro_label = large_trait[micro_idx], large_label[micro_idx]

        # shuffle D2 + D1
        sequence = list(range(large_trait.shape[0]))
        np.random.shuffle(sequence)

        # train M2~ (nn3) from scratch with [enc + DP]
        nn3 = Neural_Net(layers, delta_list_t=delta_list, dp_noise=dp_noise)
        t_start = time.time()
        nn3.train(feature=large_trait,
                  label=large_label,
                  lr=lr,
                  weight_decay=l2,
                  epoch=epoch,
                  batch_size=batch,
                  indices=sequence,
                  encryption_state=True,
                  epsilon=epsilon)
        enc_time = time.time() - t_start
        enc_acc, _ = nn3.accuracy(test_trait, test_label)

        # shuffle D2 + D1
        sequence = list(range(large_trait.shape[0]))
        np.random.shuffle(sequence)

        # train M2 (nn1) from scratch without [enc + DP]
        nn1 = Neural_Net(layers)
        t_start = time.time()
        nn1.train(feature=large_trait,
                  label=large_label,
                  lr=lr,
                  weight_decay=l2,
                  epoch=epoch,
                  batch_size=batch,
                  indices=sequence,
                  encryption_state=False,
                  epsilon=epsilon)
        large_time = time.time() - t_start
        large_acc, _ = nn1.accuracy(test_trait, test_label)

        # train M2 (net, benchmark) using PyTorch without [enc + DP]
        t_start = time.time()
        net = torch_train(feature=large_trait,
                          label=large_label,
                          w=nn1.InitWeight,
                          hidden=layers,
                          lr=lr,
                          weight_decay=l2,
                          epoch=epoch,
                          batch_size=batch,
                          indices=sequence)
        net.eval()
        feature = torch.FloatTensor(test_trait)
        with torch.no_grad():
            out = net(feature)
            pred_exp = np.exp(out.data.numpy().squeeze())
            pred = np.argmax(pred_exp, axis=1)
            target = np.argmax(test_label, axis=1)
            torch_accuracy = sum(pred == target) / len(test_trait)
        torch_time = time.time() - t_start

        # shuffle D1
        sequence = list(range(micro_trait.shape[0]))
        np.random.shuffle(sequence)

        # train M1 (nn2) from scratch without [enc + DP]
        nn2 = Neural_Net(layers)
        t_start = time.time()
        nn2.train(feature=micro_trait,
                  label=micro_label,
                  lr=lr,
                  weight_decay=l2,
                  epoch=epoch,
                  batch_size=batch,
                  indices=sequence,
                  encryption_state=False,
                  epsilon=epsilon)
        micro_time = time.time() - t_start
        micro_acc, _ = nn2.accuracy(test_trait, test_label)

        # save results
        res_part[ii, :] = [micro_time, micro_acc, large_time, large_acc, enc_time, enc_acc, torch_time, torch_accuracy]

    # write results file
    res_mean = res_part.mean(axis=0)
    column = ['micro_time', 'micro_acc', 'total_time', 'total_acc', 'enc_time', 'enc_acc', 'torch_time', 'torch_acc']
    with open(f'./res/{name}/e_{total_epsilon}.txt', 'w', encoding='utf-8') as w:
        for c, col in enumerate(column):
            w.write(col + ',') if c != len(column)-1 else w.write(col+'\n')
        for d in res_part:
            for c, i in enumerate(d):
                w.write(str(np.around(i, 4))+',')
            w.write('\n')
        w.write('\n')
        for c, d in enumerate(res_mean):
            w.write(str(np.around(d, 4))+',')


if __name__ == "__main__":
    name = sys.argv[1]
    total_epsilon = float(sys.argv[2])
    folder_path = f"./res/{name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    run(name, total_epsilon)
