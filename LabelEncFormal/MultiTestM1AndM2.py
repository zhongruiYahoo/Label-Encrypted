import sys
import tqdm
from sklearn.preprocessing import OneHotEncoder
from network import Neural_Net, torch_train
from utils import *

"""
    more data more accuracy
"""

# hyper-parameter
test_rate = 0.2

# generate D1 for M1
def d1_generator(train, micro_rate):
    # get D1 max cls
    ar, num = np.unique(train[:, -1], return_counts=True)
    rate = 1 if len(num) != 2 else 0.85
    d1_idx_max = np.random.choice(np.where(train[:, -1] == ar[np.argmax(num)])[0], int(micro_rate * max(num)*rate),
                              replace=False).tolist()
    d1_data_max = train[d1_idx_max]

    # remaining data
    remain_idx = list(set(range(len(train))) - set(d1_idx_max))
    train = train[remain_idx]

    # get D1 others with accounting for 1:9
    d1_size_total = np.ceil(len(d1_idx_max) / 0.9)

    max_cls = ar[np.argmax(num)]
    ar = list(ar)
    ar.remove(max_cls)
    sample_num = int(np.floor((d1_size_total-len(d1_idx_max))/len(ar)))

    if sample_num != 0:
        other_sample = []
        for i in ar:
            sample = np.random.choice(np.where(train[:, -1] == i)[0], sample_num, replace=False).tolist()
            other_sample += sample
    else:
        other_sample = []
    should_num_d1 = int(np.ceil(len(d1_idx_max) / 9) - len(other_sample))

    # sample twice
    if should_num_d1 != 0 and micro_rate != 0.1:
        for j in range(should_num_d1):
            while True:
                cls = np.random.choice(ar, 1)[0]
                sample = np.random.choice(np.where(train[:, -1] == cls)[0], 1, replace=False).tolist()
                if sample not in other_sample:
                    break
            other_sample += sample

    d1_others = train[other_sample]

    # combine and get integration of D1
    d1 = np.vstack((d1_data_max, d1_others))
    return d1


def uniform(data):
    ar, num = np.unique(data[:, -1], return_counts=True)
    min_sample = min(num)
    sample_idx = []
    for i in ar:
        idx = np.random.choice(np.where(data[:, -1] == i)[0], min_sample, replace=False).tolist()
        sample_idx += idx
    return data[sample_idx]

# generate D2 union D1 for M2
def m2_generator(train, micro_rate, d1):
    m2_size = int(len(d1) * (1+micro_rate) / micro_rate)
    m2_idx = np.random.choice(range(len(train)), size=m2_size, replace=False)
    return train[m2_idx]

# split data for M1 and M2
def get_data_split(name, micro_rate):
    trait, label, batch1, batch2, lr, wd, neuron = data_split(name)

    # normalization
    trait = (trait - trait.mean(axis=0)) / trait.std(axis=0)
    file_copy = np.hstack((trait, label))
    file_copy = uniform(file_copy)

    input_dim = trait.shape[1]
    output_dim = len(set(label[:, -1]))

    # category
    ar, num = np.unique(file_copy[:, -1], return_counts=True)
    distribution = num / file_copy.shape[0]

    # create test data and train data
    remain_idx = set(range(len(file_copy)))
    test_idx = []
    for i in range(len(ar)):
        cls_idx = np.random.choice(np.where(file_copy[:, -1] == ar[i])[0],
                                   int(test_rate*file_copy.shape[0]*distribution[i]), replace=False).tolist()
        remain_idx -= set(cls_idx)
        test_idx += cls_idx
    remain_idx = list(remain_idx)
    train = file_copy[remain_idx]
    test = file_copy[test_idx]

    # get D1
    d1 = d1_generator(train, micro_rate)

    # get D2
    m2 = m2_generator(train, micro_rate, d1)

    return m2, test, d1, input_dim, output_dim, file_copy, batch1, batch2, lr, wd, neuron


def run(dataset):
    """
    This part shows the variation in model accuracy under the different ratio of dataset size for all the dataset.

    :param dataset: dataset name
    :return:
    """
    micro = []
    large = []

    for mic in np.arange(0.1, 1.1, 0.1):
        print('*'*60)
        print('The ratio of D1 and D2:', mic)
        res_micro = []
        res_large = []
        for i in tqdm.tqdm(range(100)):
            # load data and parameters
            train, test, micro_train, input_dim, output_dim, original, batch1, batch2, lr, wd, neuron = get_data_split(dataset, micro_rate=mic)

            # encode target to onehot
            enc = OneHotEncoder()
            enc.fit(original[:, -1].reshape(-1, 1))

            # split data
            train_feature, train_label = train[:, :-1], enc.transform(train[:, -1].reshape(-1, 1)).toarray()
            micro_feature, micro_label = micro_train[:, :-1], enc.transform(micro_train[:, -1].reshape(-1, 1)).toarray()
            test_feature, test_label = test[:, :-1], enc.transform(test[:, -1].reshape(-1, 1)).toarray()
            train_feature, micro_feature, test_feature = train_feature.astype(float), micro_feature.astype(float), test_feature.astype(float)

            # hidden layer
            layers = [input_dim, neuron, output_dim]

            # M1 training
            nn1 = Neural_Net(layers)
            sequence = list(range(micro_feature.shape[0]))
            np.random.shuffle(sequence)
            net1 = torch_train(feature=micro_feature,
                               label=micro_label,
                               w=nn1.InitWeight,
                               hidden=layers,
                               lr=lr,
                               weight_decay=wd,
                               epoch=100,
                               batch_size=batch1,
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

            # M2 training
            sequence = list(range(train_label.shape[0]))
            np.random.shuffle(sequence)
            net2 = torch_train(feature=train_feature,
                               label=train_label,
                               w=nn1.InitWeight,
                               hidden=layers,
                               lr=lr,
                               weight_decay=wd,
                               epoch=70,
                               batch_size=batch2,
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
        micro.append(np.mean(res_micro))
        large.append(np.mean(res_large))
        print('Accuracy of M1 is:', np.mean(res_micro))
        print('Accuracy of M2 is:', np.mean(res_large))
    micro = np.array(micro).reshape(1, -1)
    large = np.array(large).reshape(1, -1)
    total = np.vstack((micro, large))
    np.save(f'./multiTest/{dataset}', total)


if __name__ == "__main__":
    run(sys.argv[1])