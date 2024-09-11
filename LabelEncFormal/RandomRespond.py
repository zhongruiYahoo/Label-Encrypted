import argparse
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from torch.nn import functional as F
import copy
from utils import *

parser = argparse.ArgumentParser(description="Dataset and Epsilon")
parser.add_argument('--dataset', type=str, help="dataset name")
parser.add_argument('--epsilon', type=str, help="Input list, separated by commas")
args = parser.parse_args()

# neural network by PyTorch
class NN(nn.Module):
    def __init__(self, layers):
        super(NN, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 1):
            if i == len(layers) - 2:
                self.net.add_module(f'l{i}-l{i + 1}', nn.Linear(layers[i], layers[i + 1], bias=False))
            else:
                self.net.add_module(f'l{i}-l{i + 1}', nn.Linear(layers[i], layers[i + 1]))
                self.net.add_module(f'l{i + 1}Fun', nn.Sigmoid())
        # print(self.net)
        num = 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data, gain=1)
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

# train with PyTorch
def torch_train(feature, label, hidden, lr, weight_decay, epoch, batch_size, indices):
    net = NN(layers=hidden)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
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


# data generator
def data_iter(feature, label, batch_size, sequence):
    indices = sequence
    for i in range(0, len(feature), batch_size):
        batch_indices = indices[i:min(i+batch_size, len(feature))]
        yield feature[batch_indices], label[batch_indices]


# Hyperparameters
rate = 0.3
round = 10


def run(name, epsilon):
    """
        Random Response for table 4 and table 8
    :param name: dataset name
    :param epsilon: epsilon value. The input can be either a list or a scalar.
    :return:
    """

    # initiate parameters for nn
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

    # load dataset
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
        case 'cifar10':
            trait, label = cifar_data('cifar10')
        case 'cifar100':
            trait, label = cifar_data('cifar100')
        case 'purchase10':
            f = pd.DataFrame(pd.read_csv('./data/purchases_full/purchases_full.csv', index_col=0))
            trait = f.iloc[:, 5:].values
            label = np.atleast_2d(f['10classes'].values).T
        case _:
            file = trait = label = None
            assert print('not exist dataset')

    # normalization
    trait = (trait - trait.mean(axis=0)) / trait.std(axis=0)
    input_dim = trait.shape[1]
    output_dim = len(set(label[:, -1]))

    # initiate architecture of nn
    layers = [nn_neuron]
    layers.insert(0, input_dim)
    layers.append(output_dim)
    enc = OneHotEncoder()
    enc.fit(label)

    # training
    for c, e in enumerate(epsilon):
        res_data = []
        for i in range(round):

            # data splitting
            test_idx = np.random.choice(range(trait.shape[0]), int(trait.shape[0] * rate), replace=False)
            train_idx = list(set(range(trait.shape[0])) - set(test_idx))
            label_onehot = enc.transform(label).toarray()
            test_feature, test_label = trait[test_idx], label_onehot[test_idx]
            train_feature, train_label = trait[train_idx], label_onehot[train_idx]

            # calculate p using epsilon. When epsilon > 1000, it would overflow, so just set it 1
            if e >= 500:
                p = 1
            else:
                p = np.e ** e / (np.e ** e+output_dim - 1)
            label_new = copy.deepcopy(label[train_idx])
            label_card = set(label[:, -1])
            print("Probability", p)

            # random response
            for cc, j in enumerate(label[train_idx]):
                if np.random.rand() > p:
                    remain_label = list(label_card - set(j))
                    label_new[cc] = np.random.choice(remain_label, 1)
            train_label_new = enc.transform(label_new).toarray()

            # statistic
            unchange_idx = []
            malicious_idx = []
            for h in range(len(train_label_new)):
                if np.array_equal(train_label_new[h, :], train_label[h, :]):
                    unchange_idx.append(h)
                else:
                    malicious_idx.append(h)
            print('Number of unchanged data：', len(unchange_idx), 'account for', len(unchange_idx)/len(train_label_new))
            print('Number of malicious data: ', len(malicious_idx), 'account for', len(malicious_idx)/len(train_label_new))
            sequence = list(range(len(train_label_new)))
            np.random.shuffle(sequence)

            # training
            net2 = torch_train(feature=train_feature,
                               label=train_label_new,
                               hidden=layers,
                               lr=lr,
                               weight_decay=l2,
                               epoch=epoch,
                               batch_size=batch,
                               indices=sequence)
            net2.eval()
            feature = torch.FloatTensor(test_feature)
            with torch.no_grad():
                out = net2(feature)
                pred_exp = np.exp(out.data.numpy().squeeze())
                pred = np.argmax(pred_exp, axis=1)
                target = np.argmax(test_label, axis=1)
                torch_accuracy_t = sum(pred == target) / len(test_feature)
            res_data.append(torch_accuracy_t)
        res = np.atleast_1d(np.mean(res_data))
        np.savetxt(f'./res/{name}/{e}_random.txt', res, fmt='%.4f')


if __name__ == '__main__':
    name = args.dataset
    epsilon = [float(i) for i in args.epsilon.split(',')]
    run(name, epsilon)
