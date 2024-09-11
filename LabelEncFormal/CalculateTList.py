import os
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from network import data_iter
import time
from fhe import *
from utils import *
import sys

decimal = 6


class Neural_Net:

    # initialize net and weight
    def __init__(self, layers: list,  *args, init_w=False, delta_list_t=0, dp_noise=0):
        self.W = []                                         # weights
        self.Layers = []                                    # net structure in detail
        self.Bias = []                                      # bias
        self.LayerOutput = []                               # layer outputs
        self.InitWeight = []                                # initial weight
        self.fhe = FHE()                                    # enc fhe model
        self.delta_list = []
        self.delta_list_t = delta_list_t
        self.dp_noise = dp_noise

        # initialize layers info
        for l in range(len(layers) - 1):
            if l + 1 != len(layers) - 1:
                self.Layers.append((layers[l], layers[l + 1], True, 'sigmoid'))
            else:
                self.Layers.append((layers[l], layers[l + 1], False, 'logSoftmax'))

        # initialize weights
        if init_w.__eq__(False):
            for l in range(len(layers) - 1):
                w = self.xavier_uniform(layers[l], layers[l + 1])
                self.W.append(w)
        else:
            self.W = args[0]

        # open it when we compare test acc against torch
        self.InitWeight = copy.deepcopy(self.W)

        # initialize bias
        for l in self.Layers:
            if not isinstance(l, str):
                self.Bias.append(np.zeros((1, l[1]))) if l[2] else self.Bias.append('None')

    # initialize weight func
    def xavier_uniform(self, layer_in, layer_out):
        limit = np.sqrt(6.0 / (layer_in + layer_out))
        return np.random.uniform(-limit, limit, size=[layer_in, layer_out])

    # sigmoid function
    def sigmoid(self, x, derivation=False):
        x = x.reshape(1, -1)
        sig = np.array([1.0 / (1.0 + np.exp(-i)) if i > 0 else np.exp(i)/(1 + np.exp(i)) for i in x[0]]).reshape(1, -1)
        if derivation:
            return sig * (1 - sig)
        else:
            return sig

    # logsoftmax function : return log(softmax(x)), softmax(x)
    def logSoftmax(self, x):
        x = np.exp(x)
        sum_ = np.sum(x, axis=1).reshape((-1, 1))
        x = x / (sum_ + 1e-4) + 1e-4
        return np.log(x), x

    # forward propagation
    def forward(self, x):
        x_der = None

        for c, l in enumerate(self.Layers):
            temp_put = []

            # correct result but np.dot is slow
            x = np.dot(x, self.W[c]) + self.Bias[c] if l[2] else np.dot(x, self.W[c])

            # faster than np.dot, same result
            temp_put.append(x)

            # calculate activation func
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

    # def cross_entropy(self, target, pred, der):
    #     loss = -np.sum(target * pred)
    #     derivation = np.mean(der - target, axis=0)
    #     return loss, derivation

    # update weight and bias
    def step(self, gradient, lr, weight_decay=0):
        for c, layer in enumerate(self.Layers):
            self.W[c] = (1 - weight_decay * lr) * self.W[c] - lr * gradient[c][0]
            if layer[2]:
                self.Bias[c] = (1 - weight_decay * lr) * self.Bias[c] - lr * gradient[c][1]

    # calculate gradients
    def gradients(self, out_der, feature, part2):
        out_derivation = None
        temp_gradient = []
        for l_num in range(len(self.Layers) - 1, -1, -1):
            l = self.Layers[l_num]
            # print(l_num, l)
            temp = []

            # for last layer
            if l_num == len(self.Layers) - 1:
                input = self.LayerOutput[l_num - 1][1][0]
                gradient = np.zeros((l[0], l[1]))
                for c, i in enumerate(input):
                    gradient[c, :] = i
                part1 = out_der * gradient
                gradient = part1 - part2
                out_derivation = out_der - part2[0, :] / (input[0] + 1e-6)
                out_derivation = out_derivation[0]
                temp.append(gradient)
            # for other layers
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
        temp_gradient.reverse()
        self.LayerOutput = []
        return temp_gradient

    # result of forward propagation
    def prepare(self, features, label):
        labels = []
        outputs = []
        for i in range(len(features)):
            f, l = features[i], label[i]
            layer = self.Layers[-1]
            self.forward(f)
            input = self.LayerOutput[-2][1][0]
            output = np.zeros((layer[0], layer[1]))
            for c, i in enumerate(input):
                output[c, :] = i
            labels.append(l)
            outputs.append(output)
            self.LayerOutput = []
        return labels, outputs

    # uniform noise
    def uniform_noise(self, batch, shape, decimal=6):
        noise_list = []
        for _ in range(batch):
            noise = np.random.uniform(0, 1, shape)
            noise = np.around(noise, decimals=decimal)
            noise_list.append(noise)
        return noise_list

    # enc process
    def en_de_cryption_step(self, labels, outputs, delta, encryption_state, decimal, epsilon):
        part2 = []
        if encryption_state.__eq__(True):
            noise = self.uniform_noise(batch=len(outputs),
                                       shape=outputs[0].shape,
                                       decimal=decimal)
            sigma = 1 / epsilon
            n_sigma = np.random.normal(loc=0, scale=sigma, size=outputs[0].shape)
            n_sigma = np.around(n_sigma, int(decimal / 2)) * 10 ** int(decimal / 2)
            n_sigma = n_sigma.astype(int)
            min_n_sigma = abs(np.min(n_sigma))
            n_sigma += min_n_sigma
            delta = int(np.around(delta, int(decimal / 2)) * 10 ** int(decimal / 2))
            for l, o, n in zip(labels, outputs, noise):
                o = np.around(o, decimals=decimal) * 10 ** decimal
                n = np.around(n, decimals=decimal) * 10 ** decimal
                o, n = o.astype(int), n.astype(int)
                l = l.reshape(1, -1).astype(int)
                n_sigma1 = n_sigma * delta
                decrypted = self.fhe.en_run_de_crypt(label=l,
                                                     output=o,
                                                     noise=n,
                                                     n_sigma=n_sigma1)
                decrypted1 = decrypted.astype(float) - delta*min_n_sigma
                decrypted = decrypted1.astype(float)
                decrypted -= n
                decrypted = decrypted / 10**decimal
                part2.append(decrypted)
        else:
            for l, o in zip(labels, outputs):
                part2.append(l * o)
        return part2

    # calculate sensitivity
    def calculate_sensitivity(self, feature):
        s = []
        for f in feature:
            self.forward(f)
            output = self.LayerOutput[-2][1][0]
            s.append(np.linalg.norm(output, ord=2))
            self.LayerOutput = []
        return max(s)

    # training
    def train(self, feature, label, lr, weight_decay, epoch, batch_size, indices, epsilon, encryption_state=False):
        delta = 0
        for i in tqdm(range(epoch)):
            # print(i)
            if encryption_state:
                delta = 2 * self.calculate_sensitivity(feature) / batch_size
                self.delta_list.append(delta)
            # open it when Model Plaintext vs Model Ciphertext
            # delta = 0

            for ff, ll in data_iter(feature, label, batch_size, indices):
                # gradients of batch sample
                Gradient = None
                labels, outputs = self.prepare(ff, ll)
                part2 = self.en_de_cryption_step(labels=labels,
                                                 outputs=outputs,
                                                 delta=delta,
                                                 encryption_state=encryption_state,
                                                 decimal=6,
                                                 epsilon=epsilon)
                for c, f_l in enumerate(zip(ff, ll)):
                    f, l = f_l[0], f_l[1]
                    out, out_der = self.forward(f)
                    # one sample gradient
                    gradient = self.gradients(out_der=out_der, feature=f, part2=part2[c])
                    if Gradient is None:
                        Gradient = copy.deepcopy(gradient)
                    else:
                        Gradient_tmp = []
                        for j in range(len(gradient)):
                            # g1 gradient of jth layer previous samples (list)
                            # g2 gradient of jth layer current sample (list)
                            g1 = Gradient[j]
                            g2 = gradient[j]
                            # merge two list
                            total = [g1[0] + g2[0]]
                            if len(g1) == 2:
                                total_bias = np.atleast_2d(g1[1] + g2[1])
                                total.append(total_bias)
                            Gradient_tmp.append(total)
                        Gradient = copy.deepcopy(Gradient_tmp)
                # average gradient
                Gradient_avg = []
                for g in Gradient:
                    avg_g = [g[0] / len(ff)]
                    if len(g) == 2:
                        avg_bias = g[1] / len(ff)
                        avg_g.append(avg_bias)
                    Gradient_avg.append(avg_g)

                # update weights and bias
                self.step(gradient=Gradient_avg, lr=lr, weight_decay=weight_decay)

    # training for the experiment of Ciphertext vs Plaintext, sensitivity(delta) = 0
    def train2(self, feature, label, lr, weight_decay, epoch, batch_size, indices, epsilon, encryption_state=False):
        delta = 0
        for i in tqdm(range(epoch)):
            delta = 0
            for ff, ll in data_iter(feature, label, batch_size, indices):
                # gradients of batch sample
                Gradient = None
                labels, outputs = self.prepare(ff, ll)
                part2 = self.en_de_cryption_step(labels=labels,
                                                 outputs=outputs,
                                                 delta=delta,
                                                 encryption_state=encryption_state,
                                                 decimal=6,
                                                 epsilon=epsilon)
                for c, f_l in enumerate(zip(ff, ll)):
                    f, l = f_l[0], f_l[1]
                    out, out_der = self.forward(f)
                    # one sample gradient
                    gradient = self.gradients(out_der=out_der, feature=f, part2=part2[c])
                    if Gradient is None:
                        Gradient = copy.deepcopy(gradient)
                    else:
                        Gradient_tmp = []
                        for j in range(len(gradient)):
                            # g1 gradient of jth layer previous samples (list)
                            # g2 gradient of jth layer current sample (list)
                            g1 = Gradient[j]
                            g2 = gradient[j]
                            # merge two list
                            total = [g1[0] + g2[0]]
                            if len(g1) == 2:
                                total_bias = np.atleast_2d(g1[1] + g2[1])
                                total.append(total_bias)
                            Gradient_tmp.append(total)
                        Gradient = copy.deepcopy(Gradient_tmp)
                # average gradient
                Gradient_avg = []
                for g in Gradient:
                    avg_g = [g[0] / len(ff)]
                    if len(g) == 2:
                        avg_bias = g[1] / len(ff)
                        avg_g.append(avg_bias)
                    Gradient_avg.append(avg_g)

                # update weights and bias
                self.step(gradient=Gradient_avg, lr=lr, weight_decay=weight_decay)

    def accuracy(self, feature, label):
        res = []
        for i in feature:
            _, out = self.forward(i)
            res.append(out)
        out = np.array(res).reshape(len(feature), -1)
        index = np.argmax(out, axis=1)
        label_index = np.argmax(label, axis=1)
        acc = sum(index == label_index) / len(label)
        return acc, out


@cnp.compiler({"n_sigma": "encrypted",
               "delta": "clear"})
def f1_t_list(n_sigma, delta):
    gradient = n_sigma * delta
    return gradient

# FHE model
class FHE_T_List:

    LABEL_SHAPE = None
    OUTPUT_SHAPE = None
    NOISE_SHAPE = None
    N_SIGMA_SHAPE = None

    def __init__(self):
        self.circuit = self.init_circuit()

    def init_circuit(self, verbose=False):
        input_set = [(np.random.randint(900, 10000, size=FHE.N_SIGMA_SHAPE),
                      np.random.randint(0, 1000))]
        circuit = f1_t_list.compile(input_set, verbose=verbose)
        circuit.keygen()
        return circuit

    def en_run_de_crypt(self, n_sigma, delta):
        decrypted = self.circuit.encrypt_run_decrypt(n_sigma, delta)
        return decrypted

def run(name):
    """
        This part is for pre-processing of t list sensitivity

    :param name: dataset name
    """

    # initiate parameters
    if name in ['iris', 'seeds', 'wine', 'abrupto', 'drebin']:
        nn_neuron = 20
        epoch = 50
        l2 = 0.01
        lr = 0.1
        batch = 256
        total_epsilon_t = [0.1, 1, 10, 100]
    else:
        nn_neuron = 128
        epoch = 70
        l2 = 0.001
        lr = 0.01
        batch = 32
        total_epsilon_t = [1, 10, 100, 1000]

    # load datasets
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

    # normalization mu std
    trait = (trait - trait.mean(axis=0)) / trait.std(axis=0)
    input_dim = trait.shape[1]
    output_dim = len(set(label[:, -1]))

    # encode label
    enc = OneHotEncoder()
    enc.fit(label)
    label = enc.transform(label).toarray()

    # initiate FHE's parameters
    FHE.LABEL_SHAPE = (1, output_dim)
    FHE.OUTPUT_SHAPE = (nn_neuron, output_dim)
    FHE.NOISE_SHAPE = (nn_neuron, output_dim)
    FHE.N_SIGMA_SHAPE = (nn_neuron, output_dim)

    # initiate architecture of nn
    layers = [nn_neuron]
    layers.insert(0, input_dim)
    layers.append(output_dim)

    # initiate range of sensitivity
    minimum = 1000
    maximum = 0

    for total_epsilon in total_epsilon_t:
        # real epsilon for each epoch
        epsilon = total_epsilon / np.sqrt(epoch)

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
        nn3 = Neural_Net(layers)
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
        enc_acc, _ = nn3.accuracy(test_trait, test_label)
        delta = nn3.delta_list
        minimum = min(min(delta), minimum)
        maximum = max(max(delta), maximum)

    # a list of 100 sensitivities
    delta_list = np.linspace(minimum*0.9, maximum*1.1, 100)

    # initiate FHE class
    FHE_T_List.LABEL_SHAPE = (1, output_dim)
    FHE_T_List.OUTPUT_SHAPE = (nn_neuron, output_dim)
    FHE_T_List.NOISE_SHAPE = (nn_neuron, output_dim)
    FHE_T_List.N_SIGMA_SHAPE = (nn_neuron, output_dim)
    fhe_t_list = FHE_T_List()

    # sample
    sample_num = len(trait) - int(trait.shape[0] * 0.3)
    dp_array = np.zeros((epoch, sample_num, 100, nn_neuron, output_dim))

    # calculate dp noise for each epsilon and t list sensitivities
    for total_epsilon in total_epsilon_t:
        total_epsilon = float(total_epsilon)
        print("epsilon:", total_epsilon)
        epsilon = total_epsilon / np.sqrt(epoch)
        sigma = 1 / epsilon
        time_list = []
        for i in range(epoch):
            tic = time.time()
            for z in range(sample_num):
                dp_list = []
                n_sigma_ori = np.random.normal(loc=0, scale=sigma, size=[nn_neuron, output_dim])
                n_sigma = np.around(n_sigma_ori, int(decimal / 2)) * 10 ** int(decimal / 2)
                n_sigma = n_sigma.astype(int)
                min_n_sigma = abs(np.min(n_sigma))
                n_sigma += min_n_sigma
                for j in delta_list:
                    delta = int(np.around(j, int(decimal/2)) * 10**int(decimal/2))
                    decrypted = fhe_t_list.en_run_de_crypt(n_sigma, delta)
                    decrypted = decrypted.astype(float)
                    decrypted = decrypted - delta * min_n_sigma
                    dp = decrypted / 10 ** decimal
                    dp_list.append(dp)
                dp_array_temp = np.array(dp_list).reshape((100, nn_neuron, output_dim))
                dp_array[i, z, :, :, :] = dp_array_temp
            tic_e = time.time() - tic
            time_list.append(tic_e)
        time_list.append(np.mean(time_list))
        np.savetxt(f'./TListDPNoise/{name}/{total_epsilon}_time_t_list.txt', time_list)
        np.save(f'./TListDPNoise/{name}/{total_epsilon}_dp_list.npy', dp_array)
    delta_list = pd.Series(delta_list).to_csv(f'./TListDPNoise/{name}/delta_list.csv')


if __name__ == "__main__":
    name = sys.argv[1]
    folder_path = f"./TListDPNoise/{name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    run(name)











