import time
from sklearn.metrics import roc_curve, auc
from torch import nn
from torch.nn import functional as F
import torch
import tqdm
from fhe import *

np.set_printoptions(suppress=True)

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
        # print(self.net)
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


# training model from PyTorch
def torch_train(feature, label, w, hidden, lr, weight_decay, epoch, batch_size, indices):
    net = NN(init_weight=w, layers=hidden)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    net.train()
    for _ in range(epoch):
        for f, l in data_iter(feature, label, batch_size, indices):
            f = torch.FloatTensor(np.atleast_2d(f))
            l = torch.FloatTensor(np.atleast_2d(l))
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


# neural network from scratch
class Neural_Net:

    # initialize net and weight
    def __init__(self, layers: list,  *args, init_w=False, delta_list_t=0, dp_noise=0):
        """

        :param layers: network structure
        :param args:
        :param init_w: given initial weight
        """

        self.W = []                                         # weights
        self.Layers = []                                    # net structure in detail
        self.Bias = []                                      # bias
        self.LayerOutput = []                               # layer outputs
        self.InitWeight = []                                # initial weight
        self.fhe = FHE()                                    # enc fhe model
        self.delta_list = []
        self.delta_list_t = delta_list_t
        self.dp_noise = dp_noise
        self.count = 0

        # initialize layers info
        for l in range(len(layers) - 1):
            if l + 1 != len(layers) - 1:
                self.Layers.append((layers[l], layers[l + 1], True, 'sigmoid'))
            else:
                self.Layers.append((layers[l], layers[l + 1], False, 'logSoftmax'))

        # display layers
        # for i in self.Layers:
        #     print(i)

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
    def en_de_cryption_step(self, labels, outputs, delta, encryption_state, decimal, epsilon, epoch):
        part2 = []
        if encryption_state.__eq__(True):
            # get dp noise
            idx = np.argmin(abs(self.delta_list_t-delta))
            noise = self.uniform_noise(batch=len(outputs),
                                       shape=outputs[0].shape,
                                       decimal=decimal)

            # training for a batch
            for l, o, n in zip(labels, outputs, noise):
                dp_noise = copy.deepcopy(self.dp_noise[epoch][self.count][idx])
                self.count += 1
                n_sigma = np.around(dp_noise, int(decimal)) * 10 ** int(decimal)
                n_sigma = n_sigma.astype(int)
                min_n_sigma = abs(np.min(n_sigma))
                n_sigma += min_n_sigma
                o = np.around(o, decimals=decimal) * 10**decimal
                n = np.around(n, decimals=decimal) * 10**decimal
                o, n = o.astype(int), n.astype(int)
                l = l.reshape(1, -1).astype(int)

                # fhe model
                decrypted1 = self.fhe.en_run_de_crypt(label=l,
                                                      output=o,
                                                      noise=n,
                                                      n_sigma=n_sigma)
                decrypted1 = decrypted1.astype(float)
                decrypted1 -= min_n_sigma
                decrypted = decrypted1.astype(float)
                decrypted -= n
                decrypted = decrypted / 10**decimal
                part2.append(decrypted)
        else:
            # for i in range(len(labels)):
            #     l, o = labels[i], outputs[i]
            #     part2.append(l * o)
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

    def train(self, feature, label, lr, weight_decay, epoch, batch_size, indices, epsilon, encryption_state=False):
        delta = 0
        for i in range(epoch):
            if encryption_state:
                delta = 2 * self.calculate_sensitivity(feature) / batch_size
                self.delta_list.append(delta)
            self.count = 0

            # training for a batch
            for ff, ll in data_iter(feature, label, batch_size, indices):
                # gradients of batch sample
                Gradient = None
                labels, outputs = self.prepare(ff, ll)
                part2 = self.en_de_cryption_step(labels=labels,
                                                 outputs=outputs,
                                                 delta=delta,
                                                 encryption_state=encryption_state,
                                                 decimal=6,
                                                 epsilon=epsilon,
                                                 epoch=i)
                # calculate gradient
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

    # calculate accuracy
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


    def train2(self, feature, label, lr, weight_decay, epoch, batch_size, indices, epsilon, encryption_state=False):
        """
        It is used for Table 7 Model plaintext and ciphertext

        :param feature:
        :param label:
        :param lr:
        :param weight_decay:
        :param epoch:
        :param batch_size:
        :param indices:
        :param epsilon:
        :param encryption_state:
        :return:
        """
        delta = None
        time_list = []
        for i in tqdm.tqdm(range(epoch)):
            t_s = time.time()
            if encryption_state:
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
                                                 epsilon=epsilon,
                                                 epoch=i)
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
            t_e = time.time()
            time_list.append(t_e - t_s)