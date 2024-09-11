import matplotlib.pyplot as plt
import numpy as np

names = ['iris.npy', 'seeds.npy', 'wine.npy', 'abrupto.npy', 'drebin.npy', 'cifar10.npy', 'cifar100.npy', 'purchase10.npy']
for name in names:
    data = np.load(f'./multiTest/{name}')
    ratio = data[0, :] / data[1, :]
    plt.plot(np.arange(0.1, 1.1, 0.1), ratio, '--*', label=name.split('.')[0])
plt.xlabel('Size ratio of D1 and D2')
plt.ylabel('Accuracy ratio of M1 and M2')
plt.xticks(np.arange(0.1, 1.1, 0.1))
plt.yticks(np.arange(0.1, 1.1, 0.1))
plt.legend(loc='upper left')
plt.savefig('./others/figure_5_multi_accuracy_ratio.pdf')
