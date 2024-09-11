from CalculateTList import *
from sklearn.preprocessing import OneHotEncoder


"""
    Model on Plaintext
        vs
    Model on Ciphertext
"""

print("Dataset: Iris")

# load data
data_name = 'iris'
file = pd.read_csv('./data/iris/iris.data', header=None)
file = file.sample(frac=1)
data = file.iloc[:, :-1].values
target_raw = file.iloc[:, -1].values.reshape(-1, 1)

# normalization
trait = (data - data.mean(axis=0)) / data.std(axis=0)
input_dim = trait.shape[1]
output_dim = len(set(target_raw[:, 0]))

# encode target to onehot
enc = OneHotEncoder()
enc.fit(target_raw)
label = enc.transform(target_raw).toarray()

# set hyperparameters
layers = [4]
FHE.LABEL_SHAPE = (1, output_dim)
FHE.OUTPUT_SHAPE = (layers[-1], output_dim)
FHE.NOISE_SHAPE = (layers[-1], output_dim)
FHE.N_SIGMA_SHAPE = (layers[-1], output_dim)
layers.insert(0, input_dim)
layers.append(output_dim)
lr = 0.1
weight_decay = 1e-2
epoch = 50
batch_size = 16
epsilon = 0.1

# splite data
test_idx = np.random.choice(range(trait.shape[0]), int(trait.shape[0] * 0.3), replace=False)
test_trait, test_label = trait[test_idx], label[test_idx]

# remainder is D2 + D1 (train M2)
remain_idx = list(set(range(trait.shape[0])) - set(test_idx))
train_trait, train_label = trait[remain_idx], label[remain_idx]

sequence = list(range(len(train_trait)))

# train with plaintext
print("train with plaintext")
nn1 = Neural_Net(layers)
train_all_start = time.time()
nn1.train2(feature=train_trait,
           label=train_label,
           lr=lr,
           weight_decay=weight_decay,
           epoch=epoch,
           batch_size=batch_size,
           indices=sequence,
           encryption_state=False,
           epsilon=epsilon)
train_all_end = time.time()
all_acc, out = nn1.accuracy(test_trait, test_label)
all_time = train_all_end - train_all_start

# train with ciphertext
print("train with ciphertext")
train_enc_start = time.time()
nn3 = Neural_Net(layers, nn1.InitWeight, init_w=True)
nn3.train2(feature=train_trait,
          label=train_label,
          lr=lr,
          weight_decay=weight_decay,
          epoch=epoch,
          batch_size=batch_size,
          indices=sequence,
          encryption_state=True,
          epsilon=epsilon)
train_enc_end = time.time()
enc_acc, out = nn3.accuracy(test_trait, test_label)
enc_time = train_enc_end - train_enc_start

np.set_printoptions(precision=4)

print("weights of neural network to plaintext")
with open('./others/table_7_plaintext.txt', 'w', encoding='utf-8') as f:
    for i in nn1.W:
        f.write(str(i))
        f.write('\n')
        print(i.T)

print("weights of neural network to ciphertext")
with open('./others/table_7_ciphertext.txt', 'w', encoding='utf-8') as f:
    for i in nn3.W:
        f.write(str(i))
        f.write('\n')
        print(i.T)


