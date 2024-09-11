import copy
import numpy as np
import concrete.numpy as cnp


@cnp.compiler({"label": "encrypted",
               "output": "clear",
               "noise": "clear",
               "n_sigma": "encrypted"})
def f1(label, output, noise, n_sigma):
    gradient = np.multiply(label, output) + noise + n_sigma
    return gradient


class FHE:

    LABEL_SHAPE = None
    OUTPUT_SHAPE = None
    NOISE_SHAPE = None
    N_SIGMA_SHAPE = None

    def __init__(self):
        self.circuit = self.init_circuit()

    def init_circuit(self, verbose=False):
        input_set = [(np.random.randint(900, 10000, size=FHE.LABEL_SHAPE),
                      np.random.randint(900, 10000, size=FHE.OUTPUT_SHAPE),
                      np.random.randint(100, 10000, size=FHE.NOISE_SHAPE),
                      np.random.randint(900, 10000, size=FHE.N_SIGMA_SHAPE))]
        circuit = f1.compile(input_set, verbose=verbose)
        circuit.keygen()
        return circuit

    def en_run_de_crypt(self, label, output, noise, n_sigma):
        n_s = copy.deepcopy(n_sigma)
        decrypted = self.circuit.encrypt_run_decrypt(label, output, noise, n_s)
        return decrypted


