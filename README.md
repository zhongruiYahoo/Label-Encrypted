
# Label Encrypted

## Environment and Requirment
The environment.txt file contains detailed information about the system environment where the Python project was developed and tested. 
```
artifacts/environment.txt
```
The requirements.txt file lists all the Python packages that are required to run the project, along with their exact versions. 
```
artifacts/requirements.txt
```
## PyTorch Implementation from Scratch
Firstly, we implemented the neural network architecture from scratch and compared it with the PyTorch. Specifically, we tested using the Iris dataset and compared the weights and ROC curves. Run the program in the terminal using the following command:

```python
python PyTorchFromScratch.py
```
In the "others" folder, you can find the following files:
- **figure_6_np_roc.png**
- **figure_6_torch_roc.png**
- **from_scratch_weights.txt**
- **torch_weights.txt**

## Plaintext vs Ciphertext
Secondly, we implemented the encryption and decryption processes using Zama. To demonstrate that these processes have no impact on precision, we tested them using the Iris dataset and compared the weights of neural networks under both ciphertext and plaintext conditions. Run the program in the terminal using the following command:

```python
python PlaintextCiphertext.py
```
You can find the following files in the "others" folder:
- **table_7_ciphertext.txt**
- **table_7_plaintext.txt**

## More Data, Better Accuracy
Practically, larger datasets generally result in higher accuracy compared to smaller datasets. To demonstrate this conclusion, we examined it in two aspects:
### Single dataset testing 
We tested using the "Abrupto" dataset, which was separated into a small dataset and a large dataset. Run the following command:
```python
python SingleTestM1AndM2.py
```
The comparison results between the two models for each repetition are saved in the "others" folder as:
- **table_3_single_test.txt**
### Multi-dataset testing 
This part shows the variation in model accuracy under different ratios of dataset size across multiple datasets. We used 8 datasets: "iris", "seeds", "wine", "abrupto", "drebin", "cifar10", "cifar100", and "purchase10". Each dataset should be input one by one using the following command:
```python
python MultiTestM1AndM2.py dataset_name
```
For example:
```python
python MultiTestM1AndM2.py iris
```
After running all the datasets, the results will be saved in the "multiTest" folder:
- **xx.npy**

**Note: Before proceeding to the plotting stage, we need to run all the datasets in this section.**
### Plotting result of multi-datasets testing
Run the following command to plot the results:
```python
python PlotMultiTestM1AndM2.py
```
The figures was saved in file "others" as **figure_5_multi_accuracy_ratio.pdf**
## Main Experiment 
The main experiment is divided into three parts. All datasets—"iris", "seeds", "wine", "abrupto", "drebin", "cifar10", "cifar100", and "purchase10"—are used in this section. The encryption and decryption methods are implemented using Zama.
### Pre-processing
Initially, we need to pre-calculate the sensitivity list (T list) and the corresponding differential privacy (DP) noise, which will be used in the main experiment. The results, including the running time of the T list, are saved in the "TlitDPNoise" file. Since Zama cannot return the ciphertext alone, our computation time includes both the encryption and decryption processes.
```python
python CalculateTList.py dataset_name
```
For example
```python
python CalculateTList.py seeds
```
### Main experiment
In this part, we compared different models with different epsilon values. The results can be found in the "res" folder:
```python
python MainExperiment.py dataset_name epsilon
```
For example
```python
python MainExperiment.py seeds 1
```
**Note: Overflow may occur if epsilon is too small. We recommend using the epsilon value provided in the paper**
### Randomized response
The only thing to note is that epsilon can be either a scalar or a list. The results are saved as epsilon_dp_list.npy and delta_list.csv in the "res" folder.
```python
python RandomRespond.py --dataset dataset_name --epsilon epsilon
```
For example
```python
python RandomRespond.py --dataset seeds --epsilon 0.1,1
```




