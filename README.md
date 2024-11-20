# Analog-QSVM

In this repository, we provide code to simulate hybrid, analog, and digital quantum kernels. The QuTiP package is utilized 
for simulating quantum system evolution, while the scikit-learn (sklearn) package is employed for performing PCA, SVR, and SVC.

# Summery
Quantum kernel methods, a promising subset of quantum machine learning algorithms, have shown potential for demonstrating
quantum advantage. While various quantum kernel methods have theoretically outperformed classical counterparts with the
design of quantum circuits, the accumulation of CNOT gate errors has hindered the experimental demonstrations. To address
this challenge, we propose two quantum kernels that replace discrete quantum gate computation with continuous quantum
evolution governed by a controllable Hamiltonian. Our benchmark results demonstrate competitive performance against other
kernel methods. Through error simulation, we establish the fault-tolerant nature of our approach and its ability to preserve
kernel geometry in noisy environments, paving the way for practical implementations of quantum kernel methods. We further
apply the proposed quantum kernel method to predict the non-Markovainity of the given dynamics without performing process
tomography. We demonstrate that our proposed method can accurately indicate non-Markovainity from the limited time points
of the given dynamics.

# How to use the code?
All the useful functions are included in qsvm.

## PCA
Before applying SVM, you can use PCA (Principal Component Analysis) to reduce the dimensionality of features.
PCA transforms the data into a set of linearly uncorrelated components, capturing the most variance in the dataset 
while simplifying the complexity of the input features. We offer function 'Pca' for you.

    '''python
    # Initialization
    pca=Pca(PCA_n=10,StandardizeOrNot=False,sklearnPCA=True)

    ```

We compare three types of quantum kernels, as depicted below. 
<img width="1260" alt="image" src="https://github.com/user-attachments/assets/07c2736a-75a6-4ffb-9015-a09a0230f300">

