# Analog-QSVM

## Summery
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

In this repository, we provide code to simulate hybrid, analog, and digital quantum kernels. The QuTiP package is utilized 
for simulating quantum system evolution, while the scikit-learn (sklearn) package is employed for performing PCA, SVR, and SVC.
