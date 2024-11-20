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

# Main Result 
Although digital quantum kernels possess potential quantum advantages, the existence of system noise shadows the practical
implementation in the NISQ era. We attempt to overcome this challenge by designing a quantum kernel in the context of
analog computing. To this aim, we consider two different schemes resulting in the analog quantum kernel, which encodes the
information into Hamiltonian instead of a parametrized circuit, and the hybrid kernel, which encodes classical information
through single qubit gates and replaces the noisy sequential cnot gate by Hamiltonian evolution.
We investigate the performance of our quantum kernels as well as the comparison to the HEA quantum kernel in the
estimating non-Markovianity problem. Such a problem suffers from the exhausted experimental resources on the BLP measure.
Our first contribution is to resolve this problem with the SVR algorithm assisted by the quantum kernel function. In contrast to
the process tomography approach, our approach requires only a sparse measurement to estimate non-Markovianity with good
precision. This highlights the effectiveness of our approach. We further employ our approach to estimate the non-Markovianity
of the dynamics simulated by the cloud-based quantum computer IonQ and justify the practicalness of our approach.
The second contribution was established on the observation of the error robustness feature on the hybrid quantum kernel.
Under a certain choice of parameter, the hybrid quantum kernel preserves high accuracy under the effect of noise. Nevertheless,
the analog quantum kernel suffers from the noise of the classical information, which is induced by the imperfect manipulation
during the encoding process. This leads to the worse performance of analog quantum kernels in realistic conditions. This
result suggests that utilizing single qubit gates for encoding and Hamiltonian evolution as entanglement may be the practical
designation of a quantum kernel in the NISQ era.
Our work naturally opens two future directions. The first will be a comprehensive study of the performance of quantum
kernels under different configurations in the Rydberg atom system. Additionally, one can also inspect the difference between
quantum kernels in terms of the kernel structure. This provides another interesting perspective on unveiling the properties
of quantum kernels. As for the second direction, we suggest that one can utilize our approach on a more complicated
non-Markovianity dynamics and follow our procedure to resolve the problem. To conclude, our work not only provides the
inspiration for designing practical quantum kernels but also offers a valuable tool for efficiently estimating non-Markovianity.
