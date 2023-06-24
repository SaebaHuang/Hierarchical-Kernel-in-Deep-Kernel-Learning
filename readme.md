# Hierarchical Kernels in Deep Kernel Learning
This is the implementation of hierarchical kernels in deep kernel learning. The modified source code of thundersvm is in ./Libs/thundersvm/.

## Prerequisites
CUDA V10.2
python libs: scikit-learn, torch, torchvision, libsvm-official, matplotlib, numpy.

## Usage
Before running the code, one should configure the cuda path in the second row of `./Libs/thundersvm_py/thundersvm.py`.

 
### Moon scattering related
If one wants to perform grid search on hyper-parameters with hierarchical Gaussian kernels, one could set the grid in `./moon_scatter.py` and run `python moon_scatter.py GridSearch`.

If one wants to evaluate on specific kernel with given hyper-parameters and plot the related decision boundary, one could one `python moon_scatter.py Evaluate kernel_id log_C log_lambda` where kernel_id is ranging from 0 to 3 representing the number of layers of hierarchical Gaussian kernels, log_C is the $\log_{2}(C)$ and log_lambda is the $\log_{2}(\lambda)$.

All the results of moon scattering will be automatically stored in `./results/moons/`.

### CIFAR-10 related
For CIFAR-10, only grid search on hyper-parameters with hierarchical Gaussian kernels is supported. One could set the grid in `./cifar10.py` and run `python cifar10.py`.

All the results of moon scattering will be automatically stored in `./results/cifar10/`.

### UCI regression related
For UCI regression tasks, only grid search on hyper-parameters with hierarchical Gaussian kernels is supported. One could set the grid in `./uci_regression.py` and run `python uci_regression.py dataset_id fold_num` where dataset_id is ranging from 0 to 2 representing elevators, kin40k and servo respectively and fold_num is the number of folds in cross-validation.

All the results of moon scattering will be automatically stored in `./results/UCI/`.

### LIBSVM regression related with hierarchical exponential kernels
For LIBSVM regression tasks with hierarchical exponetial kernels, only grid search on hyper-parameters with hierarchical exponential kernels is supported. One could set the grid in `./libsvm_regression.py` and run `python libsvm_regression.py dataset_id fold_num` where dataset_id is ranging from 0 to 2 representing bodyfat, mg and triazines respectively and fold_num is the number of folds in cross-validation.

All the results of moon scattering will be automatically stored in `./results/LIBSVM/`.
