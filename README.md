# Hybrid Least Squares/Gradient Descent Methods for DeepONets (J. Choi et al., 2026)

Python implementation of the experiments of **Hybrid least squares/gradiet descent method for DeepONets** with JAX library.
This repository contains data and python codes for experiments of hybrid least squares/gradiet descent method. 

https://arxiv.org/abs/2508.15394

## Requirements 

- **python** 3.12.13
- **jax** 0.4.26 (\[cuda13\] if available)
- **equinox** 0.11.4 
- **diffrax** 0.5.1
- **optax** 0.2.2
- **flax** 0.8.2
- **scipy** 1.13.0 
- **matplotlib** 3.8.4 
- **tqdm** 4.66.2 

Note: Newer versions of python and some packages might not operate properly, especially in JAX related packages. 

## Setup

1. Clone the repository:
```
bash
cd [Target_directory]
git clone https://github.com/chlwns0326/LSGD_DeepONet.git
```

2. Install dependencies


## Datasets

All datasets except 'Supervised Navier-Stokes' example are already included in **/data** folder as _.mat_ files. 

To obtain the Navier-Stokes dataset, you need to execute **/data_generation/NS_generator.py** by your own. (~780MB)

## Project structure

```
LSGD_DeepONet
├─ data
├─ data_generation
├─ supervised
│  ├─ Models
│  │  ├─ Advection_IBC
│  │  ├─ DiffReact
│  │  ├─ Poisson_kappa
│  │  └─ Poisson_g
|  ├─ JAX_DeepONet_Label_LSADAM.py
|  ├─ JAX_DeepONet_Label_LSADAM_all_in_one.py
|  ├─ JAX_DeepONet_draw.py
|  ├─ LSGD.py
│  ├─ init_param.py
│  ├─ initialization.py
│  ├─ loss.py
│  ├─ misc.py
│  ├─ models.py
│  ├─ networks.py
│  ├─ step.py
│  └─ train.py
├─ supervised_N-S
└─ unsupervised
   ├─ Advection_IBC
   ├─ Poisson_g
   └─ Poisson_f
```
**/supervised** folder is partially expanded to explain its subfolders and python files.

## Description of each directory

### Subdirectory
- **/data** folder contains datasets used in the training DeepONets **/supervised**, **/supervised_N-S** and **/unsupervised**.
- **/data_generation** folder contains python files for generating each corresponding dataset. 
- **/supervised** folder contains python files for training DeepONets with supervised learning for **Advection**, **Diffusion-Reaction**, **Poisson(coefficient)** and **Poisson(BC)** PDE problems.
- **/supervised_N-S** folder contains python files for training DeepONets with supervised learning for **Navier-Stokes(vorticity)** PDE problem.
- **/unsupervised** folder contains python files for training DeepONets with unsupervised learning for **Advection**, **Poisson(BC)** and **Poisson(source)** PDE problems.
- **/supervised**, **/supervised_N-S** and **/unsupervised** folders contain _Model_ subfolders, which contain partial sample training results for each PDE problem.

### Python files 

For each (a) **/supervised**, (b) **/supervised_N-S**, (c1) **/unsupervised/Advection_IBC**, (c2) **/unsupervised/Poisson-g** and (c3) **/unsupervised/Poisson-f**, 
the directory contains following python files. 

+ [x] **JAX_DeepONet_[]_LSADAM.py** is the main python file for DeepONet training. It handles model settings and hyperparameters, and unifies other python modules. The filename varies in each subdirectory.
+ [x] **JAX_DeepONet_draw.py** (JAX_DeepONet_NS_draw.py for (b)) draws sample results from saved model and outputs model evaluation result as .npy files. 
+ [ ] **initialization.py** and **init_param.py** initialzes model parameters by using 'He', 'Glorot' and 'Box(ReLU-only)'. 
+ [x] **models.py** defines the detailed DeepONet model structure. 
+ [ ] **misc.py** contains some utility and miscellaneous functions. 
+ [x] **train.py** controlls the overall training procedure of DeepONet. 
+ [ ] **networks.py** explicitly forms DeepONet and computes the model output. 
+ [ ] **loss.py** contains loss functions and metrics used for model training(backpropagation) and validation(evaluation). 
+ [ ] **step.py** updates the model parameters by taking gradient descent (GD) step or least squares (LS) step. 
+ [ ] **LSGD.py** contains the LS step for LSGD method.
+ [x] In (a), **JAX_DeepONet_Label_LSADAM_all_in_one.py** is the all-in-one file which merges **JAX_DeepONet_Label_LSADAM.py**, **models.py**, **train.py**, **networks.py**, **loss.py**, **step.py** and **LSGD.py** into one single .py file. You may use this one instead. 

To train and evaluate DeepONet model, you first need to modify the checked .py files. 

## Basic procedure

0. Choose the PDE problem you want to solve. Prepare dataset from **/data_generation** if needed. 
1. Define branch/trunk model structures and set model settings in **models.py**. You need to modify the latter part of the file.
2. In **JAX_DeepONet_[]_LSADAM.py**, modify CPU/GPU information, master seed number, load/save directory, hyperparameters for training and model, initialization method, etc.
You may modify **train.py** to enable decaying regularization weight. 
3. Run **JAX_DeepONet_[]_LSADAM.py** directly to train the corresponding DeepONet.
4. After training, modify and run **JAX_DeepONet_draw.py** to evaluate and visualize the model output. 


## Authors

- Overall project development: **Jun Choi** (Department of Mathematical Sciences, KAIST) (chlwns0326@kaist.ac.kr)
- Corresponding author: **Chang-Ock Lee** (Department of Mathematical Sciences, KAIST) (colee@kaist.edu)
- Co-author: **Minam Moon** (Department of Mathematics, Korea Military Academy) (minammoon23@gmail.com)

## Citation

```
@misc{choi2025hybridsquaresgradientdescentmethods,
      title={Hybrid Least Squares/Gradient Descent Methods for DeepONets}, 
      author={Jun Choi and Chang-Ock Lee and Minam Moon},
      year={2025},
      eprint={2508.15394},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.15394}, 
}
```

## License

This project is licensed under the [MIT Liscnse](LICENSE).
