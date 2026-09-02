# Code for 'Hybrid Least Squares/Gradient Descent Methods for DeepONets' (J. Choi et al., 2026)

Python implementation of the experiments of **Hybrid least squares/gradiet descent method for DeepONets** with JAX library.
This repository contains data and python codes for experiments of hybrid least squares/gradiet descent method.
https://arxiv.org/abs/2508.15394

## Requirements 

- Python 3.12.10
- Python library dependency TBA

## Installation

- TBA

## Datasets

All datasets are generated from the corresponding generating codes(.py) in '/data_generation' folder. 

All datasets except 'Supervised Navier-Stokes' example are already included in '/data' folder as '.mat' files. 

To obtain the Navier-Stokes dataset, you need to execute '/data_generation/NS_generator.py' by your own. (~780MB)

## Project structure

```
LSGD_DeepONet
├─ data
├─ data_generation
├─ supervised
│  ├─ Models
│  │  ├─ Advection_IBC
│  │  ├─ DiffReact
│  │  ├─ Poisson_g
│  │  └─ Poisson_kappa
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
   ├─ Poisson_f
   └─ Poisson_g
LICENSE
README.md
```
'/supervised' folder is partially expanded to explain its subfolders and python files.


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
