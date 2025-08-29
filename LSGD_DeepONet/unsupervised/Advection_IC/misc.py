# Other useful functions
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax.tree_util import tree_structure
import pickle
from pathlib import Path
from typing import Union

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
def loss_plot(losses, labels, logplot = False):
    # losses : jnp arrays
    # labels : dict() {'colors','legend','title','xlabel','ylabel','save_dir'}
    epochs = jnp.shape(losses)[0]
    numplots = jnp.shape(losses)[1]
    
    for ind in range(numplots):
        plt.plot(range(1,epochs+1), losses[:,ind], color=labels['colors'][ind], label=labels['legend'][ind], zorder=ind)
    if logplot == True:
        plt.yscale('log',base=10)
    plt.title(labels['title'])
    plt.xlabel(labels['xlabel'])
    plt.ylabel(labels['ylabel'])
    plt.legend()
    plt.savefig(labels['save_dir'])
    plt.close()
    
suffix = '.pickle'
def model_save(data: tree_structure, path: Union[str, Path], overwrite: bool = False):
    path = Path(path)
    if path.suffix != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(data, file)

def model_load(path: Union[str, Path]) -> tree_structure:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    if path.suffix != suffix:
        raise ValueError(f'Not a {suffix} file: {path}')
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data    