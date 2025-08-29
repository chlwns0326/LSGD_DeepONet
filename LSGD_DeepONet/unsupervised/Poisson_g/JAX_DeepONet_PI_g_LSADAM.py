import os, warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

# imports
import scipy.io as io
import jax, optax
import jax.numpy as jnp
from jax import random
from jax.nn.initializers import glorot_normal, he_normal
jax.config.update('jax_enable_x64', True)
from jax.flatten_util import ravel_pytree

from models import branch_model, trunk_model, model_settings
from train import *
from init_param import *
from networks import *
from loss import *
from LSGD import *
from step import *
from misc import *

# Directories
model_dir = 'unsupervised/Poisson_g/Models/Poisson_PI_g_LSAdam_100_HH_SS_bat50_L3W150_phy_1e-4_regul_1e-9_scale'
data_dir = 'data/' + 'Poisson_BC_scale_data'
createFolder(model_dir +'/models')
createFolder(model_dir +'/losses')

# Hyperparams for models
Nx, Ny = 32, 32 # Uniform grid of size (32+1)*(32+1)
xmin, xmax = 0.0,1.0
ymin, ymax = 0.0,1.0
N_train, batch_size = 1000, 50 # function train data size, adam batch
iter_per_epoch = N_train//batch_size
delay_init = 100 # 100 for LS+Adam, > total_WU for Adam_only
lr = 1e-3 # learning rate
b1 = 0.99 # 1st adam moment (0.9->0.99)
b2 = 0.999 # 2nd adam moment
weights_init = [1e-4,1,1e-9] # weights for physics, data, regularization term, resp. 
adam_num, LS_num = 5, 1 # WU for each hybrid step
total_WU = 100000 # Total work units
total_iters = (iter_per_epoch*adam_num+LS_num) * total_WU
disp_count = 100 # tqdm progress display as new lines

# Initialize
m = (Nx+1)*(Ny+1) # number of trunk input sensors
u_in_foo = jnp.zeros((batch_size,4*Nx+1)) # 1D input
xy_in_foo = jnp.zeros((m,2))

# All He Normal init + Optimizer
key = random.key(1234)
key, *keys = random.split(key,4)
branch_params = branch_model.init(keys[0], u_in_foo)
trunk_params = trunk_model.init(keys[1], xy_in_foo)
last_params = he_normal()(keys[2],(model_settings[0][-4][-1],model_settings[1][0][-1]))

key, *keys = random.split(key,3)
branch_params = apply_he(branch_params, model_settings, 0, keys[0], gmode='N',scale_b=0)
trunk_params = apply_he(trunk_params, model_settings, 1, keys[1], gmode='N',scale_b=0)
params = {'branch': branch_params, 'trunk': trunk_params, 'last': last_params}
optimizer = optax.multi_transform({'adam': optax.inject_hyperparams(optax.adam)(lr,b1=b1,b2=b2), 'zero': optax.set_to_zero()},
            {'branch':'adam', 'trunk':'adam', 'last':'adam'}) 
opt_state = optimizer.init(params)

# Used to restore the trained model parameters
_, unravel_params = ravel_pytree(params)

# Loggers
loss_WU = []
loss_iter = []
weight_WU = []
loss_WU_val = []
loss_logs = {'loss_WU':loss_WU,'loss_iter':loss_iter,'weight_WU':weight_WU,'loss_WU_val':loss_WU_val}

# data load & generation
data = io.loadmat(data_dir)
    
j = 1 # jump/stride
uin_train = jnp.asarray(data['input_train'].astype('float64'))[:N_train,:][:,::j]
uin_val = jnp.asarray(data['input_val'].astype('float64'))[:,:][:,::j]
uout_train = jnp.asarray(data['output_train'].astype('float64'))[:N_train,:,:][:,::j,:][:,:,::j]
uout_val = jnp.asarray(data['output_val'].astype('float64'))[:,::j,:][:,:,::j]

# Output sensors
x_pre = jnp.linspace(xmin,xmax,Nx+1)
y_pre = jnp.linspace(ymin,ymax,Ny+1)
x_in,y_in = jnp.meshgrid(x_pre,y_pre,indexing='xy')
xy_in = jnp.stack((x_in,y_in),axis=-1)
xy_full = jnp.reshape(xy_in,(-1,2))
xy_phys = jnp.reshape(xy_in[1:-1,1:-1,:],(-1,2))
xy_data = jnp.concatenate((xy_in[0,0:-1,:],xy_in[0:-1,-1,:],xy_in[-1,-1:-Nx-1:-1,:],xy_in[-1:-Ny-1:-1,0,:]),axis=0) # (50+50+50+51 by 2)

train(params=params, optimizer=optimizer, delay_init=delay_init,
      uin_train=uin_train, uout_train=uout_train, uin_val=uin_val, uout_val=uout_val,
      xy=xy_full, xy_phys=xy_phys, xy_data=xy_data,
      weights_init=weights_init, 
      batch_size=batch_size, adam_num=adam_num, LS_num=LS_num,
      loss_logs=loss_logs, model_dir=model_dir,
      nIter=total_WU, disp_count=disp_count)