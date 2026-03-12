import os, warnings
warnings.filterwarnings('ignore')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true' 
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.35' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

# imports
import scipy.io as io
import jax, optax
import jax.numpy as jnp
from jax import random
from jax.nn.initializers import glorot_normal, he_normal
jax.config.update('jax_enable_x64', True)

from models import branch_model, trunk_model, model_settings # need to modify models in models.py
from train import *
from init_param import *
from networks import *
from loss import *
from LSGD import *
from step import *
from misc import *

# Seed control for reproducibility
MasterKey = 1
seedA, seedB = random.randint(random.key(MasterKey),shape=(2,),minval=0,maxval=2**31)

# Directories (Current directory: /LSGD_DeepONet/supervised)
folder = 'Models_v3/'
keyword = 'Advection_IBC' # 'Advection_IBC', 'DiffReact', 'Poisson_kappa', 'Poisson_g'
model_desc = 'L3W100'
datafolder = '../data/'

# Hyperparameters
LS_num = 1 # Adam = 0, LS+Adam > 0
weights_init = [1,1e-4] # weights for data, regularization term, resp. 
N_train = 1000 # Number of training function dataset 
delay_init_pre = 500 # Delayed LS step after the Adam epochs
lr, b1, b2 = 1e-3, 0.99, 0.999 # Adam learning rate/1st moment/2nd moment

# Adam vs LS+Adam
if LS_num > 0: # LS+Adam
    batch_size = 200 # predetermined adam batch 200->50
    batch_new = 50
    adam_num = 2 # Adam epochs for each WU (LS step performed for each adam_num epochs)
    total_WU = 200000 // adam_num 
    delay_init = delay_init_pre//adam_num
    reg_lambda = '{:.0e}'.format(weights_init[1])
    model_dir = folder + keyword + '/' + keyword + '_LSAdam_R' + str(adam_num) + '_' + str(delay_init_pre) + \
        '_HH_SS_bat' + str(batch_size) + 'to' + str(batch_new) + '_' + \
        model_desc + '_seed_' + str(MasterKey) + '_regul_' + reg_lambda 
else: # Adam
    batch_size = 200 # predetermined adam batch
    batch_new = batch_size
    adam_num = 1 # Always epochwise
    weights_init[1] = 0 # No last layer regularization weight
    total_WU = 500000 // adam_num 
    delay_init = delay_init_pre
    model_dir = folder + keyword + '/' + keyword + '_Adam_NoLS_HH_SS_bat'+ str(batch_size) + '_' + \
        model_desc + '_seed_' + str(MasterKey)

createFolder(model_dir +'/models')
createFolder(model_dir +'/losses')

# Hyperparams for models
Nx, Ny = 32, 32 # Uniform grid of size (32+1)*(32+1)
xmin, xmax = 0.0, 1.0 
ymin, ymax = 0.0, 1.0
disp_count = int(0.002*total_WU) # tqdm progress display as new lines 

# Initialize & Data load
if keyword == 'Advection_IBC':
    u_in_foo = jnp.zeros((batch_size,2*Nx+1)) 
    dataname = 'Advection_PQ_data'
    in_dims = 1
elif keyword == 'DiffReact':
    u_in_foo = jnp.zeros((batch_size,Nx+1))
    dataname = 'ADR_f_data'
    in_dims = 1
elif keyword == 'Poisson_kappa':
    u_in_foo = jnp.zeros((batch_size,Nx,Ny,1))
    dataname = 'Poisson_kappa_data'
    in_dims = 2
elif keyword == 'Poisson_g':
    u_in_foo = jnp.zeros((batch_size,4*Nx+1))
    dataname = 'Poisson_g_data'
    in_dims = 1
m = (Nx+1)*(Ny+1)  # number of trunk input sensors
xy_in_foo = jnp.zeros((m,2))

# All He Normal init + Optimizer
key = random.key(seedA)
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

# Loggers
loss_WU = []
weight_WU = []
loss_WU_val = []
loss_logs = {'loss_WU':loss_WU,'weight_WU':weight_WU,'loss_WU_val':loss_WU_val}

# data load & generation
data_dir = datafolder + dataname
data = io.loadmat(data_dir)
j = 1 # jump/stride
if in_dims == 2:
    uin_train = jnp.asarray(data['input_train'].astype('float64'))[:N_train,:,:,None][:,::j,:,:][:,:,::j,:]
    uin_val = jnp.asarray(data['input_val'].astype('float64'))[:,:,:,None][:,::j,:,:][:,:,::j,:]  
else:
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

train(params=params, optimizer=optimizer, seed=seedB, delay_init=delay_init,
      uin_train=uin_train, uout_train=uout_train, uin_val=uin_val, uout_val=uout_val, xy=xy_full,
      weights_init=weights_init,
      batch_size=batch_size, batch_new=batch_new, adam_num=adam_num, LS_num=LS_num,
      loss_logs=loss_logs, model_dir=model_dir,
      nIter=total_WU, disp_count=disp_count)