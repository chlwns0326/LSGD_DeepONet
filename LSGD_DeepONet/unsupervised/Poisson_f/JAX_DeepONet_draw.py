import warnings
warnings.filterwarnings('ignore')

import scipy.io as io
import jax
import jax.numpy as jnp
from jax import random
from jax.nn.initializers import glorot_normal
jax.config.update('jax_enable_x64', True)
import matplotlib
from matplotlib import cm
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import branch_model, trunk_model, model_settings
from networks import *
from loss import *
from misc import *

def draw_sample(xy_in,uin,uout,uout_pred,inds,savedir):
    # Sample data draw
    x_in, y_in = xy_in[:,:,0], xy_in[:,:,1]
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    for ind in inds:
        uout_ind,uout_pred_ind = uout[ind,:,:],uout_pred[ind,:,:]
        fig = plt.figure(figsize=(1920*px,1080*px))
        # input
        uin_ind = uin[ind,:,:,0]
        ax = fig.add_subplot(2,2,1,projection='3d')
        im = ax.plot_surface(x_in, y_in, uin_ind.T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.colorbar(im,ax=ax)
        ax.set_title(f'Input function')
        
        ax = fig.add_subplot(2,2,2,projection='3d')
        im = ax.plot_surface(x_in, y_in, uout_ind.T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_title(f'Label')
        plt.colorbar(im,ax=ax)
        
        ax = fig.add_subplot(2,2,4,projection='3d')
        im = ax.plot_surface(x_in, y_in, uout_pred_ind.T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_title(f'Output function prediction')
        plt.colorbar(im,ax=ax)
        
        ax = fig.add_subplot(2,2,3,projection='3d')
        im = ax.plot_surface(x_in, y_in, (uout_ind-uout_pred_ind).T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_title(f'Error')
        plt.colorbar(im,ax=ax)

        fig.suptitle('Data ' + str(ind+1))
        plt.savefig(savedir+'/Data_'+str(ind+1)+'.png')
        plt.close()
        
def data_result_to_npy(uin, uout, xy_in, Q_size, model_dir, save_path, model_path='/models/model_save_besttrain.pickle', suffix=''):
    # Train/Test data save to npy and plot some of them
    params = model_load(path=model_dir+model_path)
    xy_fold = jnp.reshape(xy_in,(-1,2))

    # network output data
    uout_pred_pre = operator_net(params, uin, xy_fold) # P by Q
    uout_pred = uout_pred_pre.reshape((-1,Q_size[0],Q_size[1])).swapaxes(1,2) # P by Qx by Qy
    jnp.save(model_dir + save_path + '/u_out' + suffix + '.npy',uout_pred)
    jnp.save(model_dir + save_path + '/last_param.npy',params['last'])
    
    # sample pics
    inds = range(4)
    draw_sample(xy_in,uin,uout,uout_pred,inds,model_dir+save_path)
    
    # output l2 errors
    N = jnp.shape(uout)[0]
    stat = jnp.zeros((N,2))
    for ind in range(N):
        l2err, l2rel = loss_l2(params,uin[ind:ind+1,:,:], uout[ind:ind+1,:,:], xy_fold)
        stat = stat.at[ind,:].set([l2err,l2rel])
    jnp.save(model_dir + save_path + '/u_out_stat' + suffix + '.npy',stat)

# Directories
model_dir = 'unsupervised/Poisson_f/Models/Poisson_PI_f_LSAdam_100_HH_SS_bat50_L3W150_phy_1e-4_regul_1e-9to14_100to1k'
data_dir = 'data/' + 'Poisson_f_data'
max_echs = [10000,100000] 

# Hyperparams for models
Nx, Ny = 32, 32
xmin, xmax = 0.0,1.0
ymin, ymax = 0.0,1.0
N_train, N_test = 100, 100
Q_size = [Nx+1, Ny+1]

# Initialize
m = (Nx+1)*(Ny+1)   # number of trunk input sensors
u_in_foo = jnp.zeros((N_train,Nx+1,Ny+1,1))
xy_in_foo = jnp.zeros((m,2))

key = random.key(1234)
key, *keys = random.split(key,4)
branch_params = branch_model.init(keys[0], u_in_foo)
trunk_params = trunk_model.init(keys[1], xy_in_foo)
last_params = glorot_normal()(keys[2],(model_settings[0][-4][-1],model_settings[1][0][-1]))

# data load
data = io.loadmat(data_dir)
    
j = 1 # jump/stride
uin_train = jnp.asarray(data['input_train'].astype('float64'))[:N_train,:,:,None][:,::j,:,:][:,:,::j,:]
uin_val = jnp.asarray(data['input_val'].astype('float64'))[:N_test,:,:,None][:,::j,:,:][:,:,::j,:]
uout_train = jnp.asarray(data['output_train'].astype('float64'))[:N_train,:,:][:,::j,:][:,:,::j]
uout_val = jnp.asarray(data['output_val'].astype('float64'))[:N_test,::j,:][:,:,::j]

# Output sensors
x_pre = jnp.linspace(xmin,xmax,Nx+1)
y_pre = jnp.linspace(ymin,ymax,Ny+1)
x_in,y_in = jnp.meshgrid(x_pre,y_pre,indexing='xy')
xy_in = jnp.stack((x_in,y_in),axis=-1)
xy_full = jnp.reshape(xy_in,(-1,2)) 

saveech =  [*range(100,201,20), *range(250,501,50), *range(600,1001,100), 
            *range(1200,2001,200), *range(2500,5001,500), *range(6000,10001,1000), 
            *range(12000,20001,2000), *range(25000,50001,5000), *range(60000,100001,10000), 
            *range(120000,200001,20000), *range(250000,500001,50000), *range(600000,1000001,100000)]

# Train/Test data save and plot for the best saved models 
for max_ech in max_echs:
    createFolder(model_dir +'/train_result_'+str(max_ech))
    createFolder(model_dir +'/test_result_'+str(max_ech))
    loadech = [i for i in saveech if i <= max_ech][::-1]
    for j in loadech:
        # Train data -> Best model with train loss
        try:
            data_result_to_npy(uin_train, uout_train, xy_in, Q_size, model_dir, 
                            save_path='/train_result_'+str(max_ech), model_path='/models/model_save_besttrain_'+str(j)+'.pickle') # train
            break
        except:
            continue
    for j in loadech:
        # Test data -> Best model with validation accuracy (rel L2 error)
        try:
            data_result_to_npy(uin_val, uout_val, xy_in, Q_size, model_dir, 
                            save_path='/test_result_'+str(max_ech), model_path='/models/model_save_bestval_'+str(j)+'.pickle') # train
            break
        except:
            continue    