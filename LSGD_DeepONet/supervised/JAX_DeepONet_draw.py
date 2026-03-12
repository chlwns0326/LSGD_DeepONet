import warnings
warnings.filterwarnings('ignore')

import scipy.io as io
import jax
import jax.numpy as jnp
from jax import random, jit
from jax.nn.initializers import glorot_normal, he_normal
jax.config.update('jax_enable_x64', True)
import matplotlib
from matplotlib import cm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jax.nn import silu
from models import MLP, CNN_MLP 
from init_param import *
from misc import *

def draw_sample(xy_in,uin,uout,uout_pred,inds,keyword,savedir):
    # Sample data draw
    x_in, y_in = xy_in[:,:,0], xy_in[:,:,1]
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    for ind in inds:
        uout_ind,uout_pred_ind = uout[ind,:,:],uout_pred[ind,:,:]
        fig = plt.figure(figsize=(1920*px,1080*px))
        # input
        if keyword == 'Advection_IBC':
            uin_ind = uin[ind,:]
            ax = fig.add_subplot(2,2,1)
            im = ax.plot(jnp.linspace(-1,1,jnp.shape(uin_ind)[0]), uin_ind)
        elif keyword == 'DiffReact':
            uin_ind = uin[ind,:]
            ax = fig.add_subplot(2,2,1)
            im = ax.plot(x_in[0,:], uin_ind)
        elif keyword == 'Poisson_kappa':
            uin_ind = uin[ind,:,:,0]
            ax = fig.add_subplot(2,2,1,projection='3d')
            im = ax.plot_surface(x_in[:-1,:-1], y_in[:-1,:-1], uin_ind.T, cmap=cm.coolwarm, linewidth=0, antialiased=False) # kappa
            plt.colorbar(im,ax=ax)
        elif keyword == 'Poisson_g':
            uin_ind = uin[ind,:]
            ax = fig.add_subplot(2,2,1)
            im = ax.plot(jnp.linspace(0,4,jnp.shape(uin_ind)[0]), uin_ind) # Pois BC
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
        
def data_result_to_npy(uin, uout, xy_in, Q_size, model_dir, in_ftn, save_path, model_path='/models/model_save_besttrain.pickle', suffix=''):
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
    draw_sample(xy_in,uin,uout,uout_pred,inds,in_ftn,model_dir+save_path)
    
    # output l2 errors
    N = jnp.shape(uout)[0]
    stat = jnp.zeros((N,2))
    for ind in range(N):
        if uin.ndim == 2:
            l2err, l2rel = loss_l2(params,uin[ind:ind+1,:], uout[ind:ind+1,:,:], xy_fold)
        elif uin.ndim == 4:
            l2err, l2rel = loss_l2(params,uin[ind:ind+1,:,:], uout[ind:ind+1,:,:], xy_fold)
        stat = stat.at[ind,:].set([l2err,l2rel])
    jnp.save(model_dir + save_path + '/u_out_stat' + suffix + '.npy',stat)

## Loss        
@jit
def loss_l2(params, u_in, u_out, xy):
    u_pred = operator_net(params, u_in, xy)
    u_out_ = jnp.transpose(u_out,(0,2,1)).reshape((u_out.shape[0], -1))
    axis = tuple(range(1,u_out_.ndim))        
    diff_sq = jnp.mean((u_out_ - u_pred)**2,axis=axis)
    data_sq = jnp.mean((u_out_)**2,axis=axis)
    l2relsqs = diff_sq/data_sq
    l2loss = jnp.mean(diff_sq) 
    l2rel = jnp.mean(jnp.sqrt(l2relsqs))
    return l2loss, l2rel

@jit
def loss_reguls(params):
    loss = jnp.sum(params['last']**2)
    return loss

@jit
def loss(params, u_in, u_out, xy, weights):
    loss_l2_value, l2rel = loss_l2(params, u_in, u_out, xy)
    loss_reguls_value = loss_reguls(params)
    loss = weights[0]*loss_l2_value + weights[1]*loss_reguls_value
    return loss

@jit
def loss_comps(params, u_in, u_out, xy, weights):
    loss_l2_value, l2rel = loss_l2(params, u_in, u_out, xy)
    loss_reguls_value = loss_reguls(params)
    loss = weights[0]*loss_l2_value + weights[1]*loss_reguls_value
    return jnp.array([loss, weights[0]*loss_l2_value, weights[1]*loss_reguls_value, l2rel])
## Networks
@jit
def operator_net(params, u, xy):
    B = branch_model.apply(params['branch'], u) 
    T = trunk_model.apply(params['trunk'], xy) 
    W = params['last']
    outputs = B @ W @ T.T
    return outputs
    
# Directories (Current directory: /LSGD_DeepONet/supervised)
folder = 'Models_v3/'
keywords = ['Advection_IBC', 'DiffReact', 'Poisson_kappa', 'Poisson_g']
model_descs = ['L3W100','L3W100','L3W150','L3W150']
lambdas = [1e-4,1e-7,1e-8,1e-4]
# etimes = [4000,4000,600,5000]
etimes = [2000,2000,200,2000] # tighter
datafolder = '../data/'
LS_nums = [0,1] # Adam = 0, LS+Adam > 0
seeds = ['1','2','3']

# Common hyperparams
N_train, N_test = 100, 100  # Number of training/validation function dataset to use
delay_init_pre = 500 # Delayed LS step after the Adam epochs
Nx, Ny = 32, 32 # Uniform grid of size (32+1)*(32+1)
xmin, xmax = 0.0, 1.0 
ymin, ymax = 0.0, 1.0
Q_size = [Nx+1, Ny+1]

saveech =  [10,20,30,50, *range(100,201,20), *range(250,501,50), *range(600,1001,100), 
                *range(1200,2001,200), *range(2500,5001,500), *range(6000,10001,1000), 
                *range(12000,20001,2000), *range(25000,50001,5000), *range(60000,100001,10000), 
                *range(120000,200001,20000), *range(250000,500001,50000), *range(600000,1000001,100000)]    

for keyword,model_desc,lamb,etime in zip(keywords,model_descs,lambdas,etimes):
    # Hyperparameters
    weights_init = [1,lamb] # weights for data, regularization term, resp. 
  
    # Initialize & Data load
    if keyword == 'Advection_IBC':
        u_in_foo = jnp.zeros((N_train,2*Nx+1)) 
        dataname = 'Advection_PQ_data'
        in_dims = 1
        model_settings = [([100]*2,silu,1,False),([100]*3,silu,1,False)]
        branch_model = MLP(*model_settings[0])
        trunk_model = MLP(*model_settings[1])
    elif keyword == 'DiffReact':
        u_in_foo = jnp.zeros((N_train,Nx+1))
        dataname = 'ADR_f_data'
        in_dims = 1
        model_settings = [([100]*2,silu,1,False),([100]*3,silu,1,False)]
        branch_model = MLP(*model_settings[0])
        trunk_model = MLP(*model_settings[1])
    elif keyword == 'Poisson_kappa':
        u_in_foo = jnp.zeros((N_train,Nx,Ny,1))
        dataname = 'Poisson_kappa_data'
        in_dims = 2
        model_settings = [([[16,(2,2),(2,2),'VALID'],[32,(2,2),(2,2),'VALID'],[64,(2,2),(2,2),'VALID']],[150],silu,1,False),([150]*3,silu,1,False)]
        branch_model = CNN_MLP(*model_settings[0])
        trunk_model = MLP(*model_settings[1])
    elif keyword == 'Poisson_g':
        u_in_foo = jnp.zeros((N_train,4*Nx+1))
        dataname = 'Poisson_g_data'
        in_dims = 1
        model_settings = [([150]*2,silu,1,False),([150]*3,silu,1,False)]
        branch_model = MLP(*model_settings[0])
        trunk_model = MLP(*model_settings[1])
    m = (Nx+1)*(Ny+1)  # number of trunk input sensors
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
    
    for seed in seeds:
        for LS_num in LS_nums:
            # Adam vs LS+Adam
            if LS_num > 0: # LS+Adam
                batch_size, batch_new = 200, 50 # predetermined adam batch 200->50
                adam_num = 2 # Adam epochs for each WU (LS step performed for each adam_num epochs)
                reg_lambda = '{:.0e}'.format(weights_init[1])
                model_dir = folder + keyword + '/' + keyword + '_LSAdam_R' + str(adam_num) + '_' + str(delay_init_pre) + \
                    '_HH_SS_bat'+ str(batch_size) + 'to' + str(batch_new) + '_' + \
                    model_desc + '_seed_' + seed + '_regul_' + reg_lambda 
            else: # Adam
                batch_size, batch_new = 200, 200 # predetermined adam batch 
                model_dir = folder + keyword + '/' + keyword + '_Adam_NoLS_HH_SS_bat'+ str(batch_size) + '_' + \
                    model_desc + '_seed_' + seed
            # load train time and find the max epoch less than the train time
            mdir_time = model_dir + '/losses/training_time.npy'
            max_ech = jnp.argmax(jnp.load(mdir_time) > etime) - 1
            # Train/Test data save and plot for the best saved models 
            createFolder(model_dir +'/train_result_time_'+str(etime))
            createFolder(model_dir +'/test_result_time_'+str(etime))
            loadech = [i for i in saveech if i <= max_ech][::-1] 
            for j in loadech:
                # Train data -> Best model with train loss
                try:
                    data_result_to_npy(uin_train, uout_train, xy_in, Q_size, model_dir, keyword,
                                    save_path='/train_result_time_'+str(etime), model_path='/models/model_save_besttrain_'+str(j)+'.pickle') # train
                    break
                except:
                    continue
            for j in loadech:
                # Test data -> Best model with validation accuracy (rel L2 error)
                try:
                    data_result_to_npy(uin_val, uout_val, xy_in, Q_size, model_dir, keyword,
                                    save_path='/test_result_time_'+str(etime), model_path='/models/model_save_bestval_'+str(j)+'.pickle') # test
                    break
                except:
                    continue    