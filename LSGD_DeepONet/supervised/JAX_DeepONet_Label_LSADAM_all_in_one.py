#### All in one file where modification of other python files are not needed for different trainings
#### need to import functions from "models.py", "init_param.py", and "misc.py"

import os, warnings
warnings.filterwarnings('ignore')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true' 
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.35' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# imports
import scipy.io as io
import jax, optax, time
import jax.numpy as jnp
from jax import random, grad, jit
from functools import partial
from jax.nn.initializers import glorot_normal, he_normal
jax.config.update('jax_enable_x64', True)
from jax.nn import silu, relu
from models import MLP, CNN_MLP, res_MLP, CNN_res_MLP
from init_param import *
from misc import *
from tqdm import tqdm

# Seed control for reproducibility
MasterKey = 1
seedA, seedB = random.randint(random.key(MasterKey),shape=(2,),minval=0,maxval=2**31)

#### Things we will modify
batch_Adam = 200 # initial adam batch 
batch_LSAdam = 50
adam_num = 2 # 1 if Adam
Adam_const = 5000 # 5000
LSAdam_const = 8000 # 8000 

# Directories (Current directory: /LSGD_DeepONet/supervised)
folder = 'Models/'
subfolder = 'temp/' # 'Ablation_new/' # 'New_train/'
keyword = 'Poisson_g'
model_desc = 'L3W150'
datafolder = '../data/'

# Hyperparameters
LS_num = 1 # Adam = 0, LS+Adam = 1
weights_init = [1,1e-4] # weights for data, regularization term, resp. 
N_train = 1000 # Number of training function dataset 
delay_init_pre = 500 
lr, b1, b2 = 1e-3, 0.99, 0.999 # Adam learning rate/1st moment/2nd moment
activation = silu
network = MLP
initialization = apply_he
weight_decay = False
decay_target = 1 # 1e-3
decay_start = 500 # > delay_init_pre
decay_end = 500 # 500

# Hyperparams for models
Nx, Ny = 32, 32 # Uniform grid of size (32+1)*(32+1)
xmin, xmax = 0.0, 1.0 
ymin, ymax = 0.0, 1.0

## Train
def train(params, optimizer, seed, delay_init,
          uin_train, uout_train, uin_val, uout_val, xy,
          weights_init, 
          batch_size, batch_new, adam_num, LS_num,
          weight_decay, decay_target, decay_start, decay_end,
          loss_logs, model_dir,
          nIter, disp_count):
    
    N_train = jnp.shape(uin_train)[0]
    
    opt_state = optimizer.init(params)
    iter_per_epoch = N_train//batch_size
    key = random.key(seed) # 1133 default
    
    # Number of manual save WUs
    savewu =  [10,20,30,50, *range(100,201,20), *range(250,501,50), *range(600,1001,100), 
                *range(1200,2001,200), *range(2500,5001,500), *range(6000,10001,1000), 
                *range(12000,20001,2000), *range(25000,50001,5000), *range(60000,100001,10000), 
                *range(120000,200001,20000), *range(250000,500001,50000), *range(600000,1000001,100000)]
        
    start_time = time.time()
    total_time = 0
    
    weights = [weights_init[0],0] # Adam LL
    WU_time = [] # 
        
    # Main training loop
    for wu in tqdm(range(nIter)):
        # timer init
        part_time_1, part_time_2, part_time_3 = 0,0,0

        ## Manual weight decay
        ## Decaying weights from a(start) to a*b(end)
        if weight_decay == True:
            if wu > decay_start and wu <= decay_end:
                rate = decay_target**(1/(decay_end-decay_start)) # Gradual decay for each WU
                p1 = weights[1] * rate
                weights = [weights_init[0],p1] 
        
        # LL regul coeff
        if wu == delay_init:
            weights = [weights_init[0],weights_init[1]] # LS LL
        # GD step         
        if wu == 0: # No Adam, init state
            # only forward
            losses = loss_comps(params, uin_train, uout_train, xy, weights)
        else: # Adam, training backprop
            for ad_repeat in range(adam_num):
                # Batch formation
                params['last'].block_until_ready() ### Foo code 
                timer_1 = time.perf_counter()
                key, subkey = random.split(key)
                perm = random.permutation(key, N_train)
                uin_batch = jnp.split(uin_train[perm,:],iter_per_epoch) # N_train/batch_size should be integer
                uout_batch = jnp.split(uout_train[perm,:],iter_per_epoch)
                uout_batch[0][0].block_until_ready() ### Foo code
                part_time_1 = part_time_1 + (time.perf_counter()-timer_1)
                
                # Adam iters
                for ind, (uin, uout) in enumerate(zip(uin_batch,uout_batch)):
                    params['last'].block_until_ready() ### Foo code 
                    timer_2 = time.perf_counter()
                    params, opt_state = step_GD(params, optimizer, opt_state, uin, uout, xy, weights)
                    params['last'].block_until_ready() ### Foo code 
                    part_time_2 = part_time_2 + (time.perf_counter()-timer_2)
        
        # LS step 
        if LS_num > 0 and wu >= delay_init:
            if wu == delay_init: # change Adam to LSAdam       
                lr = opt_state[0]['adam'][0].hyperparams['learning_rate']
                b1 = opt_state[0]['adam'][0].hyperparams['b1']
                b2 = opt_state[0]['adam'][0].hyperparams['b2']
                optimizer = optax.multi_transform({'adam': optax.inject_hyperparams(optax.adam)(lr,b1=b1,b2=b2), 'zero': optax.set_to_zero()},
                    {'branch':'adam', 'trunk':'adam', 'last':'zero'})
                opt_state = optimizer.init(params)
                iter_per_epoch = N_train//batch_new
        
            params['last'].block_until_ready() ### Foo code 
            timer_3 = time.perf_counter()
            params = step_LS(params, uin_train, uout_train, xy, weights)
            params['last'].block_until_ready() ### Foo code 
            part_time_3 = time.perf_counter()-timer_3
            
        # Time computation
        part_time = part_time_1 + part_time_2 + part_time_3
        total_time = total_time + part_time
        WU_time.append(total_time)
            
        # Forward    
        losses = loss_comps(params, uin_train, uout_train, xy, weights)                    
        loss_logs['loss_WU'].append(losses)
        loss_logs['weight_WU'].append(weights)
        # if wu < 10 or (wu+1) % 10 == 0: # compute initials and every 10 wu (This conputation is time consuming)
        #     condB, condT = cond_numbs_LS(params, uin_train, uout_train, xy)
        #     loss_logs['conds'].append([condB, condT])
        
        # Validation
        losses_val = loss_comps(params, uin_val, uout_val, xy, weights)
        loss_logs['loss_WU_val'].append(losses_val)

        if (wu+1) % disp_count == 0 or wu+1 == nIter:
            # Print losses
            logger1 = (f'Work Unit {wu+1:d} :\t Train total loss :\t\t{losses[0]:.6e}\tL2 loss :\t{losses[1]:.6e}\t'
                       f'Regularization loss :\t{losses[2]:.6e}\tRelative L2 Error :\t{losses[3]:.6e}')
            logger2 = (f'Work Unit {wu+1:d} :\t Validation total loss :\t{losses_val[0]:.6e}\tL2 loss :\t{losses_val[1]:.6e}\t'
                       f'Regularization loss :\t{losses_val[2]:.6e}\tRelative L2 Error :\t{losses_val[3]:.6e}')
            logger3 = (f'Work Unit {wu+1:d} :\t Training time: \t{total_time:0.2f} seconds') 
            with open(model_dir + '/result.txt', 'a') as f:
                print(logger1)
                print(logger2)
                print(logger3)
                f.write(logger1 + '\n')  
                f.write(logger2 + '\n') 
                f.write(logger3 + '\n')  
            
        if (wu+1) in savewu or wu+1 == nIter:
            # loss plot
            loss_WU_val = jnp.array(loss_logs['loss_WU'])
            weight_WU_val = jnp.array(loss_logs['weight_WU'])
            loss_WU_val_val = jnp.array(loss_logs['loss_WU_val'])
            cond_WU_val = jnp.array(loss_logs['conds'])
            colors = ['red','green','black'] 
            legend_unsqueeze = ['L2 Loss','Regularization Loss','Rel. L2 error']
            legend_weights = [r'λ_{data}',r'λ_{LL Regul}']
            title = f'Loss'
            labels_WU = {
                'colors':colors,'legend':legend_unsqueeze,'title':title,'xlabel':'Work Unit','ylabel':'Loss',
                'save_dir':model_dir+'/losses/loss_curve_WU.png'}
            labels_weights = {
                'colors':colors, 'legend':legend_weights,'title':'Weights','xlabel':'Work Unit','ylabel':'weight',
                'save_dir':model_dir+'/losses/weight_WU.png'}
            labels_WU_val = {
                'colors':colors,'legend':legend_unsqueeze,'title':'Validation '+title,'xlabel':'Work Unit','ylabel':'Loss',
                'save_dir':model_dir+'/losses/loss_curve_WU_val.png'}
            labels_cond_WU = {
                'colors':['red','blue'], 'legend':[r'B^{T}B',r'T^{T}T'],'title':'Condition number of LS matrices','xlabel':'Work Unit / 10','ylabel':'Cond num',
                'save_dir':model_dir+'/losses/conds_WU.png'}
            
            loss_plot(loss_WU_val[:,1:], labels_WU, logplot=True)
            loss_plot(weight_WU_val[:,:], labels_weights, logplot=True)
            loss_plot(loss_WU_val_val[:,1:], labels_WU_val, logplot=True)
            # loss_plot(cond_WU_val[:,:], labels_cond_WU, logplot=True) ## Condition number plot

            jnp.save(model_dir+'/losses/weight',weight_WU_val)
            jnp.save(model_dir+'/losses/training_loss',loss_WU_val)
            jnp.save(model_dir+'/losses/training_loss_val',loss_WU_val_val)
            jnp.save(model_dir+'/losses/training_time',jnp.array(WU_time))
            # jnp.save(model_dir+'/losses/conds',cond_WU_val) ## Condition number save

        if wu == 0:
            min_loss = losses[0]
            min_loss_val = losses_val[-1]
            
        if wu > 0 and losses[0] < min_loss:
            min_loss = losses[0]
            if nIter < 20 or wu+1 > 20-1:
                # Model save if best train loss
                wustr = str(min(i for i in savewu if i >= wu+1))
                model_save(data=params, path=model_dir+'/models/model_save_besttrain_'+wustr+'.pickle',overwrite=True)
                
        if wu > 0 and losses_val[-1] < min_loss_val:
            min_loss_val = losses_val[-1]
            if nIter < 20 or wu+1 > 20-1:
                # Model save if best validation accuracy 
                wustr = str(min(i for i in savewu if i >= wu+1))
                model_save(data=params, path=model_dir+'/models/model_save_bestval_'+wustr+'.pickle',overwrite=True)

    full_time = time.time() - start_time
    logger1 = (f'Training done:\t Work Unit {wu+1:d} :\t Training time: {full_time:0.2f} seconds')
    with open(model_dir + '/result.txt', 'a') as f:
        print(logger1)
        f.write(logger1 + '\n')
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
    # return jnp.array([loss, weights[0]*loss_l2_value, loss_reguls_value, l2rel]) # When regul_value itself is need (for lambda = 0)
## Networks
@jit
def operator_net(params, u, xy):
    B = branch_model.apply(params['branch'], u) 
    T = trunk_model.apply(params['trunk'], xy) 
    W = params['last']
    outputs = B @ W @ T.T
    return outputs
## LS solve
@jit
def construct_LS(params, u_in, u_out, xy):
    B = branch_model.apply(params['branch'], u_in) # P by J ##
    T = trunk_model.apply(params['trunk'], xy) # Q by I ##
    F = (jnp.transpose(u_out,(0,2,1))).reshape((u_out.shape[0], -1)) # P by Q
    return B, T, F
@jit
def solve_LS(params, u_in, u_out, xy, weights):
    B, T, F = construct_LS(params, u_in, u_out, xy)
    
    numP = jnp.shape(B)[0]
    numQ = jnp.shape(T)[0]
    
    # LS scale is magnified by PQ times from the loss scale
    lamb0 = weights[1]/weights[0]
    LSSlamb_regul = lamb0 * numP * numQ # lamb LL regul
 
    E = B.T @ (F@T) # RHS
    
    # SVD of Gram matrices
    _, dB, VBT = jnp.linalg.svd(B.T @ B, hermitian=True)
    _, dT, VTT = jnp.linalg.svd(T.T @ T, hermitian=True)
    
    E_tilde = VBT @ E @ VTT.T
    h_coeff = jnp.outer(dB,dT) + LSSlamb_regul*jnp.ones_like(jnp.outer(dB,dT))
    Y = jnp.reciprocal(h_coeff) * E_tilde
    
    C = VBT.T @ Y @ VTT 
    
    return C
# Condition number computation of BTB and TTT
@jit
def cond_numbs_LS(params, u_in, u_out, xy):
    B, T, F = construct_LS(params, u_in, u_out, xy)
        
    # optional cond number
    condB = jnp.linalg.cond(B.T @ B)
    condT = jnp.linalg.cond(T.T @ T)
    
    return condB, condT
## LS/GD steps
@partial(jit, static_argnums=(1,))
def step_GD(params, optimizer, opt_state, u_in, u_out, xy, weights):
    grads = grad(loss)(params, u_in, u_out, xy, weights)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state
@jit
def step_LS(params, u_in, u_out, xy, weights):
    param_last = solve_LS(params, u_in, u_out, xy, weights)
    params['last'] = param_last
    return params          

# for weights_init in weights_inits:
# Initialize 
temp_batch = 100
if keyword == 'Advection_IBC':
    u_in_foo = jnp.zeros((temp_batch,Nx+Ny+1)) 
    dataname = 'Advection_PQ_data'
    in_dims = 1
    model_settings = [([100]*2,activation,1,False),([100]*3,activation,1,False)]
    branch_model = network(*model_settings[0])
    trunk_model = network(*model_settings[1])
elif keyword == 'DiffReact':
    u_in_foo = jnp.zeros((temp_batch,Nx+1))
    dataname = 'ADR_f_data'
    in_dims = 1
    model_settings = [([100]*2,activation,1,False),([100]*3,activation,1,False)]
    branch_model = network(*model_settings[0])
    trunk_model = network(*model_settings[1])
elif keyword == 'Poisson_kappa':
    u_in_foo = jnp.zeros((temp_batch,Nx,Ny,1))
    dataname = 'Poisson_kappa_data'
    in_dims = 2
    model_settings = [([[16,(2,2),(2,2),'VALID'],[32,(2,2),(2,2),'VALID'],[64,(2,2),(2,2),'VALID']],
                    [150],activation,1,False),([150]*3,activation,1,False)]
    if network == res_MLP:
        branch_model = CNN_res_MLP(*model_settings[0])
    else:
        branch_model = CNN_MLP(*model_settings[0])
    trunk_model = network(*model_settings[1])
elif keyword == 'Poisson_g':
    u_in_foo = jnp.zeros((temp_batch,2*Nx+2*Ny+1))
    dataname = 'Poisson_g_data'
    in_dims = 1
    model_settings = [([150]*2,activation,1,False),([150]*3,activation,1,False)]
    branch_model = network(*model_settings[0])
    trunk_model = network(*model_settings[1])
m = (Nx+1)*(Ny+1)  # number of trunk input sensors
xy_in_foo = jnp.zeros((m,2))

# Data load & generation
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

# flags
if activation == silu:
    actflag = 'SS'
elif activation == relu:
    actflag = 'RR'
if network == res_MLP:
    netflag = 'res'   
else:
    netflag = ''
if initialization == apply_he:
    initflag = 'HH'
elif initialization == apply_box:
    initflag = 'BB'
weights_init_post = weights_init[:]
adam_num_post = adam_num
# Init + Optimizer
key = random.key(seedA)
key, *keys = random.split(key,4)
branch_params = branch_model.init(keys[0], u_in_foo)
trunk_params = trunk_model.init(keys[1], xy_in_foo)
last_params = he_normal()(keys[2],(model_settings[0][-4][-1],model_settings[1][0][-1]))

key, *keys = random.split(key,3)
branch_params = initialization(branch_params, model_settings, 0, keys[0], gmode='N')
trunk_params = initialization(trunk_params, model_settings, 1, keys[1], gmode='N')
params = {'branch': branch_params, 'trunk': trunk_params, 'last': last_params}
optimizer = optax.multi_transform({'adam': optax.inject_hyperparams(optax.adam)(lr,b1=b1,b2=b2), 'zero': optax.set_to_zero()},
            {'branch':'adam', 'trunk':'adam', 'last':'adam'}) 
opt_state = optimizer.init(params)

# Loggers
loss_WU = []
weight_WU = []
loss_WU_val = []
conds = []
loss_logs = {'loss_WU':loss_WU,'weight_WU':weight_WU,'loss_WU_val':loss_WU_val,'conds':conds}

# Adam vs LS+Adam
reg_lambda = '{:.0e}'.format(weights_init[1])
reg_decay = '{:.0e}'.format(decay_target)
if LS_num > 0: # LS+Adam
    total_WU = int(LSAdam_const * batch_LSAdam // adam_num_post) # Total work units
    delay_init = delay_init_pre // adam_num
    if delay_init_pre == 0:
        model_dir = folder + keyword + '/' + subfolder + keyword + '_LSAdam_R' + str(adam_num) + \
            '_' + str(delay_init_pre) + '_' + initflag + '_' + actflag + '_bat' + \
            str(batch_LSAdam) + '_' + netflag + model_desc + '_seed_' + str(MasterKey) + \
            '_regul_' + reg_lambda 
    else:
        model_dir = folder + keyword + '/' + subfolder + keyword + '_LSAdam_R' + str(adam_num) + \
            '_' + str(delay_init_pre) + '_' + initflag + '_' + actflag + '_bat' + str(batch_Adam) + 'to' + \
            str(batch_LSAdam) + '_' + netflag + model_desc + '_seed_' + str(MasterKey) + \
            '_regul_' + reg_lambda 
    if weight_decay == True:
        model_dir = model_dir + '_decay_' + reg_decay + '_at_' + str(decay_end)
        
else: # Adam
    adam_num_post = 1 # Always epochwise
    weights_init_post[1] = 0 # No last layer regularization weight
    total_WU = int(Adam_const * batch_Adam // adam_num_post) # Total work units
    delay_init = 0
    model_dir = folder + keyword + '/' + subfolder + keyword + '_Adam_NoLS_' + \
        initflag + '_' + actflag + '_bat' + str(batch_Adam) + '_' + netflag + \
        model_desc + '_seed_' + str(MasterKey)        
createFolder(model_dir +'/models')
createFolder(model_dir +'/losses')
disp_count = int(0.01*total_WU) # tqdm progress display as new lines 

train(params=params, optimizer=optimizer, seed=seedB, delay_init=delay_init,
    uin_train=uin_train, uout_train=uout_train, uin_val=uin_val, uout_val=uout_val, xy=xy_full,
    weights_init=weights_init_post,
    batch_size=batch_Adam, batch_new=batch_LSAdam, adam_num=adam_num_post, LS_num=LS_num,
    weight_decay=weight_decay, decay_target=decay_target, decay_start=decay_start, decay_end=decay_end,
    loss_logs=loss_logs, model_dir=model_dir,
    nIter=total_WU, disp_count=disp_count)
