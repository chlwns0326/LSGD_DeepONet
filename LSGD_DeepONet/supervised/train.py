import sys, logging, warnings
warnings.filterwarnings('ignore')

# imports
import time, jax
import jax.numpy as jnp
from jax import random
jax.config.update('jax_enable_x64', True)
from loss import *
from LSGD import *
from step import *
from misc import *
from tqdm import tqdm

# Training stage
def train(params, optimizer, seed, delay_init, 
          uin_train, uout_train, uin_val, uout_val, xy,
          weights_init, 
          batch_size, batch_new, adam_num, LS_num,
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
        ## Uncomment below section to activate weight decay
        # decay_target = 1e-3 # b
        # decay_start = 100 # Start point of decay, constant before decay_start WU
        # decay_end = 1000 # End point of decay, stay constant afterward
        # if wu > decay_start and wu <= decay_end:
        #     rate = decay_target**(1/(decay_end-decay_start)) # Gradual decay for each WU
        #     p1 = weights[1] * rate
        #     weights = [weights_init[0],p1] 
        
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
            colors = ['red','green','black'] 
            legend_unsqueeze = ['L2 Loss','Regularization Loss','Rel. L2 error']
            legend_weights = [r'λ_{data}',r'λ_{LL Regul}']
            title = f'Loss'
            labels_WU = {
                'colors':colors,'legend':legend_unsqueeze,'title':title,'xlabel':'Work Unit','ylabel':'Loss',
                'save_dir':model_dir+'/losses/loss_curve_WU.png'}
            labels_WU_total = {
                'colors':colors, 'legend':['Total Loss'],'title':title,'xlabel':'Work Unit','ylabel':'Loss',
                'save_dir':model_dir+'/losses/loss_curve_WU_total.png'}
            labels_weights = {
                'colors':colors, 'legend':legend_weights,'title':'Weights','xlabel':'Work Unit','ylabel':'weight',
                'save_dir':model_dir+'/losses/weight_WU.png'}
            labels_WU_val = {
                'colors':colors,'legend':legend_unsqueeze,'title':'Validation '+title,'xlabel':'Work Unit','ylabel':'Loss',
                'save_dir':model_dir+'/losses/loss_curve_WU_val.png'}
            labels_WU_val_total = {
                'colors':colors, 'legend':['Total Loss'],'title':'Validation Total '+title,'xlabel':'Work Unit','ylabel':'Loss',
                'save_dir':model_dir+'/losses/loss_curve_WU_val_total.png'}
            
            loss_plot(loss_WU_val[:,1:], labels_WU, logplot=True)
            loss_plot(loss_WU_val[:,0:1], labels_WU_total, logplot=True)
            loss_plot(weight_WU_val[:,:], labels_weights, logplot=True)
            loss_plot(loss_WU_val_val[:,1:], labels_WU_val, logplot=True)
            loss_plot(loss_WU_val_val[:,0:1], labels_WU_val_total, logplot=True)
            
            jnp.save(model_dir+'/losses/weight',weight_WU_val)
            jnp.save(model_dir+'/losses/training_loss',loss_WU_val)
            jnp.save(model_dir+'/losses/training_loss_val',loss_WU_val_val)
            jnp.save(model_dir+'/losses/training_time',jnp.array(WU_time))
            
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
