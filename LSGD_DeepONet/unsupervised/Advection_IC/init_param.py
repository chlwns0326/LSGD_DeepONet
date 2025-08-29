import warnings
warnings.filterwarnings('ignore')

# imports
import jax
import jax.numpy as jnp
from jax import random
jax.config.update('jax_enable_x64', True)
from initialization import MLP_init_box, res_MLP_init_box, CNN_init, prop_orth_init

# CNN initialization uses default init for flax
# All related sections are commented out

# Box init.
# input: (param_tree, model_settings, net_index, key, args...)
def apply_box(param_tree, model_settings, netind, key, from_r=0.5,to_r=0.5,zerofix=False,from_r_init=0.5,zerofix_init=False,
              gmode='N',scale_W=1,scale_b=0,cnn_first_rescale=1):
    params = param_tree['params']
    for layer in params:
        activation = model_settings[netind][-3]
        shape = params[layer]['kernel'].shape
        curr_L, max_L = int(layer[-1])+1, len(params)
        if 'Dense_Layer_with_Skip_Conn_' in layer:
            if curr_L == 1:
                W,b = res_MLP_init_box(key, shape, activation, curr_L, max_L, from_r_init, to_r, zerofix_init)
            else:
                W,b = res_MLP_init_box(key, shape, activation, curr_L, max_L, from_r, to_r, zerofix)
            params[layer]['kernel'] = W
            params[layer]['bias'] = b
        elif 'Dense_Layer_' in layer:
            if curr_L == 1:
                W,b = MLP_init_box(key, shape, activation, from_r_init, to_r, zerofix_init)
            else:
                W,b = MLP_init_box(key, shape, activation, from_r, to_r, zerofix)
            params[layer]['kernel'] = W
            params[layer]['bias'] = b
        # rescale 1st layer param
        elif 'Conv_Layer_' in layer:
            i = int(layer[-1])
            if i == 0:
                params[layer]['kernel'] = cnn_first_rescale * params[layer]['kernel']
        # elif 'Conv_Layer_' in layer:
        #     key1, key2 = random.split(key,2)
        #     i = int(layer[-1])
        #     if i == 0:
        #         shape_cnn = (model_settings[netind][0][i][1][0],model_settings[netind][0][i][1][1],1,model_settings[netind][0][i][0])
        #         # kH,kW,Cin,Cout
        #     else:
        #         shape_cnn = (model_settings[netind][0][i][1][0],model_settings[netind][0][i][1][1],model_settings[netind][0][i-1][0],model_settings[netind][0][i][0])
        #         # kH,kW,Cin,Cout
        #     d_in = shape_cnn[0]*shape_cnn[1]*shape_cnn[-2]
        #     d_out = shape_cnn[-1]
        #     if gmode == 'U':
        #         he_factor = jnp.sqrt(6/(d_in)) 
        #         W = scale_W * he_factor * random.uniform(key1, shape_cnn, minval=-1)
        #         b = scale_b * he_factor * random.uniform(key2, (d_out,), minval=-1)
        #     elif gmode == 'N':
        #         he_factor = jnp.sqrt(2/(d_in)) # Uniform 6 Normal 2 
        #         W = scale_W * he_factor * random.normal(key1, shape_cnn)
        #         b = scale_b * he_factor * random.normal(key2, (d_out,))
        #     params[layer]['kernel'] = W
        #     params[layer]['bias'] = b
    param_tree['params'] = params
    return param_tree

# Glorot(Xavier) init.       
def apply_glorot(param_tree, model_settings, netind, key, gmode='N',scale_W=1,scale_b=0,cnn_first_rescale=1):
    params = param_tree['params']
    for layer in params:
        key1, key2 = random.split(key,2)
        shape = params[layer]['kernel'].shape
        if 'Dense_Layer_with_Skip_Conn_' in layer:
            d_in, d_out = shape
            if gmode == 'U':
                glorot_factor = jnp.sqrt(6/(d_in+d_out)) 
                W = scale_W * glorot_factor * random.uniform(key1, (d_in, d_out), minval=-1)
                b = scale_b * glorot_factor * random.uniform(key2, (d_out,), minval=-1)
            elif gmode == 'N':
                glorot_factor = jnp.sqrt(2/(d_in+d_out)) # Uniform 6 Normal 2 
                W = scale_W * glorot_factor * random.normal(key1, (d_in, d_out))
                b = scale_b * glorot_factor * random.normal(key2, (d_out,))
            params[layer]['kernel'] = W
            params[layer]['bias'] = b
        elif 'Dense_Layer_' in layer:
            d_in, d_out = shape
            if gmode == 'U':
                glorot_factor = jnp.sqrt(6/(d_in+d_out)) 
                W = scale_W * glorot_factor * random.uniform(key1, (d_in, d_out), minval=-1)
                b = scale_b * glorot_factor * random.uniform(key2, (d_out,), minval=-1)
            elif gmode == 'N':
                glorot_factor = jnp.sqrt(2/(d_in+d_out)) # Uniform 6 Normal 2 
                W = scale_W * glorot_factor * random.normal(key1, (d_in, d_out))
                b = scale_b * glorot_factor * random.normal(key2, (d_out,))
            params[layer]['kernel'] = W
            params[layer]['bias'] = b
        # rescale 1st layer param
        elif 'Conv_Layer_' in layer:
            i = int(layer[-1])
            if i == 0:
                params[layer]['kernel'] = cnn_first_rescale * params[layer]['kernel']
        # elif 'Conv_Layer_' in layer:
        #     i = int(layer[-1])
        #     if i == 0:
        #         shape_cnn = (model_settings[netind][0][i][1][0],model_settings[netind][0][i][1][1],1,model_settings[netind][0][i][0])
        #         # kH,kW,Cin,Cout
        #     else:
        #         shape_cnn = (model_settings[netind][0][i][1][0],model_settings[netind][0][i][1][1],model_settings[netind][0][i-1][0],model_settings[netind][0][i][0])
        #         # kH,kW,Cin,Cout
        #     d_in = shape_cnn[0]*shape_cnn[1]*shape_cnn[-2]
        #     d_out = shape_cnn[-1]
        #     if gmode == 'U':
        #         glorot_factor = jnp.sqrt(6/(d_in+d_out)) 
        #         W = scale_W * glorot_factor * random.uniform(key1, shape_cnn, minval=-1)
        #         b = scale_b * glorot_factor * random.uniform(key2, (d_out,), minval=-1)
        #     elif gmode == 'N':
        #         glorot_factor = jnp.sqrt(2/(d_in+d_out)) # Uniform 6 Normal 2 
        #         W = scale_W * glorot_factor * random.normal(key1, shape_cnn)
        #         b = scale_b * glorot_factor * random.normal(key2, (d_out,))
        #     params[layer]['kernel'] = W
        #     params[layer]['bias'] = b
    param_tree['params'] = params
    return param_tree  

# Orthogonal init.        
def apply_orth_init(param_tree, model_settings, netind, key, eps_orth=0.1):
    params = param_tree['params']
    for layer in params:
        shape = params[layer]['kernel'].shape
        range_cnn = 0.1 # 1
        if 'Dense_Layer_' in layer:
            W,b = prop_orth_init(shape,eps_orth)
            params[layer]['kernel'] = W
            params[layer]['bias'] = b
        # elif 'Conv_Layer_' in layer:
        #     i = int(layer[-1])
        #     if i == 0:
        #         shape_cnn = (model_settings[netind][0][i][1][0],model_settings[netind][0][i][1][1],1,model_settings[netind][0][i][0])
        #         # kH,kW,Cin,Cout
        #     else:
        #         shape_cnn = (model_settings[netind][0][i][1][0],model_settings[netind][0][i][1][1],model_settings[netind][0][i-1][0],model_settings[netind][0][i][0])
        #         # kH,kW,Cin,Cout
        #     W,b = CNN_init(key,shape_cnn,range_cnn)
        #     params[layer]['kernel'] = W
        #     params[layer]['bias'] = b
        
    param_tree['params'] = params
    return param_tree

def apply_he(param_tree, model_settings, netind, key, gmode='N',scale_W=1,scale_b=0,cnn_first_rescale=1):
    params = param_tree['params']
    for layer in params:
        key1, key2 = random.split(key,2)
        shape = params[layer]['kernel'].shape
        if 'Dense_Layer_with_Skip_Conn_' in layer:
            d_in, d_out = shape
            if gmode == 'U':
                he_factor = jnp.sqrt(6/(d_in)) 
                W = scale_W * he_factor * random.uniform(key1, (d_in, d_out), minval=-1)
                b = scale_b * he_factor * random.uniform(key2, (d_out,), minval=-1)
            elif gmode == 'N':
                he_factor = jnp.sqrt(2/(d_in)) # Uniform 6 Normal 2 
                W = scale_W * he_factor * random.normal(key1, (d_in, d_out))
                b = scale_b * he_factor * random.normal(key2, (d_out,))
            params[layer]['kernel'] = W
            params[layer]['bias'] = b
        elif 'Dense_Layer_' in layer:
            d_in, d_out = shape
            if gmode == 'U':
                he_factor = jnp.sqrt(6/(d_in)) 
                W = scale_W * he_factor * random.uniform(key1, (d_in, d_out), minval=-1)
                b = scale_b * he_factor * random.uniform(key2, (d_out,), minval=-1)
            elif gmode == 'N':
                he_factor = jnp.sqrt(2/(d_in)) # Uniform 6 Normal 2 
                W = scale_W * he_factor * random.normal(key1, (d_in, d_out))
                b = scale_b * he_factor * random.normal(key2, (d_out,))
            params[layer]['kernel'] = W
            params[layer]['bias'] = b
        # rescale 1st layer param
        elif 'Conv_Layer_' in layer:
            i = int(layer[-1])
            if i == 0:
                params[layer]['kernel'] = cnn_first_rescale * params[layer]['kernel']
        # elif 'Conv_Layer_' in layer:
        #     i = int(layer[-1])
        #     if i == 0:
        #         shape_cnn = (model_settings[netind][0][i][1][0],model_settings[netind][0][i][1][1],1,model_settings[netind][0][i][0])
        #         # kH,kW,Cin,Cout
        #     else:
        #         shape_cnn = (model_settings[netind][0][i][1][0],model_settings[netind][0][i][1][1],model_settings[netind][0][i-1][0],model_settings[netind][0][i][0])
        #         # kH,kW,Cin,Cout
        #     d_in = shape_cnn[0]*shape_cnn[1]*shape_cnn[-2]
        #     d_out = shape_cnn[-1]
        #     if gmode == 'U':
        #         he_factor = jnp.sqrt(6/(d_in)) 
        #         W = scale_W * he_factor * random.uniform(key1, shape_cnn, minval=-1)
        #         b = scale_b * he_factor * random.uniform(key2, (d_out,), minval=-1)
        #     elif gmode == 'N':
        #         he_factor = jnp.sqrt(2/(d_in)) # Uniform 6 Normal 2 
        #         W = scale_W * he_factor * random.normal(key1, shape_cnn)
        #         b = scale_b * he_factor * random.normal(key2, (d_out,))
        #     params[layer]['kernel'] = W
        #     params[layer]['bias'] = b
    param_tree['params'] = params
    return param_tree  