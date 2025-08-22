# Losses
import jax
import jax.numpy as jnp
from jax import jit
from networks import *
jax.config.update('jax_enable_x64', True)

# Define l2 loss
@jit
def loss_l2(params, u_in, u_out, xy):
    u_pred = operator_net(params, u_in, xy)
    u_out_ = jnp.transpose(u_out,(0,2,1)).reshape((u_out.shape[0], -1))
    axis = tuple(range(1,u_out_.ndim))        
    diff_sq = jnp.mean((u_out_ - u_pred)**2,axis=axis)
    data_sq = jnp.mean((u_out_)**2,axis=axis)
    l2relsqs = diff_sq/data_sq
    l2err = jnp.mean(jnp.sqrt(diff_sq)) 
    l2rel = jnp.mean(jnp.sqrt(l2relsqs))
    return l2err, l2rel

# Define last layer regularization loss
@jit
def loss_reguls(params):
    loss = jnp.sum(params['last']**2)
    return loss

# Define total loss
@jit
def loss(params, u_in, u_out, xy, weights):
    loss_l2_value, l2rel = loss_l2(params, u_in, u_out, xy)
    loss_reguls_value = loss_reguls(params)
    loss = weights[0]*loss_l2_value + weights[1]*loss_reguls_value
    return loss

# return each components
@jit
def loss_comps(params, u_in, u_out, xy, weights):
    loss_l2_value, l2rel = loss_l2(params, u_in, u_out, xy)
    loss_reguls_value = loss_reguls(params)
    loss = weights[0]*loss_l2_value + weights[1]*loss_reguls_value
    return jnp.array([loss, weights[0]*loss_l2_value, weights[1]*loss_reguls_value, l2rel])