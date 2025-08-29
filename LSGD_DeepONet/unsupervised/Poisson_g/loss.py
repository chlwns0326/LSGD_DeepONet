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
    l2err = jnp.mean(jnp.sqrt(diff_sq)) ### Only for computing error
    l2rel = jnp.mean(jnp.sqrt(l2relsqs))
    return l2err, l2rel

# Define physics loss
@jit
def loss_phys(params, u_in, xy_phys):
    pred = residual_net(params, u_in, xy_phys)
    loss = jnp.mean((pred)**2) # denominator : P(#u) * Qp(#xy)
    return loss

# Define data loss
@jit
def loss_data(params, u_in, xy_data):
    ibc_pred = operator_net(params, u_in, xy_data)
    loss = jnp.mean((u_in[:,:-1] - ibc_pred)**2) # denominator : P(#u) * Qd(#xy)
    return loss     

# Define last layer regularization loss
@jit
def loss_reguls(params):
    loss = jnp.sum(params['last']**2)
    return loss

# Define total loss
@jit
def loss(params, u_in, xy_phys, xy_data, weights):
    loss_phys_value = loss_phys(params, u_in, xy_phys)
    loss_data_value = loss_data(params, u_in, xy_data)
    loss_reguls_value = loss_reguls(params)
    loss =  weights[0]*loss_phys_value + weights[1]*loss_data_value + weights[2]*loss_reguls_value
    return loss

# return each components
@jit
def loss_comps(params, u_in, u_out, xy, xy_phys, xy_data, weights):
    loss_phys_value = loss_phys(params, u_in, xy_phys)
    loss_data_value = loss_data(params, u_in, xy_data)
    loss_reguls_value = loss_reguls(params)
    loss = weights[0]*loss_phys_value + weights[1]*loss_data_value + weights[2]*loss_reguls_value
    l2err, l2rel = loss_l2(params, u_in, u_out, xy)
    return jnp.array([loss, weights[0]*loss_phys_value, weights[1]*loss_data_value, weights[2]*loss_reguls_value, l2err, l2rel])