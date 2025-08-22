# Network and computations
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from models import branch_model, trunk_model
jax.config.update('jax_enable_x64', True)

# derivative of trunk net w.r.t x,y
@partial(jit, static_argnums=(1,))
def trunk_only_net_deriv(trunk_params, xy):
    ufun = lambda xy0: trunk_model.apply(trunk_params, xy0) ##
    _ux = lambda xy0: jax.jvp(ufun,(xy0,),(jnp.array([1.0,0.0]),))[1] # ux
    _uy = lambda xy0: jax.jvp(ufun,(xy0,),(jnp.array([0.0,1.0]),))[1] # uy
    _ux_x = lambda xy0: jax.jvp(_ux,(xy0,),(jnp.array([1.0,0.0]),))[1] # (ux)x
    _uy_y = lambda xy0: jax.jvp(_uy,(xy0,),(jnp.array([0.0,1.0]),))[1] # (uy)y
    ux_x = vmap(_ux_x)
    uy_y = vmap(_uy_y)
    Tres = -ux_x(xy)-uy_y(xy)
    return Tres

# Define DeepONet architecture with Doubly batched input; u and (x,y) have totally individual batched input
@jit
def operator_net(params, u, xy):
    B = branch_model.apply(params['branch'], u) ##
    T = trunk_model.apply(params['trunk'], xy) ##
    W = params['last']
    outputs = B @ W @ T.T
    return outputs

# Batched version of deriv of separable DeepONet w.r.t the trunk input variables
@partial(jit, static_argnums=(1,))
def operator_net_deriv(params, u, xy):
    B = branch_model.apply(params['branch'], u) ##
    Tres = trunk_only_net_deriv(params['trunk'], xy)
    W = params['last']
    out_res = B @ W @ Tres.T
    return out_res

# Define PDE residual; batched version as P by Q matrix
@partial(jit, static_argnums=(1,))
def residual_net(params, u, xy):
    out = operator_net_deriv(params, u, xy)
    res = u[:,1:-1,1:-1,:].transpose(0,2,1,3).reshape((jnp.shape(u)[0],-1)) - out # F - LU; F = varies
    return res