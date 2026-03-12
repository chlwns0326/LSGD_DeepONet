# optimizer step
import jax, optax
from jax import grad, jit
from functools import partial
from loss import *
from LSGD import *
jax.config.update('jax_enable_x64', True)

# Define a compiled update step for GD
@partial(jit, static_argnums=(1,))
def step_GD(params, optimizer, opt_state, u_in, u_out, xy, weights):
    grads = grad(loss)(params, u_in, u_out, xy, weights)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# Define a compiled update step for LS # Full batch
@jit
def step_LS(params, u_in, u_out, xy, weights):
    param_last = solve_LS(params, u_in, u_out, xy, weights)
    params['last'] = param_last
    return params