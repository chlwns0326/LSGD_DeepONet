# Network and computations
import jax
from jax import jit
from models import branch_model, trunk_model
jax.config.update('jax_enable_x64', True)

# Define DeepONet architecture with Doubly batched input; u and (x,y) have totally individual batched input
@jit
def operator_net(params, u, xy):
    B = branch_model.apply(params['branch'], u) 
    T = trunk_model.apply(params['trunk'], xy) 
    W = params['last']
    outputs = B @ W @ T.T
    return outputs