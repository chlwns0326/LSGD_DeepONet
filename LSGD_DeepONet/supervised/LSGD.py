# LSGD system formation and solve
import jax
import jax.numpy as jnp
from jax import jit
from networks import *
jax.config.update('jax_enable_x64', True)

# Construct LS system
@jit
def construct_LS(params, u_in, u_out, xy):
    B = branch_model.apply(params['branch'], u_in) # P by J ##
    T = trunk_model.apply(params['trunk'], xy) # Q by I ##
    F = (jnp.transpose(u_out,(0,2,1))).reshape((u_out.shape[0], -1)) # P by Q
    return B, T, F

# Solve LS system for the Last layer
@jit
def solve_LS(params, u_in, u_out, xy, weights):
    B, T, F = construct_LS(params, u_in, u_out, xy)
    
    numP = jnp.shape(B)[0]
    numQ = jnp.shape(T)[0]
    
    # LS scale is magnified by PQ times from the loss scale
    lamb0 = weights[1]/weights[0]
    LSSlamb_regul = lamb0 * numP * numQ # lamb LL regul
 
    E = B.T @ (F@T) # RHS
        
    # SVD of component matrices
    # _, dB, VBT = jnp.linalg.svd(B)
    # _, dT, VTT = jnp.linalg.svd(T)
    
    # E_tilde = VBT @ E @ VTT.T
    # h_coeff = jnp.outer(dB**2,dT**2) + LSSlamb_regul*jnp.ones_like(jnp.outer(dB,dT))
    # Y = jnp.reciprocal(h_coeff) * E_tilde
    
    # C = VBT.T @ Y @ VTT 
    
    # SVD of Gram matrices
    _, dB, VBT = jnp.linalg.svd(B.T @ B, hermitian=True)
    _, dT, VTT = jnp.linalg.svd(T.T @ T, hermitian=True)
    
    E_tilde = VBT @ E @ VTT.T
    h_coeff = jnp.outer(dB,dT) + LSSlamb_regul*jnp.ones_like(jnp.outer(dB,dT))
    Y = jnp.reciprocal(h_coeff) * E_tilde
    
    C = VBT.T @ Y @ VTT 
    
    return C