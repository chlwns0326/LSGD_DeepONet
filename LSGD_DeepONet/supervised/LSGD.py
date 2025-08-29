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
    
    eps = 1e-14 # Ensure nonnegative singular values for numerical SVD. 
    numP = jnp.shape(B)[0]
    numQ = jnp.shape(T)[0]
    
    # LS scale is magnified by PQ times from the loss scale
    lamb0 = weights[1]/weights[0]
    LSSlamb_regul = lamb0 * numP * numQ # lamb LL regul
 
    E = B.T @ (F@T) # RHS
    
    # Normal matrix construction and EigDecomposition
    TTT = T.T @ T
    BTB = B.T @ B
    
    Q1, d1, Q1T = jnp.linalg.svd(BTB + eps*jnp.eye(jnp.shape(B)[1]), hermitian=True)
    Q2, d2, Q2T = jnp.linalg.svd(TTT + eps*jnp.eye(jnp.shape(T)[1]), hermitian=True)
    
    E_tilde = Q1.T @ E @ Q2
    h_coeff = jnp.outer(d1,d2) + LSSlamb_regul*jnp.ones_like(jnp.outer(d1,d2))
    Y = jnp.reciprocal(h_coeff) * E_tilde

    C = Q1 @ Y @ Q2.T 
    
    return C