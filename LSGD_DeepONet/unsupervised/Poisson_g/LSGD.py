# LSGD system formation and solve
import jax
import jax.numpy as jnp
from jax import jit
from networks import *
jax.config.update('jax_enable_x64', True)

# Construct LS system
@jit
def construct_LS(params, u_in, xy_phys, xy_data):
    B = branch_model.apply(params['branch'], u_in) # P by J ##
    Tphys = trunk_only_net_deriv(params['trunk'], xy_phys) # Q by I
    Fphys = jnp.zeros((u_in.shape[0],xy_phys.shape[0])) # P by Q
    Tdata = trunk_model.apply(params['trunk'], xy_data) # Qb by I ##
    Fdata = u_in[:,:-1] # P by Qb 
    return B, Tphys, Fphys, Tdata, Fdata

# Solve LS system for Last layer
@jit
def solve_LS(params, u_in, xy_phys, xy_data, weights):
    B, Tphys, Fphys, Tdata, Fdata = construct_LS(params, u_in, xy_phys, xy_data)
    
    eps = 1e-14 # Ensure nonnegative singular values for numerical SVD. 
    numP = jnp.shape(B)[0]
    numQphys = jnp.shape(Tphys)[0]
    numQdata = jnp.shape(Tdata)[0]
    
    # LS scale is magnified by PQ times from the loss scale
    lamb0 = weights[1]/weights[0]
    lamb1 = weights[2]/weights[0]

    LSSlamb_data = lamb0 * numQphys / numQdata # lamb data
    LSSlamb_regul = lamb1 * numP * numQphys # lamb regul
    
    E = B.T @ (Fphys@Tphys + LSSlamb_data*Fdata@Tdata)
    
    # Normal matrix construction and EigDecomposition
    TTT = Tphys.T @ Tphys + LSSlamb_data*Tdata.T @ Tdata
    BTB = B.T @ B
    
    Q1, d1, Q1T = jnp.linalg.svd(BTB + eps*jnp.eye(jnp.shape(B)[1]), hermitian=True)
    Q2, d2, Q2T = jnp.linalg.svd(TTT + eps*jnp.eye(jnp.shape(Tphys)[1]), hermitian=True)
    
    E_tilde = Q1.T @ E @ Q2
    h_coeff = jnp.outer(d1,d2) + LSSlamb_regul*jnp.ones_like(jnp.outer(d1,d2))
    Y = jnp.reciprocal(h_coeff) * E_tilde

    C = Q1 @ Y @ Q2.T
    
    return C