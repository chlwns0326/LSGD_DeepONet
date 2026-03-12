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
    Fphys = u_in[:,1:-1,1:-1,0].transpose(0,2,1).reshape((jnp.shape(u_in)[0],-1)) # P by Q
    Tdata = trunk_model.apply(params['trunk'], xy_data) # Qb by I ##
    Fdata = jnp.zeros((jnp.shape(B)[0],jnp.shape(Tdata)[0])) # P by Qb # zero dirichlet
    return B, Tphys, Fphys, Tdata, Fdata

# Solve LS system for Last layer
@jit
def solve_LS(params, u_in, xy_phys, xy_data, weights):
    B, Tphys, Fphys, Tdata, Fdata = construct_LS(params, u_in, xy_phys, xy_data)
    
    numP = jnp.shape(B)[0]
    numQphys = jnp.shape(Tphys)[0]
    numQdata = jnp.shape(Tdata)[0]
    
    # LS scale is magnified by PQ times from the loss scale
    lamb0 = weights[1]/weights[0]
    lamb1 = weights[2]/weights[0]

    LSSlamb_data = lamb0 * numQphys / numQdata # lamb data
    LSSlamb_regul = lamb1 * numP * numQphys # lamb regul
    
    E = B.T @ (Fphys@Tphys + LSSlamb_data*Fdata@Tdata)
    
    # Normal matrix construction and SVD of gram matrix
    TTT = Tphys.T @ Tphys + LSSlamb_data*Tdata.T @ Tdata
    
    _, d1, V1T = jnp.linalg.svd(B.T @ B, hermitian=True)
    _, d2, V2T = jnp.linalg.svd(TTT, hermitian=True)
    
    E_tilde = V1T @ E @ V2T.T
    h_coeff = jnp.outer(d1,d2) + LSSlamb_regul*jnp.ones_like(jnp.outer(d1,d2))
    Y = jnp.reciprocal(h_coeff) * E_tilde

    C = V1T.T @ Y @ V2T
    
    return C