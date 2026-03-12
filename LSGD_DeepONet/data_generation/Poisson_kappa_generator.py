import os, warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #################################

# imports
import scipy.io as io
from scipy.stats import hmean
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from tqdm import tqdm
jax.config.update('jax_enable_x64', True)

def GRF_2D_Neumann(data_num, freq, Nx, Ny, gamma, tau, sigma, key):
    keys = random.split(key,3)
    # Eigenvalue generation
    ex,ey = np.meshgrid(np.arange(freq),np.arange(freq),indexing='xy')
    my_eigs = ((tau**(gamma-1)*np.abs(sigma))*(np.pi**2*(ex**2+ey**2) + tau**2)**(-gamma/2)).T
    # Random normal coeffs
    xi = random.normal(keys[0],(data_num,freq,freq))
    coeff = my_eigs*xi
    bases = np.zeros((freq,freq,Nx,Ny))
    x = np.linspace(0+1/(2*Nx),1-1/(2*Nx),Nx)
    y = np.linspace(0+1/(2*Ny),1-1/(2*Ny),Ny)
    xx,yy = np.meshgrid(x,y,indexing='xy')
    # Basis function generation
    for i in np.arange(freq):
        for j in np.arange(freq):
            bases[i,j,:,:] = np.cos((i)*np.pi*xx)*np.cos((j)*np.pi*yy)
    out_GRF = np.einsum('dpq,pqnm->dnm',coeff,bases)
    return out_GRF

def RBF_exp_quad_2D(xl, yl, var, lx, ly):
    diff1 = jnp.abs(xl[None,:]-xl[:,None])
    diff2 = jnp.abs(yl[None,:]-yl[:,None])
    return var * jnp.exp(-diff1**2/(2*lx**2)-diff2**2/(2*ly**2))

def gen_f_2D(key, x, y, var, lx, ly):
    # Generate subkeys
    subkeys = random.split(key, 2)
    Nx, Ny = jnp.shape(x)[0]-1, jnp.shape(y)[0]-1
    xx, yy = jnp.meshgrid(x,y,indexing='xy')
    xx_long = xx.transpose((1,0)).flatten()
    yy_long = yy.transpose((1,0)).flatten()
    # Generate a GP sample
    jitter = 1e-12
    K = RBF_exp_quad_2D(xx_long, yy_long, var, lx, ly)
    L = jnp.linalg.cholesky(K + jitter*jnp.eye((Nx+1)*(Ny+1)))
    pre_out = jnp.dot(L, random.normal(subkeys[0], ((Nx+1)*(Ny+1),)))
    out = pre_out.reshape((Ny+1,Nx+1)).transpose((1,0))   
    return out

# FDM
def solve_Poisson(kappa,BC,f):
    """Solve 1D
    -div(kappa*grad(u)) = f
    with zero Dirichlet BCs on unit square.
    """
    # Create grid
    Nx, Ny = np.shape(kappa)[0], np.shape(kappa)[1]
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    x = np.linspace(xmin, xmax, Nx+1)
    y = np.linspace(ymin, ymax, Ny+1)
    dx = x[1]-x[0]
    dy = y[1]-y[0]
            
    # Initialize solution and apply initial condition
    u = np.zeros((Nx+1, Ny+1))
    u[:-1,0] = (BC[0:Nx])
    u[-1,:-1] = (BC[Nx:Nx+Ny])
    u[Nx:0:-1,-1] = (BC[Nx+Ny:2*Nx+Ny])
    u[0,Ny:0:-1] = (BC[2*Nx+Ny:-1])
    
    matL = np.zeros(((Nx+1)*(Ny+1),(Nx+1)*(Ny+1)))
    matL_inds = np.zeros(((Nx-1)*(Ny-1),),dtype=np.int64)
    
    kappa_mid = (kappa[0:-1,0:-1]+kappa[0:-1,1:]+kappa[1:,0:-1]+kappa[1:,1:])/4
    kappa_x = ((kappa[1:,0:-1]+kappa[1:,1:])-(kappa[0:-1,0:-1]+kappa[0:-1,1:]))/(2*dx)
    kappa_y = ((kappa[0:-1,1:]+kappa[1:,1:])-(kappa[0:-1,0:-1]+kappa[1:,0:-1]))/(2*dy)

    for i in np.arange(start=1,stop=Nx):
        for j in np.arange(start=1,stop=Ny):
            indc = (Nx+1)*j+i
            ind_matL = (Nx-1)*(j-1)+(i-1)
            indl,indr,indd,indu = indc-1,indc+1,indc-(Nx+1),indc+(Nx+1)
            matL[indc,indc] = (4*Nx*Ny*kappa_mid[i-1,j-1])
            matL[indc,indl] = (Nx*kappa_x[i-1,j-1]/2-Nx*Ny*kappa_mid[i-1,j-1])
            matL[indc,indr] = (-Nx*kappa_x[i-1,j-1]/2-Nx*Ny*kappa_mid[i-1,j-1])
            matL[indc,indd] = (Ny*kappa_y[i-1,j-1]/2-Nx*Ny*kappa_mid[i-1,j-1])
            matL[indc,indu] = (-Ny*kappa_y[i-1,j-1]/2-Nx*Ny*kappa_mid[i-1,j-1])
            matL_inds[ind_matL] = indc
            
    vecb = f[1:-1,1:-1].transpose((1,0)).reshape(((Nx-1)*(Ny-1),1))
    flatsol = jnp.linalg.solve(matL[matL_inds,:][:,matL_inds],vecb)
    u[1:-1,1:-1] = (flatsol.reshape((Nx-1,Ny-1)).transpose((1,0)))
    
    return u

def main():
    # Directories and hyperparams
    data_dir = 'data/' 
    data_name = 'Poisson_kappa_data.mat' 

    N_train, N_test = 1000, 100
    Nx, Ny = 128, 128
    jx, jy = Nx//32, Ny//32
    xmin, xmax = 0, 1
    f = np.ones((Nx+1,Ny+1))
    BC = np.zeros((2*Nx+2*Ny+1,))
    lx, ly = 0.1, 0.1
    var = 1 
    scale = 0.2
    xk = np.linspace(xmin+1/(2*Nx), xmax-1/(2*Nx), Nx)
    yk = np.linspace(xmin+1/(2*Ny), xmax-1/(2*Ny), Ny)
       
    # train generation
    key = random.key(0)
    keys = random.split(key,N_train)
    input_train = np.exp(scale * jax.vmap(lambda k: gen_f_2D(k, xk, yk, var, lx, ly))(keys))
    output_train = np.zeros((N_train,Nx+1,Ny+1))
    for ind in tqdm(range(N_train)):
        output_train[ind,:,:] = (solve_Poisson(input_train[ind,:,:],BC,f))

    # val generation
    key = random.key(113355)
    keys = random.split(key,N_test)
    input_val = np.exp(scale * jax.vmap(lambda k: gen_f_2D(k, xk, yk, var, lx, ly))(keys))
    output_val = np.zeros((N_test,Nx+1,Ny+1))
    for ind in tqdm(range(N_test)):
        output_val[ind,:,:] = (solve_Poisson(input_val[ind,:,:],BC,f))
    
    # input homogenization as reciprocals
    input_train_pool = hmean(input_train.reshape(N_train, 32, jx, 32, jy),axis=(2, 4))
    input_val_pool = hmean(input_val.reshape(N_test, 32, jx, 32, jy),axis=(2, 4))
     
    mdic = {"input_train": input_train_pool, "output_train": output_train[:,::jx,::jy],
            "input_val": input_val_pool, "output_val": output_val[:,::jx,::jy]}
    io.savemat(data_dir+data_name,mdic)
    
if __name__ == '__main__':    
    main()