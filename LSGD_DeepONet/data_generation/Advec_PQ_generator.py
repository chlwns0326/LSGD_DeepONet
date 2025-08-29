import os, warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #################################

# imports
import scipy.io as io
import jax
import jax.numpy as jnp
from jax import random
jax.config.update('jax_enable_x64', True)

def RBF(x1, x2, var, l):
    # Original source from https://zenodo.org/records/5206676
    diff = jnp.abs(x1[None,:]-x2[:,None])
    return var * jnp.exp(-diff**2/(2*l**2))

def gen_f(key, NX, a, length_scale):
    # Original source from https://zenodo.org/records/5206676
    subkeys = random.split(key, 2)
    # Generate a GP sample
    xmin, xmax = 0, 1
    jitter = 1e-12
    X = jnp.linspace(xmin, xmax, NX+1)
    AX = jnp.linspace(-a*xmax, xmax, int(a*NX+NX+1))
    K = RBF(AX, AX, 1, length_scale)
    L = jnp.linalg.cholesky(K + jitter*jnp.eye(int(a*NX+NX+1)))
    gp_sample = jnp.dot(L, random.normal(subkeys[0], (int(a*NX+NX+1),)))
    # Create a callable interpolation function  
    f = lambda x: jnp.interp(x, AX.flatten(), gp_sample)    
    return f

# Constant coefficient Advection solver
def solve_advec_const(Nx, Nt, a, pq):
    """Solve 1D
    u_t + au_x = 0
    a(x)>0
    with given initial(t=0) and boundary(x=0) conditions.
    """
    # Create grid
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    x = jnp.linspace(xmin, xmax, Nx+1)
    t = jnp.linspace(tmin, tmax, Nt+1)
    xx,tt = jnp.meshgrid(x,t,indexing='ij')
    # Exact solution generation
    pq_ = lambda x: jnp.interp(x, jnp.linspace(-a*xmax, xmax, jnp.shape(pq)[0]), pq)
    UU = pq_(xx-a*tt)
    return UU

def elongate(a, vec):
    xmin, xmax = 0, 1
    ax = jnp.linspace(-a*xmax, 0, jnp.shape(vec)[0])
    ax2 = jnp.linspace(-a*xmax, 0, 2*jnp.shape(vec)[0]-1)
    U = jnp.interp(ax2, ax, vec)
    return U

# Directories and hyperparams
data_dir = 'data/'
data_name = 'Advection_PQ_data.mat' 

N_train, N_test = 1000, 100
length_scale = 0.2
Nx, Nt = 128, 128
jx, jt = Nx//32, Nt//32
a = 0.5
q = lambda t: jnp.zeros_like(t)
xmin, xmax = 0, 1
x = jnp.linspace(xmin, xmax, Nx+1)
ax = jnp.linspace(-a*xmax, xmax, int(a*Nx+Nx+1))

# train generation
key = random.key(0)
keys = random.split(key,N_train)
pre_inputs = jax.vmap(lambda k: gen_f(k, Nx, a, length_scale)(ax))(keys)
output_train = jax.vmap(lambda PQ: solve_advec_const(Nx, Nt, a, PQ))(pre_inputs)
Qpart = jax.vmap(lambda pq: elongate(a,pq))(pre_inputs[:,:int(a*Nx+1)]) # vmap a*Nx+1 -> Nx+1
Ppart = pre_inputs[:,int(a*Nx+1):] 
input_train = jnp.concatenate((Qpart,Ppart),axis=1)

# val generation
key = random.key(113355)
keys = random.split(key,N_test)
pre_inputs = jax.vmap(lambda k: gen_f(k, Nx, a, length_scale)(ax))(keys)
output_val = jax.vmap(lambda PQ: solve_advec_const(Nx, Nt, a, PQ))(pre_inputs)
Qpart = jax.vmap(lambda pq: elongate(a,pq))(pre_inputs[:,:int(a*Nx+1)]) # vmap a*Nx+1 -> Nx+1
Ppart = pre_inputs[:,int(a*Nx+1):] 
input_val = jnp.concatenate((Qpart,Ppart),axis=1)

mdic = {"input_train": input_train[:,::jx], "output_train": output_train[:,::jx,::jt],
        "input_val": input_val[:,::jx], "output_val": output_val[:,::jx,::jt]}
io.savemat(data_dir+data_name,mdic)