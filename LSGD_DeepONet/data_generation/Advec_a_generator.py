import os, warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #################################

# imports
import scipy.io as io
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, random
jax.config.update('jax_enable_x64', True)

def RBF(x1, x2, var, l):
    # Original source from https://zenodo.org/records/5206676
    diff = jnp.abs(x1[None,:]-x2[:,None])
    return var * jnp.exp(-diff**2/(2*l**2))

def gen_f(key, NX, length_scale):
    # Original source from https://zenodo.org/records/5206676
    subkeys = random.split(key, 2)
    # Generate a GP sample
    xmin, xmax = 0, 1
    jitter = 1e-12
    X = jnp.linspace(xmin, xmax, NX+1)
    K = RBF(X, X, 1, length_scale)
    L = jnp.linalg.cholesky(K + jitter*jnp.eye(NX+1))
    gp_sample = jnp.dot(L, random.normal(subkeys[0], (NX+1,)))
    f = lambda x: jnp.interp(x, X.flatten(), gp_sample)    
    return f

# Advection solver with Lax-Wendroff scheme
def solve_advec_LW(Nx, Nt, a, p, q):
    """Solve 1D
    u_t + a(x)u_x = 0
    a(x)>0
    with given initial(t=0) and boundary(x=0) conditions.
    """
    # Create grid
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    x = jnp.linspace(xmin, xmax, Nx+1)
    t = jnp.linspace(tmin, tmax, Nt+1)
    dx = x[1]-x[0]
    dt = t[1]-t[0]

    # Compute coefficients and IBC
    a_ = jnp.interp(x, jnp.linspace(xmin, xmax, jnp.shape(a)[0]), a) # 0~Nx
    p_ = p(x) #0~Nx
    q_ = q(t) #0~Nt
    ax = (a_[2:]-a_[:-2])/(2*dx) #1~Nx-1

    # Initialize solution and apply IBC
    U = jnp.zeros((Nx+1, Nt+1))
    U = U.at[:,0].set(p_)
    U = U.at[0,:].set(q_)
    
    # Time-stepping update
    def body_fn(i,U):
        u_curr = U[:,i] #0~Nx
        # deriv of u_n
        ux = (u_curr[2:]-u_curr[:-2])/(2*dx) #1~Nx-1
        # 2nd deriv of u_n
        uxx = (u_curr[2:]-2*u_curr[1:-1]+u_curr[:-2])/(dx**2) #1~Nx-1
        # Next timestep inside domain
        u_next = u_curr[1:-1] - dt*a_[1:-1]*ux + (dt**2/2)*(((a_[1:-1])**2)*uxx + a_[1:-1]*ax*ux) #1~Nx-1
        U = U.at[1:-1,i+1].set(u_next)
        # Next timestep at right boundary using Beam Warming
        r = a_[-1] * (dt/dx)
        u_next_r = (r*(r-1)/2)*u_curr[-3] + (2*r-r**2)*u_curr[-2] + (1-(3*r-r**2)/2)*u_curr[-1]
        U = U.at[-1,i+1].set(u_next_r)
        return U
    # Run loop
    UU = lax.fori_loop(0, Nt, body_fn, U)
    return UU

# Directories and hyperparams
data_dir = 'data/' 
data_name = 'Advection_a_data.mat' 

N_train, N_test = 1000, 100
length_scale = 0.2
scale = 0.2
Nx, Nt = 128, 256
jx, jt = Nx//32, Nt//32
p = lambda x: jnp.sin(jnp.pi*x)
q = lambda t: jnp.zeros_like(t)
xmin, xmax = 0, 1
x = jnp.linspace(xmin, xmax, Nx+1)

# train generation
key = random.key(0)
keys = random.split(key,N_train)
input_train = np.exp(scale * jax.vmap(lambda k: gen_f(k, Nx, length_scale)(x))(keys))
output_train = jax.vmap(lambda A: solve_advec_LW(Nx, Nt, A, p, q))(input_train)

# val generation
key = random.key(113355)
keys = random.split(key,N_test)
input_val = np.exp(scale * jax.vmap(lambda k: gen_f(k, Nx, length_scale)(x))(keys))
output_val = jax.vmap(lambda A: solve_advec_LW(Nx, Nt, A, p, q))(input_val)

mdic = {"input_train": input_train[:,::jx], "output_train": output_train[:,::jx,::jt],
        "input_val": input_val[:,::jx], "output_val": output_val[:,::jx,::jt]}
io.savemat(data_dir+data_name,mdic)