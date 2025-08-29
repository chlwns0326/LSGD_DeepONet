import os, warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #################################

# imports
import scipy.io as io
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
    # Create a callable interpolation function  
    f = lambda x: jnp.interp(x, X.flatten(), gp_sample)    
    return f

# A diffusion-reaction numerical solver
# Original source from https://zenodo.org/records/5206676
def solve_ADR(Nx, Nt, D, v, g, dg, fvec, u0):
    """Solve 1D
    u_t = (D(x) u_x)_x - v(x) u_x + g(u) + f(x)
    with zero initial and boundary conditions.
    """
    # Create grid
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    x = jnp.linspace(xmin, xmax, Nx+1)
    t = jnp.linspace(tmin, tmax, Nt+1)
    h = x[1]-x[0]
    dt = t[1]-t[0]

    # Compute coefficients and forcing
    D = D(x)
    v = v(x)
    f = jnp.interp(x, jnp.linspace(xmin, xmax, jnp.shape(fvec)[0]), fvec)

    # Compute finite difference operators
    D1 = jnp.eye(Nx+1,k=1) - jnp.eye(Nx+1,k=-1)
    D2 = -2*jnp.eye(Nx+1) + jnp.eye(Nx+1,k=-1) + jnp.eye(Nx+1,k=1)
    D3 = jnp.eye(Nx-1)
    M = -jnp.diag(D1@D)@D1 - 4*jnp.diag(D)@D2
    m_bond = 8*(h**2)/dt*D3 + M[1:-1,1:-1]
    v_bond = 2*h*jnp.diag(v[1:-1]) @ D1[1:-1,1:-1] + 2*h*jnp.diag(v[2:]-v[:Nx-1])
    mv_bond = m_bond + v_bond
    c = 8*(h**2)/dt*D3 - M[1:-1,1:-1] - v_bond

    # Initialize solution and apply initial condition
    u = jnp.zeros((Nx+1, Nt+1))
    u = u.at[:,0].set(u0(x))
    # Time-stepping update
    def body_fn(i,u):
        gi = g(u[1:-1,i])
        dgi = dg(u[1:-1,i])
        h2dgi = jnp.diag(4*(h**2)*dgi)
        A = mv_bond - h2dgi
        b1 = 8*(h**2)*(0.5*f[1:-1] + 0.5*f[1:-1] + gi)
        b2 = (c - h2dgi) @ u[1:-1,i].T
        u = u.at[1:-1,i+1].set(jnp.linalg.solve(A,b1+b2))
        return u
    # Run loop
    UU = lax.fori_loop(0, Nt, body_fn, u)
    return UU

# Directories and hyperparams
data_dir = 'data/'
data_name = 'DiffReact_data.mat'

N_train, N_test = 1000, 100
length_scale = 0.2
scale = 0.5
Nx, Nt = 128, 256
jx, jt = Nx//32, Nt//32
D = lambda x: 0.01*jnp.ones_like(x)
v = lambda x: jnp.zeros_like(x)
g = lambda u: u**2 
dg = lambda u: 2*u
u0 = lambda x: jnp.zeros_like(x)
xmin, xmax = 0, 1
x = jnp.linspace(xmin, xmax, Nx+1)

# train generation
key = random.key(0)
keys = random.split(key,N_train)
input_train = scale * jax.vmap(lambda key: gen_f(key, Nx, length_scale)(x))(keys)
output_train = jax.vmap(lambda fvec: solve_ADR(Nx, Nt, D, v, g, dg, fvec, u0))(input_train)

# val generation
key = random.key(113355)
keys = random.split(key,N_test)
input_val = scale * jax.vmap(lambda key: gen_f(key, Nx, length_scale)(x))(keys)
output_val = jax.vmap(lambda fvec: solve_ADR(Nx, Nt, D, v, g, dg, fvec, u0))(input_val)

mdic = {"input_train": input_train[:,::jx], "output_train": output_train[:,::jx,::jt],
        "input_val": input_val[:,::jx], "output_val": output_val[:,::jx,::jt]}
io.savemat(data_dir+data_name,mdic)