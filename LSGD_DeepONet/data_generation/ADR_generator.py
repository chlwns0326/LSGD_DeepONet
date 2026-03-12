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
    diff = jnp.abs(x1[None,:]-x2[:,None])
    return var * jnp.exp(-diff**2/(2*l**2))

# Original source from https://zenodo.org/records/5206676
def gen_f(key, NX, l, xmin=0, xmax=1):
    subkeys = random.split(key, 2)
    # Generate a GP sample
    jitter = 1e-12
    X = jnp.linspace(xmin, xmax, NX+1)
    K = RBF(X, X, 1, l)
    L = jnp.linalg.cholesky(K + jitter*jnp.eye(NX+1))
    gp_sample = jnp.dot(L, random.normal(subkeys[0], (NX+1,)))
    # Create a callable interpolation function  
    f = lambda x: jnp.interp(x, X.flatten(), gp_sample)    
    return f

# Advection-diffusion-reaction problem FDM(Crank-Nicolson) solver 
# Original source from https://zenodo.org/records/5206676
def solve_ADR(Nx, Nt, D, v, R, dR, f, u0):
    """
    Solve 1D Advection-Diffusion-Reaction (conservative form) on the unit interval 
    u_t = div (D(x) grad_u - v(x) u) + R(u) + f(x)
        = (D u_x - vu)_x + R(u) + f
        = D_x u_x + D u_xx - v_x u - v u_x + R(u) + f
    with initial u0 and zero Dirichlet boundary conditions 
    using the Crank-Nicolson scheme and 1st Taylor approx of R.
    
    Un: u_next, Uc: u_current.
    The Crank-Nicolson scheme is
    (Un+Uc)/dt = 1/2 (F(Un) + F(Uc)),
    where F(u) = D_x u_x + D u_xx - v_x u - v u_x + R(u) + f 
    and the 1st Taylor approximation is given as R(Un) = R(Uc) + R'(Uc) (Un-Uc).
    So, 
    1/dt * Un - 1/2 * (D_x Un_x + D Un_xx - v_x Un - v Un_x) - 1/2 * R'(Uc)Un
    = 1/dt * Uc + 1/2 * (D_x Uc_x + D Uc_xx - v_x Uc - v Uc_x) - 1/2 * R'(Uc)Uc + R(Uc) + f.
    This reads to the system of variable Un:
    A1 Un = A2 Uc + b2, 
    where A1, A2, Uc, b2 are given.
    """
    
    # Create grid
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    x = jnp.linspace(xmin, xmax, Nx+1)
    t = jnp.linspace(tmin, tmax, Nt+1)
    dx = x[1]-x[0]
    dt = t[1]-t[0]

    # Compute time independent values
    D_ = D(x)
    v_ = v(x)
    f_ = f(x)
    
    # Compute finite difference operators
    Diff0 = jnp.eye(Nx+1)
    Diff1 = 1/(2*dx) * (jnp.eye(Nx+1,k=1) - jnp.eye(Nx+1,k=-1))
    Diff2 = 1/dx**2 * (jnp.eye(Nx+1,k=1) + jnp.eye(Nx+1,k=-1) - 2*jnp.eye(Nx+1))
    
    D_term = (jnp.diag(Diff1@D_) @ Diff1 + jnp.diag(D_) @ Diff2)[1:-1,1:-1]
    v_term = -(jnp.diag(Diff1@v_) @ Diff0 + jnp.diag(v_) @ Diff1)[1:-1,1:-1]
    Dv_term = D_term + v_term

    # Initialize solution and apply initial condition
    u = jnp.zeros((Nx+1, Nt+1))
    u = u.at[:,0].set(u0(x))
    
    # Timestep update (Crank-Nicolson)
    def next_timestep(i,u):
        Ri = R(u[1:-1,i])
        dRi = dR(u[1:-1,i])
        R_Taylor_correction = 1/2 * jnp.diag(dRi)
        A_left = 1/dt * Diff0[1:-1,1:-1] - 1/2 * Dv_term - R_Taylor_correction
        A_right = 1/dt * Diff0[1:-1,1:-1] + 1/2 * Dv_term - R_Taylor_correction
        b1 = A_right @ u[1:-1,i].T
        b2 = f_[1:-1] + Ri
        u = u.at[1:-1,i+1].set(jnp.linalg.solve(A_left,b1+b2))
        return u
    
    # Run loop
    UU = lax.fori_loop(0, Nt, next_timestep, u)
    return UU

# Directory and hyperparams
data_dir = 'data/'
suffix = ''
mode = 'f' # 'f' 'v' 'R' 

N_train, N_test = 1000, 100
Nx, Nt = 128, 256
jx, jt = Nx//32, Nt//32
pi = jnp.pi
D = lambda x: 0.01*jnp.ones_like(x)
u0 = lambda x: jnp.zeros_like(x)
xmin, xmax = 0, 1
x = jnp.linspace(0, 1, Nx+1)
x_pm = jnp.linspace(-1, 1, Nx+1)
x_pm_elong = jnp.linspace(-1-1/Nx, 1+1/Nx, Nx+3) # One more point for each endpoint; below -1 and above +1.

# train/val generation
key = random.key(0)
keys = random.split(key,N_train)
rkey = random.key(113355)
rkeys = random.split(key,N_test)

# f input: scale = 0.5, l = 0.2, v = 0, R = u**2
if mode == 'f':
    l, scale = 0.2, 0.5
    v = lambda x: 0.0*jnp.sin(pi*x)
    R, dR= lambda u: u**2, lambda u: 2*u
    input_train = scale*jax.vmap(lambda key: gen_f(key, Nx, l)(x))(keys) 
    output_train = jax.vmap(lambda fvec: solve_ADR(Nx, Nt, D, v, R, dR, lambda xq: jnp.interp(xq, x, fvec), u0))(input_train)
    input_val = scale*jax.vmap(lambda key: gen_f(key, Nx, l)(x))(rkeys)
    output_val = jax.vmap(lambda fvec: solve_ADR(Nx, Nt, D, v, R, dR, lambda xq: jnp.interp(xq, x, fvec), u0))(input_val)
    
# v input: scale = 0.2, l = 0.2, f = -1+0.5*pi*sin(pi*x), R = u**2
elif mode == 'v':
    l, scale = 0.2, 0.2
    f = lambda x: 0.5*(-2 + pi*jnp.sin(pi*x))
    R, dR= lambda u: u**2, lambda u: 2*u
    input_train = scale*jax.vmap(lambda key: gen_f(key, Nx, l)(x))(keys) 
    output_train = jax.vmap(lambda vvec: solve_ADR(Nx, Nt, D, lambda xq: jnp.interp(xq, x, vvec), R, dR, f, u0))(input_train)
    input_val = scale*jax.vmap(lambda key: gen_f(key, Nx, l)(x))(rkeys)
    output_val = jax.vmap(lambda vvec: solve_ADR(Nx, Nt, D, lambda xq: jnp.interp(xq, x, vvec), R, dR, f, u0))(input_val)
    
# R input: scale = 0.2, l = 0.3, v = 0.1*sin(pi*x), f = 0.2 + 0.2*cos(2*pi*x)
elif mode == 'R':
    l, scale = 0.2, 0.3
    f = lambda x: 0.2 + 0.2*jnp.cos(2*pi*x)
    v = lambda x: 0.1*jnp.sin(pi*x)
    input_train_long = scale*jax.vmap(lambda key: gen_f(key, Nx, l, xmin=-1, xmax=1)(x_pm_elong))(keys) 
    input_train = input_train_long[:,1:-1] 
    input_train_diff = (input_train_long[:,2:]-input_train_long[:,:-2])/(2*Nx)
    output_train = jax.vmap(lambda Rvec, dRvec: solve_ADR(Nx, Nt, D, v, \
        lambda xq: jnp.interp(xq, x, Rvec), lambda xq: jnp.interp(xq, x, dRvec), f, u0),(0,0))(input_train,input_train_diff)
    input_val_long = scale*jax.vmap(lambda key: gen_f(key, Nx, l, xmin=-1, xmax=1)(x_pm_elong))(rkeys) 
    input_val = input_val_long[:,1:-1] 
    input_val_diff = (input_val_long[:,2:]-input_val_long[:,:-2])/(2*Nx)
    output_val = jax.vmap(lambda Rvec, dRvec: solve_ADR(Nx, Nt, D, v, \
        lambda xq: jnp.interp(xq, x, Rvec), lambda xq: jnp.interp(xq, x, dRvec), f, u0),(0,0))(input_val,input_val_diff)

data_name = 'ADR_' + mode + '_data' + suffix + '.mat'
mdic = {"input_train": input_train[:,::jx], "output_train": output_train[:,::jx,::jt],
        "input_val": input_val[:,::jx], "output_val": output_val[:,::jx,::jt]}
io.savemat(data_dir+data_name,mdic)