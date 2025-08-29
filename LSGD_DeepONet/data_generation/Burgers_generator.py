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

def GRF_periodic(data_num, freq, Nx, gamma, tau, sigma, key):
    keys = random.split(key,3)
    ex = np.arange(freq)
    my_eigs = ((np.sqrt(2)*np.abs(sigma))*(4*np.pi**2*ex**2 + tau**2)**(-gamma/2)).T
    my_eigs_full = np.concatenate((my_eigs,my_eigs))
    xi = random.normal(keys[0],(data_num,2*freq))
    coeff = my_eigs_full*xi
    bases = np.zeros((2*freq,Nx+1))
    x = np.linspace(0,1,Nx+1)
    for i in np.arange(1,freq+1):
        bases[i-1,:] = np.sin(2*(i)*np.pi*x)
        bases[freq+i-1,:] = np.cos(2*(i)*np.pi*x)
    out_GRF = coeff @ bases
    return out_GRF

# periodic kernel
def RBF_periodic(x1, x2, var, l, p):
    # Original source from https://zenodo.org/records/5206676
    diff = jnp.abs(x1[None,:]-x2[:,None])
    return var * jnp.exp(-(2/l**2)*jnp.sin(jnp.pi*diff/p)**2)

def gen_f(key, Nx, var, length_scale, period):
    # Original source from https://zenodo.org/records/5206676
    subkeys = random.split(key, 2)
    # Generate a GP sample
    xmin, xmax = 0, 1
    jitter = 1e-12
    X = jnp.linspace(xmin, xmax, Nx+1)
    K = RBF_periodic(X, X, var, length_scale, period)
    L = jnp.linalg.cholesky(K + jitter*jnp.eye(Nx+1))
    gp_sample = jnp.dot(L, random.normal(subkeys[0], (Nx+1,)))
    # Create a callable interpolation function  
    f = lambda x: jnp.interp(x, X.flatten(), gp_sample)    
    return f

# FVM; Godunov method
def solve_Burgers(Nx, Nt, f, nu, gvec, jumpx, jumpt):
    """Solve 1D
    u_t + f(u)_x = nu u_xx
    with IC given and periodic boundary conditions.
    """
    # Create grid
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    x = jnp.linspace(xmin, xmax, Nx+1)
    t = jnp.linspace(tmin, tmax, Nt+1)
    h = x[1]-x[0]
    dt = t[1]-t[0]
    g = jnp.interp(x[:-1], jnp.linspace(xmin, xmax, jnp.shape(gvec)[0]), gvec)
    
    def limiter(r):
        # set all r < 0 to 0, also to prevent division by 0 for r = -1
        _r = jnp.maximum(r,0)
        # Van Leer limiter
        phi = 2*_r/(1 + _r)
        phi = jnp.nan_to_num(phi,posinf=0)
        return phi

    def godunov(ul, ur, f):
        # Burgers flux
        fL = 0.5 * ul**2 # fL = f(ul)
        fR = 0.5 * ur**2 # fR = f(ur)
        f_min = 0
        f_interface = jnp.zeros_like(ul)
        # Flux goes to the right
        f_interface = jnp.where(ul > ur, jnp.maximum(fL, fR),f_interface)
        # Flux goes to the left
        f_interface = jnp.where(ul <= ur, jnp.minimum(fL, fR),f_interface)
        # Minimum flux at u* 
        f_interface = jnp.where((ul <= 0) & (ur > 0), f_min, f_interface)
        return f_interface
    
    def cyc_perm(f,k):
        return jnp.hstack([f[k:], f[0:k]])

    def dudt(u, t, f, nu, dx):
        # u denotes state in cell centers (in the formulas sometimes written as \bar{u})
        # f denotes f(u) in cell centers, using Burgers' equation f(u) = u^2/2
        # periodic BC imposed    
        
        # Reconstruct u_i_plus_half_left
        r = (cyc_perm(u,1) - u)/(u - cyc_perm(u,-1))
        phi = limiter(r)
        u_iph_left = u + 0.5*(u - cyc_perm(u,-1))*phi
        
        # Reconstruct u_i_plus_half_right
        r = (u - cyc_perm(u,1))/(cyc_perm(u,1) - cyc_perm(u,2))
        phi = limiter(r)
        u_iph_right = cyc_perm(u,1) + 0.5*(cyc_perm(u,1) - cyc_perm(u,2))*phi
        
        f_interface = godunov(u_iph_left, u_iph_right, f)
        f_interface_ext = np.hstack([f_interface[-1], f_interface])

        diffusion_term = nu * (cyc_perm(u,-1) - 2*u + cyc_perm(u,1)) / (dx**2) 
        du_dt = (f_interface_ext[0:-1] - f_interface_ext[1:])/dx + diffusion_term
        return du_dt
        
    def upwind_next(u, t, f, nu, dx, dt):
        # u denotes state in cell centers (in the formulas sometimes written as \bar{u})
        # f denotes f(u) in cell centers, using Burgers' equation f(u) = u^2/2
        # periodic BC imposed    
        
        # Reconstruct u_i_plus_half_left
        r = (cyc_perm(u,1) - u)/(u - cyc_perm(u,-1))
        phi = limiter(r)
        u_iph_left = u + 0.5*(u - cyc_perm(u,-1))*phi
        
        # Reconstruct u_i_plus_half_right
        r = (u - cyc_perm(u,1))/(cyc_perm(u,1) - cyc_perm(u,2))
        phi = limiter(r)
        u_iph_right = cyc_perm(u,1) + 0.5*(cyc_perm(u,1) - cyc_perm(u,2))*phi
        
        f_interface = godunov(u_iph_left, u_iph_right, f)
        f_interface_ext = jnp.hstack([f_interface[-1], f_interface])

        diffusion_term = nu * (cyc_perm(u,-1) - 2*u + cyc_perm(u,1)) / (dx**2)
        
        u_next = u + dt*((f_interface_ext[0:-1] - f_interface_ext[1:])/dx + diffusion_term)
        
        return u_next
        
    # Initialize solution and apply initial condition
    u = jnp.zeros((Nx, Nt+1))
    u = u.at[:,0].set(g)
    # Time-stepping update
    def body_fn(i,u):
        u = u.at[:,i+1].set(upwind_next(u[:,i], t, f, nu, h, dt))
        return u
    # Run loop
    UU = lax.fori_loop(0, Nt, body_fn, u)
    uout = jnp.concatenate([UU[::jumpx,::jumpt], UU[0:1,::jumpt]])
    
    return uout  

def main():
    # Directories and hyperparams
    data_dir = 'data/' 
    data_name = 'Burgers_data.mat' 

    N_train, N_test = 1000, 100
    Nx, Nt = 128, 2048 
    jx, jt = Nx//32, Nt//32
    freq = 8
    gamma, tau, sigma = 4, 5, 25**2 # GRF sigma*(-Lap+tau**2*I)**(-gamma)
    nu = 0.02
    f = 0 # lambda x: 0.5 * x**2 # lamg func does not work, implemented f=0.5*u**2 inside godunov

    # train generation
    key = random.key(0)
    input_train = GRF_periodic(N_train, freq, Nx, gamma, tau, sigma, key)
    output_train = jax.vmap(lambda gvec: solve_Burgers(Nx, Nt, f, nu, gvec, jx, jt))(input_train) 
    
    # val generation
    key = random.key(113355)
    input_val = GRF_periodic(N_test, freq, Nx, gamma, tau, sigma, key)
    output_val = jax.vmap(lambda gvec: solve_Burgers(Nx, Nt, f, nu, gvec, jx, jt))(input_val) 

    mdic = {"input_train": input_train[:,::jx], "output_train": output_train[:,:,:],
            "input_val": input_val[:,::jx], "output_val": output_val[:,:,:]}
    io.savemat(data_dir+data_name,mdic)
    
if __name__ == '__main__':    
    main()