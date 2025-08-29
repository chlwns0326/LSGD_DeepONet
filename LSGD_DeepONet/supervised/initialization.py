# Functions used for model param inits
import jax
import jax.numpy as jnp
from jax import random
from jax.nn import tanh, relu
from typing import Any, Callable, Sequence
jax.config.update('jax_enable_x64', True)

# Box init (see Cyr. et al. 2020), with custom activation implemented (tanh, sin)
# Robust Training and Initialization of Deep Neural Networks: An Adaptive Basis Viewpoint
def MLP_init_box(key, shape, activation, from_r, to_r, zerofix):
    k1, k2 = random.split(key)
    d_in, d_out = shape
    if activation == relu:
        Pij = random.uniform(k1, (d_in, d_out))
        Npre = random.normal(k2, (d_in, d_out))
        Nij = Npre * jnp.reciprocal(jnp.tile(jnp.sqrt(jnp.sum(Npre**2,axis=0,keepdims=True)),(d_in,1)))
        Pmax = relu(jnp.sign(Nij))
        Ki = jnp.reciprocal(jnp.sum((Pmax-Pij)*Nij,axis=0,keepdims=True))
        Ki_long = jnp.tile(Ki,(d_in,1))
        W = Ki_long * Nij
        b = -jnp.ravel(Ki * jnp.sum((Nij * Pij),axis=0,keepdims=True))
    if activation == jnp.tanh:
        Npre = random.normal(k2, (d_in, d_out))
        Nij = Npre * jnp.reciprocal(jnp.tile(jnp.sqrt(jnp.sum(Npre**2,axis=0,keepdims=True)),(d_in,1)))
        if zerofix == True:
            Pmax = from_r * jnp.maximum(jnp.sign(Nij),-0.1) # [0,r] # [-0.1r,r] due to division by 0
        else :
            Pmax = from_r * jnp.sign(Nij) # [-r,r]
        const = jnp.arctanh(to_r)
        Ki = const * jnp.reciprocal(jnp.sum(Pmax*Nij,axis=0,keepdims=True))
        Ki_long = jnp.tile(Ki,(d_in,1))
        if zerofix == True:
            W = 2 * Ki_long * Nij
            b = -0.5*from_r*jnp.sum(W,axis=0,keepdims=False)
        else:
            W = Ki_long * Nij
            b = jnp.zeros(d_out)
    if activation == jnp.sin:
        Npre = random.normal(k2, (d_in, d_out))
        Nij = Npre * jnp.reciprocal(jnp.tile(jnp.sqrt(jnp.sum(Npre**2,axis=0,keepdims=True)),(d_in,1)))
        if zerofix == True:
            Pmax = from_r * jnp.maximum(jnp.sign(Nij),-0.1) # [0,r] # [-0.1r,r] due to division by 0
        else :
            Pmax = from_r * jnp.sign(Nij) # [-r,r]
        const = jnp.arcsin(to_r)
        Ki = const * jnp.reciprocal(jnp.sum(Pmax*Nij,axis=0,keepdims=True))
        Ki_long = jnp.tile(Ki,(d_in,1))
        if zerofix == True:
            W = 2 * Ki_long * Nij
            b = -0.5*from_r*jnp.sum(W,axis=0,keepdims=False)
        else:
            W = Ki_long * Nij
            b = jnp.zeros(d_out)
    return W, b

# Box init w/ skip connection (Cyr. et al. 2020), with custom activation implemented (tanh, sin)
# Robust Training and Initialization of Deep Neural Networks: An Adaptive Basis Viewpoint
def res_MLP_init_box(key, shape, activation, curr_L, max_L, from_r, to_r, zerofix):
    k1, k2 = random.split(key)
    d_in, d_out = shape
    m = (1 + 1/(max_L-1))**(curr_L-2)
    if activation == relu:
        if curr_L == 1:
            Pij = random.uniform(k1, (d_in, d_out))
            Npre = random.normal(k2, (d_in, d_out))
            Nij = Npre * jnp.reciprocal(jnp.tile(jnp.sqrt(jnp.sum(Npre**2,axis=0,keepdims=True)),(d_in,1)))
            Pmax = relu(jnp.sign(Nij))
            Ki = jnp.reciprocal(jnp.sum((Pmax-Pij)*Nij,axis=0,keepdims=True))
            Ki_long = jnp.tile(Ki,(d_in,1))
            W = Ki_long * Nij
            b = -jnp.ravel(Ki * jnp.sum((Nij * Pij),axis=0,keepdims=True))
        else:
            Pij = m*random.uniform(k1, (d_in, d_out))
            Npre = random.normal(k2, (d_in, d_out))
            Nij = Npre * jnp.reciprocal(jnp.tile(jnp.sqrt(jnp.sum(Npre**2,axis=0,keepdims=True)),(d_in,1)))
            Pmax = m*relu(jnp.sign(Nij))
            Ki = jnp.reciprocal(jnp.sum((Pmax-Pij)*Nij,axis=0,keepdims=True)) / (max_L-1)
            Ki_long = jnp.tile(Ki,(d_in,1))
            W = Ki_long * Nij
            b = -jnp.ravel(Ki * jnp.sum((Nij * Pij),axis=0,keepdims=True))
    if activation == jnp.tanh:
        if curr_L == 1:
            Npre = random.normal(k2, (d_in, d_out))
            Nij = Npre * jnp.reciprocal(jnp.tile(jnp.sqrt(jnp.sum(Npre**2,axis=0,keepdims=True)),(d_in,1)))
            if zerofix == True:
                Pmax = from_r * jnp.maximum(jnp.sign(Nij),-0.1) # [0,r] # [-0.1r,r] due to division by 0
            else :
                Pmax = from_r * jnp.sign(Nij) # [-r,r]
            const = jnp.arctanh(to_r)
            Ki = const * jnp.reciprocal(jnp.sum(Pmax*Nij,axis=0,keepdims=True))
            Ki_long = jnp.tile(Ki,(d_in,1))
            if zerofix == True:
                W = 2 * Ki_long * Nij
                b = -0.5*from_r*jnp.sum(W,axis=0,keepdims=False)
            else:
                W = Ki_long * Nij
                b = jnp.zeros(d_out)
        else :
            Npre = random.normal(k2, (d_in, d_out))
            Nij = Npre * jnp.reciprocal(jnp.tile(jnp.sqrt(jnp.sum(Npre**2,axis=0,keepdims=True)),(d_in,1)))
            if zerofix == True:
                Pmax = from_r*m*jnp.maximum(jnp.sign(Nij),-0.1) # [0,rm] # [-0.1rm,rm] due to division by 0
            else :
                Pmax = from_r*m*jnp.sign(Nij) # [-rm,rm]
            const = jnp.arctanh(from_r*m/(max_L-1))
            Ki = const * jnp.reciprocal(jnp.sum(Pmax*Nij,axis=0,keepdims=True))
            Ki_long = jnp.tile(Ki,(d_in,1))
            if zerofix == True:
                W = 2 * Ki_long * Nij
                b = -0.5*from_r*jnp.sum(W,axis=0,keepdims=False)
            else:
                W = Ki_long * Nij
                b = jnp.zeros(d_out)
    if activation == jnp.sin:
        if curr_L == 1:
            Npre = random.normal(k2, (d_in, d_out))
            Nij = Npre * jnp.reciprocal(jnp.tile(jnp.sqrt(jnp.sum(Npre**2,axis=0,keepdims=True)),(d_in,1)))
            if zerofix == True:
                Pmax = from_r * jnp.maximum(jnp.sign(Nij),-0.1) # [0,r] # [-0.1r,r] due to division by 0
            else :
                Pmax = from_r * jnp.sign(Nij) # [-r,r]
            const = jnp.arcsin(to_r)
            Ki = const * jnp.reciprocal(jnp.sum(Pmax*Nij,axis=0,keepdims=True))
            Ki_long = jnp.tile(Ki,(d_in,1))
            if zerofix == True:
                W = 2 * Ki_long * Nij
                b = -0.5*from_r*jnp.sum(W,axis=0,keepdims=False)
            else:
                W = Ki_long * Nij
                b = jnp.zeros(d_out)
        else :
            Npre = random.normal(k2, (d_in, d_out))
            Nij = Npre * jnp.reciprocal(jnp.tile(jnp.sqrt(jnp.sum(Npre**2,axis=0,keepdims=True)),(d_in,1)))
            if zerofix == True:
                Pmax = from_r*m*jnp.maximum(jnp.sign(Nij),-0.1) # [0,rm] # [-0.1rm,rm] due to division by 0
            else :
                Pmax = from_r*m*jnp.sign(Nij) # [-rm,rm]
            const = jnp.arcsin(from_r*m/(max_L-1))
            Ki = const * jnp.reciprocal(jnp.sum(Pmax*Nij,axis=0,keepdims=True))
            Ki_long = jnp.tile(Ki,(d_in,1))
            if zerofix == True:
                W = 2 * Ki_long * Nij
                b = -0.5*from_r*jnp.sum(W,axis=0,keepdims=False)
            else:
                W = Ki_long * Nij
                b = jnp.zeros(d_out)
    return W, b

# CNN layer init
def CNN_init(key, shape, range):
    k1, k2 = random.split(key)
    W = range * random.normal(k1, shape)
    b = jnp.zeros((shape[-1],))
    
    return W, b

# Proper orthogonal init (H. Lee. et al., 2024.)
# Improved weight initialization for deep and narrow feedforward neural network
def prop_orth_init(shape, eps):
    d_in, d_out = shape # n, m
    Jn = jnp.ones((d_in,d_in)) + eps*jnp.eye(d_in)
    Jm = jnp.ones((d_out,d_out)) + eps*jnp.eye(d_out)
    Qn, Rn = jnp.linalg.qr(Jn,mode='complete')
    Qm, Rm = jnp.linalg.qr(Jm,mode='complete')
    
    W = Qn @ jnp.eye(d_in, d_out) @ Qm.T
    b = jnp.zeros((shape[-1],))
    
    return W, b