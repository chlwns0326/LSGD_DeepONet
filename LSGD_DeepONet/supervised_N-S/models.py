# Model implemented as classes
import jax
import jax.numpy as jnp
from jax.nn import tanh, relu, leaky_relu, silu
from jax.nn.initializers import glorot_normal
from flax import linen as nn
from typing import Any, Callable, Sequence
from initialization import *
jax.config.update('jax_enable_x64', True)

class MLP(nn.Module):
    features: Sequence[int]
    activation: Callable
    init_scale: float
    init_sine: bool
            
    @nn.compact
    def __call__(self, inputs):
        x = self.init_scale*inputs
        if x.ndim > 1:
            x = x.reshape((x.shape[0], -1))
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, kernel_init=glorot_normal(), name=f'Dense_Layer_{i}',dtype=jnp.float64)(x)
            if i == 0 and self.init_sine == True:
                x = jnp.sin(x)
            else:
                x = self.activation(x)
        return x
    
class FFT_MLP(nn.Module):
    features: Sequence[int]
    activation: Callable
    init_scale: float
    init_sine: bool
            
    @nn.compact
    def __call__(self, inputs):
        x = self.init_scale*inputs
        b, d1, d2 = x.shape[0], x.shape[-2], x.shape[-1]
        t = jnp.fft.rfft2(x[:,:-1,:-1]) # cut duplicates at the period, reduced rfft returned (of size Nx * Ny//2+1)
        t_real = jnp.real(t) # coeffs
        t_imag = jnp.imag(t)
        COR = t_real[:,:d1//2+1,0].reshape((b, -1)) # Nx//2 + 1
        COI = t_imag[:,1:d1//2,0].reshape((b, -1)) # Nx//2 - 1
        CMR = t_real[:,:,1:-1].reshape((b, -1)) # Nx * (Ny//2 - 1)
        CMI = t_imag[:,:,1:-1].reshape((b, -1)) # Nx * (Ny//2 - 1)
        CKR = t_real[:,:d1//2+1,-1].reshape((b, -1)) # Nx//2 + 1
        CKI = t_imag[:,1:d1//2,-1].reshape((b, -1)) # Nx//2 - 1
        x = jnp.concatenate([COR, COI, CMR, CMI, CKR, CKI], axis=1) # Batch by (Nx*Ny)
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, kernel_init=glorot_normal(), name=f'Dense_Layer_{i}',dtype=jnp.float64)(x)
            if i == 0 and self.init_sine == True:
                x = jnp.sin(x)
            else:
                x = self.activation(x)
        return x
    
class res_MLP(nn.Module):
    features: Sequence[int]
    activation: Callable
    init_scale: float
    init_sine: bool
    
    @nn.compact
    def __call__(self, inputs):
        x = self.init_scale*inputs
        if x.ndim > 1:
            x = x.reshape((x.shape[0], -1))
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, kernel_init=glorot_normal(), name=f'Dense_Layer_with_Skip_Conn_{i}',dtype=jnp.float64)(x)
            if i == 0 and self.init_sine == True:
                if i+1 < len(self.features) and feat == self.features[i+1]:
                    x = x + jnp.sin(x) # skip connection where L_i == L_i+1
                else:
                    x = jnp.sin(x)
            else:
                if i+1 < len(self.features) and feat == self.features[i+1]:
                    x = x + self.activation(x) # skip connection where L_i == L_i+1
                else:
                    x = self.activation(x)
        return x
    
class CNN_MLP(nn.Module): # NHWC
    features_CNN: Sequence[list] # [#outC, (kernel), (stride), (padding)]
    features_MLP: Sequence[int]
    activation: Callable
    init_scale: float
    init_sine: bool
  
    @nn.compact
    def __call__(self, inputs):
        x = self.init_scale*inputs
        for i, feat in enumerate(self.features_CNN):
            x = nn.Conv(features=feat[0], kernel_size=feat[1], strides=feat[2], padding=feat[3], name=f'Conv_Layer_{i}',dtype=jnp.float64)(x)
            if i == 0 and self.init_sine == True:
                x = jnp.sin(x)
            else:
                x = self.activation(x)
        x = x.reshape((x.shape[0], -1))
        for i, feat in enumerate(self.features_MLP):
            x = nn.Dense(features=feat, kernel_init=glorot_normal(), name=f'Dense_Layer_{i}',dtype=jnp.float64)(x)
            x = self.activation(x)
        return x
    
class CNN_res_MLP(nn.Module):
    features_CNN: Sequence[list] # [#outC, (kernel), (stride)]
    features_MLP: Sequence[int]
    activation: Callable
    init_scale: float
    init_sine: bool
  
    @nn.compact
    def __call__(self, inputs):
        x = self.init_scale*inputs
        for i, feat in enumerate(self.features_CNN):
            x = nn.Conv(features=feat[0], kernel_size=feat[1], strides=feat[2], padding='VALID', name=f'Conv_Layer_{i}',dtype=jnp.float64)(x)
            if i == 0 and self.init_sine == True:
                x = jnp.sin(x)
            else:
                x = self.activation(x)
        x = x.reshape((x.shape[0], -1))
        for i, feat in enumerate(self.features_MLP):
            x = nn.Dense(feat, kernel_init=glorot_normal(), name=f'Dense_Layer_with_Skip_Conn_{i}',dtype=jnp.float64)(x)
            if i+1 < len(self.features_MLP) and feat == self.features_MLP[i+1]:
                x = x + self.activation(x) # skip connection where L_i == L_i+1
            else:
                x = self.activation(x)
        return x        

lrelu = lambda x: leaky_relu(x,negative_slope=0.1)
  
#### Hyperparams

## Structure
## A
# Branch: (65,65,11)-> (16,16,32)-> (8,8,64) -> Flatten 4096 -> 64 -> 64 (LL out)
# Trunk: 2->128->128->64
# branch_layers_CNN = [[32,(5,5),(4,4),'VALID'],[64,(2,2),(2,2),'VALID']] 
# branch_layers_MLP = [64] * (1)
# trunk_layers =  [128,128,64]

## B
# B Branch: (65,65,11)-> (32,32,16)-> (16,16,32)-> (8,8,64) -> Flatten 4096 -> 150 -> 150 (LL out)
# B Trunk: 2->150->150->150
branch_layers_CNN = [[16,(3,3),(2,2),'VALID'],[32,(2,2),(2,2),'VALID'],[64,(2,2),(2,2),'VALID']]
branch_layers_MLP = [150] * (1) 
trunk_layers =  [150] * 3

## Activation, etc (All fixed)
activation_branch = silu # Activation function for branch # relu tanh lrelu silu
init_scale_branch = 1 # Rescale branch input 
init_sine_branch = False # First branch layer sine activation? 

activation_trunk = silu # Activation function for trunk
init_scale_trunk = 1 # Rescale trunk input
init_sine_trunk = False # First trunk layer sine activation? 

## Network construction

model_settings = [(branch_layers_CNN,branch_layers_MLP,activation_branch,init_scale_branch,init_sine_branch), 
                  (trunk_layers,activation_trunk,init_scale_trunk,init_sine_trunk)] # 2D branch network with CNN

branch_model = CNN_MLP(*model_settings[0])
trunk_model = MLP(*model_settings[1])