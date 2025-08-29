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

## Structure
C = 100 # Number of hidden layer units
L = 3 # Number of layers

## Branch
branch_layers = [C] * (L-1) # FCN only, last layer excluded

## Trunk
trunk_layers =  [C] * L

## Activation, etc (All fixed)
activation_branch = silu # Activation function for branch # relu tanh lrelu silu
init_scale_branch = 1 # Rescale branch input 
init_sine_branch = False # First branch layer sine activation? 

activation_trunk = silu # Activation function for trunk
init_scale_trunk = 1 # Rescale trunk input
init_sine_trunk = False # First trunk layer sine activation? 

## Network construction.
model_settings = [(branch_layers,activation_branch,init_scale_branch,init_sine_branch),
                   (trunk_layers,activation_trunk,init_scale_trunk,init_sine_trunk)] # 1D branch network

branch_model =  MLP(*model_settings[0])
trunk_model = MLP(*model_settings[1])