# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as onp
import jax
import jax.experimental.optimizers
import jax.numpy as jnp

activation_type = 'elu'

def matmul(a,b,c,np=jnp):
    if c is None:
        c = np.zeros(1)

    return np.dot(a,b)+c

seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42 # for luck
onp.random.seed(seed)

sizes = list(map(int,sys.argv[1].split("-")))
dimensions = [tuple([x]) for x in sizes]
neuron_count = sizes
ops = [matmul]*(len(sizes)-1)

# Let's not overcomplicate things.
# Initialize with a standard gaussian initialization.
# Yes someone with their xavier spectral kaiming initialization might do better
# But we're memorizing some random points. This works.
A = []
B = []
for i,(op, a,b) in enumerate(zip(ops, sizes, sizes[1:])):
    A.append(onp.random.normal(size=(a,b))/(b**.5))
    B.append(onp.zeros((b,)))

def activation_function(x):
    if(activation_type == 'relu'):
        #return x*(x>0)
        return jax.nn.relu(x)
    elif(activation_type == 'leaky'):
        #return x*(x>0) + 0.01*x*(x<0)
        return jax.nn.leaky_relu(x)
    elif(activation_type == 'elu'):
        return jax.nn.elu(x)
    else:
        print(str(activation_type) + ", is not a valid activation function")
        return x


def run(x, A, B):
    """
    Run the neural network forward on the input x using the matrix A,B.
    """
          
    for i,(op,a,b) in enumerate(zip(ops,A,B)):
        # Compute the matrix product.
        # This is a right-matrix product which means that rows/columns are flipped
        # from the definitions in the paper.
        # This was the first method I wrote and it doesn't make sense.
        # Please forgive me.
        x = op(x,a,b)
        if i < len(sizes)-2:
            x = activation_function(x)

    return x

def getinner(x, A, B):
    """
    Cheat to get the inner layers of the neural network.
    """
    region = []
    for i,(op,a,b) in enumerate(zip(ops,A,B)):
        x = op(x,a,b)
        region.append(onp.copy(x))
        if i < len(sizes)-2:
            x = activation_function(x)
    return region
        

def loss(params, inputs, targets):
    logits = run(inputs, params[0], params[1])
    # L2 loss is best loss
    res = (targets-logits.flatten())**2
    return jnp.mean(res)

# generate random training data

params = [A,B]


SAMPLES = 20

# Again, let's not think. Just optimize with adam.
# Your cosine cyclic learning rate schedule can have fun elsewhere.
# We just pick 3e-4 because Karpathy said so.
init, opt_update, get_params = jax.experimental.optimizers.adam(3e-4)

X = onp.random.normal(size=(SAMPLES, sizes[0]))
Y = onp.array(onp.random.normal(size=SAMPLES)>0,dtype=onp.float32)

loss_grad = jax.grad(loss)

@jax.jit
def update(i, opt_state, batch_x, batch_y):
    params = get_params(opt_state)
    return opt_update(i, loss_grad(params, batch_x, batch_y), opt_state)
opt_state = init(params)


# Who are we kidding.
# Not like we're running on a TPU pod and need that batch size of 16384
BS = 4

# Train loop.

step = 0
for i in range(100):
    if i%10 == 0:
        print('loss', loss(params, X, Y))

    for j in range(0,SAMPLES,BS):
        batch_x = X[j:j+BS]
        batch_y = Y[j:j+BS]

        # gradient descent!
        opt_state = update(step, opt_state, batch_x, batch_y)
        params = get_params(opt_state)

        step += 1
        
# Save our amazing model.
onp.save("models/" + str(seed) + "_" + "-".join(map(str,sizes)), params)
