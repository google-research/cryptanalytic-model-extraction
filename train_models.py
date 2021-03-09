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

def matmul(a,b,c,np=jnp):
    if c is None:
        c = np.zeros(1)

    return np.dot(a,b)+c

class SimpleClassifier:
    def __init__(self, sizes, activation):
        self.ops = [matmul]*(len(sizes)-1)
        self.weights = []
        self.biases = []
        self.activation = activation
        for i,(op, a,b) in enumerate(zip(self.ops, sizes, sizes[1:])):
            self.weights.append(onp.random.normal(size=(a,b))/(b**.5))
            self.biases.append(onp.zeros((b,)))
        return
        
    def run(self, x, weights, biases):
        """
        Run the neural network forward on the input x using the matrix A,B.
        """
        for i,(op,a,b) in enumerate(zip(self.ops,weights,biases)):
            x = op(x,a,b)
            if i < len(sizes)-2:
                x = self.activation(x)
        return x
        
    def getParams(self):
        return [self.weights, self.biases]
        
    def loss(self, params, inputs, targets):
        logits = self.run(inputs, params[0], params[1])
        # L2 loss is best loss
        res = (targets-logits.flatten())**2
        return jnp.mean(res)

@jax.jit
def update(i, opt_state, batch_x, batch_y):
    params = get_params(opt_state)
    return opt_update(i, loss_grad(params, batch_x, batch_y), opt_state)

seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42 # for luck
onp.random.seed(seed)

sizes = list(map(int,sys.argv[1].split("-")))

model = SimpleClassifier(sizes, jax.nn.relu)


SAMPLES = 20

init, opt_update, get_params = jax.experimental.optimizers.adam(3e-4)

X = onp.random.normal(size=(SAMPLES, sizes[0]))
Y = onp.array(onp.random.normal(size=SAMPLES)>0,dtype=onp.float32)

loss_grad = jax.grad(model.loss)
params = model.getParams()
opt_state = init(params)

BS = 4

# Train loop.

step = 0
for i in range(100):
    if i%10 == 0:
        print('loss', model.loss(params, X, Y))

    for j in range(0,SAMPLES,BS):
        batch_x = X[j:j+BS]
        batch_y = Y[j:j+BS]

        # gradient descent!
        opt_state = update(step, opt_state, batch_x, batch_y)
        params = get_params(opt_state)

        step += 1
        
# Save our amazing model.
onp.save("models/" + str(seed) + "_" + "-".join(map(str,sizes)), params)
