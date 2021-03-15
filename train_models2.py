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
#import jax
#import jax.experimental.optimizers
#import jax.numpy as jnp
from torch import nn
import torch
import torch.optim as optim
import numpy as np
from collections import OrderedDict

activation_function_dict = {"relu": nn.ReLU, "elu": nn.ELU, "leaky": nn.LeakyReLU}

def matmul(a,b,c,np=jnp):
    if c is None:
        c = np.zeros(1)

    return np.dot(a,b)+c

class SimpleClassifier:   
    def __init__(self, sizes, activation):
		super().__init__()
		layers = OrderedDict()
		for i in range(len(sizes) - 1):
			layers.update({f"fc{i}": nn.Linear(sizes[i],sizes[i+1])})
			layers.update({f"activation{i}": activation()})
		self.model = nn.Sequential(OrderedDict(layers))
		
	def forward(self,x):
		return self.model.forward(x)
    
    #def getParams(self):
    #    return [self.weights, self.biases]
        
    #def loss(self, params, inputs, targets):
    #    logits = self.run(inputs, params[0], params[1])
    #    # L2 loss is best loss
    #    res = (targets-logits.flatten())**2
    #    return jnp.mean(res)
        
    
@jax.jit
def update(i, opt_state, batch_x, batch_y):
    params = get_params(opt_state)
    return opt_update(i, loss_grad(params, batch_x, batch_y), opt_state)

sizes = list(map(int,sys.argv[1].split("-")))
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42 # for luck
activation_function = activation_function_dict[sys.argv[3]] if len(sys.argv) > 3 else nn.Relu

model = SimpleClassifier(sizes, activation_function)


SAMPLES = 20
np.random.seed(seed)
X = np.random.normal(size=(SAMPLES, sizes[0]))
Y = np.array(np.random.normal(size=SAMPLES)>0,dtype=np.float32)


optimizer = torch.optim.Adam(model.parameters(), lr = 0.0003)


init, opt_update, get_params = jax.experimental.optimizers.adam(3e-4)


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
torch.save(model, "./models/" + str(seed) + "_" + "-".join(map(str,sizes)) + "_" + activation_function)
#np.save("./models/" + str(seed) + "_" + "-".join(map(str,sizes)), params)
