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

import jax
import sys
import numpy as onp
import jax.numpy as jnp
import numpy as np
import numpy.linalg
import networkx as nx
import matplotlib.pyplot as plt
import scipy.optimize
import pickle

from jax.config import config
config.update("jax_enable_x64", True)

def relu(x):
    return x * (x>0)

#@jax.jit
def run(x,A,B,debug=True, np=np):
    for i,(a,b) in enumerate(zip(A,B)):
        x = np.dot(x,a)+b
        if i < len(A)-1:
            x = x*(x>0)
    return x


name = sys.argv[1] if len(sys.argv) > 1 else "40-20-10-10-1"

prefix = "/tmp/"

A1, B1 = pickle.load(open(prefix+"real-%s.p"%name,"rb"))
A2, B2 = pickle.load(open(prefix+"extracted-%s.p"%name,"rb"))

A1 = [np.array(x,dtype=np.float64) for x in A1]
A2 = [np.array(x,dtype=np.float64) for x in A2]

B1 = [np.array(x,dtype=np.float64) for x in B1]
B2 = [np.array(x,dtype=np.float64) for x in B2]


print("Compute the matrix alignment for the SVD upper bound")
for layer in range(len(A1)-1):
    M_real = np.copy(A1[layer])
    M_fake = np.copy(A2[layer])

    scores = []
    
    for i in range(M_real.shape[1]):
        vec = M_real[:,i:i+1]
        ratio = np.abs(M_fake/vec)

        scores.append(np.std(A2[layer]/vec,axis=0))

    
    i_s, j_s = scipy.optimize.linear_sum_assignment(scores)
    
    for i,j in zip(i_s, j_s):
        vec = M_real[:,i:i+1]
        ratio = np.abs(M_fake/vec)

        ratio = np.median(ratio[:,j])
        #print("Map from", i, j, ratio)

        gap = np.abs(M_fake[:,j]/ratio - M_real[:,i])
        
        A2[layer][:,j] /= ratio
        B2[layer][j] /= ratio
        A2[layer+1][j,:] *= ratio

    A2[layer] = A2[layer][:,j_s]
    B2[layer] = B2[layer][j_s]
    
    A2[layer+1] = A2[layer+1][j_s,:]

A2[1] *= np.sign(A2[1][0])
A2[1] *= np.sign(A1[1][0])

B2[1] *= np.sign(B2[1])
B2[1] *= np.sign(B1[1])

print("Finished alignment. Now compute the max error in the matrix.")
max_err = 0
for l in range(len(A1)):
    print("Matrix diff", np.sum(np.abs(A1[l]-A2[l])))
    print("Bias diff", np.sum(np.abs(B1[l]-B2[l])))
    max_err = max(max_err, np.max(np.abs(A1[l]-A2[l])))
    max_err = max(max_err, np.max(np.abs(B1[l]-B2[l])))

print("Number of bits of precision in the weight matrix",
      -np.log(max_err)/np.log(2))

print("\nComputing SVD upper bound")
high = np.ones(A1[0].shape[0])
low = -np.ones(A1[0].shape[0])
input_bound = np.sum((high-low)**2)**.5
prev_bound = 0
for i in range(len(A1)):
    largest_value = np.linalg.svd(A1[i]-A2[i])[1][0] * input_bound
    largest_value += np.linalg.svd(A1[i])[1][0] * prev_bound
    largest_value += np.sum((B1[i]-B2[i])**2)**.5
    prev_bound = largest_value
    print("\tAt layer", i, "loss is bounded by", largest_value)

print('Upper bound on number of bits of precision in the output through SVD', -np.log(largest_value)/np.log(2))

print("\nFinally estimate it through random samples to make sure we haven't made a mistake") # not that that would ever happen
def loss(x):
    return np.abs(run(x, A=A1, B=B1)-run(x, A=A2, B=B2))

ls = []
for _ in range(100):
    if _%10 == 0:
        print("Iter %d/100"%_)
    inp = onp.random.normal(0, 1, (int(1000000/A1[0].shape[0]), A1[0].shape[0]))
    inp /= np.sum(inp**2,axis=1,keepdims=True)**.5
    inp *= np.sum((np.ones(A1[0].shape[0])*2)**2)**.5
    ell = loss(inp).flatten()
    ls.extend(ell)

ls = onp.array(ls).flatten()

print("Fewest number of bits of precision over", len(ls), "random samples:", -np.log(np.max(ls))/np.log(2))

# Finally plot a distribution of the values to see
plt.hist(ls,30)
plt.semilogy()
plt.savefig("/tmp/a.pdf")
exit(0)
