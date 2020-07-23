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

import numpy as np
import jax.numpy as jnp
import sys

from src.global_vars import *
from src.global_vars import __cheat_A, __cheat_B


def matmul(a,b,c,np=np):
    if c is None:
        c = np.zeros(1)

    return np.dot(a,b)+c
        
def relu(x):
    return x * (x>0)

# Okay so this is an ugly hack
# I want to track where the queries come from.
# So in order to pretty print line numer -> code
# open up the current file and use this as a lookup.

TRACK_LINES = False
self_lines = open(sys.argv[0]).readlines()

# We're going to keep track of all queries we've generated so that we can use them later on
# (in order to save on query efficiency)
# Format: [(x, f(x))]
SAVED_QUERIES = []


def run(x,inner_A=__cheat_A,inner_B=__cheat_B):
    """
    Run the neural network forward on the input x using the matrix A,B.
    
    Log the result as having happened so that we can debug errors and
    improve query efficiency.
    """
    global query_count
    query_count += x.shape[0]
    assert len(x.shape) == 2

    orig_x = x

    for i,(a,b) in enumerate(zip(inner_A,inner_B)):
        # Compute the matrix product.
        # This is a right-matrix product which means that rows/columns are flipped
        # from the definitions in the paper.
        # This was the first method I wrote and it doesn't make sense.
        # Please forgive me.
        x = matmul(x,a,b)
        if i < len(sizes)-2:
            x = x*(x>0)
    SAVED_QUERIES.extend(zip(orig_x,x))

    if TRACK_LINES:
        for line in traceback.format_stack():
            if 'repeated' in line: continue
            line_no = int(line.split("line ")[1].split()[0][:-1])
            if line_no not in query_count_at:
                query_count_at[line_no] = 0
            query_count_at[line_no] += x.shape[0]

    return x


class NoCheatingError(Exception):
    """
    This error is thrown by functions that cheat if we're in no-cheating mode.

    To debug code it's helpful to be able to look at the weights directly,
    and inspect the inner activations of the model.
    
    But sometimes debug code can be left in by accident and we might pollute
    the actual results of the paper by cheating. This error is thrown by all
    functions that cheat so that we can't possibly do it by accident.
    """
    
class AcceptableFailure(Exception):
    """
    Sometimes things fail for entirely acceptable reasons (e.g., we haven't
    queried enough points to have seen all the hyperplanes, or we get stuck
    in a constant zero region). When that happens we throw an AcceptableFailure
    because life is tough but we should just back out and try again after
    making the appropriate correction.
    """
    def __init__(self, *args, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class GatherMoreData(AcceptableFailure):
    """
    When gathering witnesses to hyperplanes, sometimes we don't have
    enough and need more witnesses to *this particular neuron*.
    This error says that we should gather more examples of that one.
    """
    def __init__(self, data, **kwargs):
        super(GatherMoreData, self).__init__(data=data, **kwargs)

def _cheat_get_inner_layers(x,A=__cheat_A,B=__cheat_B, as_list=False):
    """
    Cheat to get the inner layers of the neural network.
    """
    region = []
    for i,(a,b) in enumerate(zip(A,B)):
        x = matmul(x,a,b)
        region.append(np.copy(x))
        if i < len(sizes)-2:
            x = x*(x>0)
    return region

def cheat_get_inner_layers(x,A=A,B=B, as_list=False):
    if not CHEATING: raise NoCheatingError()
    return _cheat_get_inner_layers(x,A,B,as_list)

def _cheat_get_polytope_id(x,A=__cheat_A,B=__cheat_B, as_list=False, flatten=True):
    """
    Cheat to get the polytope ID of the network.
    """
    if not CHEATING: raise NoCheatingError()
    region = []
    for i,(a,b) in enumerate(zip(A,B)):
        x = matmul(x,a,b)
        if i < len(sizes)-2:
            region.append(x<0)
            x = x*(x>0)
    if flatten:
        arr = np.array(np.concatenate(region,axis=1),dtype=np.int64)
    else:
        arr = region
    if as_list:
        return arr
    arr *= 1<<np.arange(arr.shape[1])
    return np.sum(arr,axis=1)


def cheat_get_polytope_id(x,A=A,B=B, as_list=False, flatten=False):
    if not CHEATING: raise NoCheatingError()
    return _cheat_get_polytope_id(x,A,B,as_list, flatten)

def cheat_num_relu_crosses(low, high):
    """
    Compute the number of relu crosses between low and high.
    This can be a lower bound if some relu goes from 0 to 1 and back to 0,
    the function here will return 0 for that relu.
    """
    if not CHEATING: raise NoCheatingError()
    r1 = cheat_get_polytope_id(low, as_list=True, flatten=False)
    r2 = cheat_get_polytope_id(high, as_list=True, flatten=False)

    o = []
    for layer1,layer2 in zip(r1,r2):
        o.append(np.sum(layer1 != layer2))

    return o

def basis(i, N=DIM):
    """
    Standard basis vector along dimension i
    """
    a = np.zeros(N, dtype=np.float64)
    a[i] = 1
    return a

def which_is_zero(layer, values):
    which = np.argmin(np.abs(values[layer]),axis=-1)
    return which
    

def get_polytope_at(known_T, known_A, known_B, x, prior=True):
    """
    Get the polytope for an input using the known transform and known A.
    This function IS NOT CHEATING.
    """
    if prior:
        which_polytope = known_T.get_polytope(x)
    else:
        which_polytope = tuple()
    LAYER = len(known_T.A)+1
    hidden = known_T.forward(x[np.newaxis,:],with_relu=True)
    which_polytope += tuple(np.int32(np.sign(matmul(hidden, known_A, known_B)))[0])
    return which_polytope


def get_hidden_at(known_T, known_A, known_B, LAYER, x, prior=True):
    """
    Get the hidden value for an input using the known transform and known A.
    This function IS NOT CHEATING.
    """
    if prior:
        which_activation = [y for x in known_T.get_hidden_layers(x) for y in x]
    else:
        which_activation = []
    which_activation += list(matmul(known_T.forward(x[np.newaxis,:], with_relu=True), known_A, known_B)[0])
    return tuple(which_activation)

class KnownT:
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def extend_by(self, a, b):
        return KnownT(self.A+[a], self.B+[b])
        
    def forward(self, x, with_relu=False, np=np):
        for i,(a,b) in enumerate(zip(self.A,self.B)):
            x = matmul(x,a,b,np)
            if (i < len(self.A)-1) or with_relu:
                x = x*(x>0)
        return x
    def forward_at(self, point, d_matrix):
        if len(self.A) == 0:
            return d_matrix

        mask_vectors = [layer > 0 for layer in self.get_hidden_layers(point)]

        h_matrix = np.array(d_matrix)
        for i,(matrix,mask) in enumerate(zip(self.A, mask_vectors)):
            h_matrix = matmul(h_matrix, matrix, None) * mask
        
        return h_matrix
    def get_hidden_layers(self, x, flat=False, np=np):
        if len(self.A) == 0: return []
        region = []
        for i,(a,b) in enumerate(zip(self.A,self.B)):
            x = matmul(x,a,b,np=np)
            if np == jnp:
                region.append(x)
            else:
                region.append(np.copy(x))
            if i < len(self.A)-1:
                x = x*(x>0)
        if flat:
            region = np.concatenate(region,axis=0)
        return region
    def get_polytope(self, x):
        if len(self.A) == 0: return tuple()
        h = self.get_hidden_layers(x)
        h = np.concatenate(h, axis=0)
        return tuple(np.int32(np.sign(h)))

def check_quality(layer_num, extracted_normal, extracted_bias, do_fix=False):
    """
    Check the quality of the solution.
    
    The first function is read-only, and just reports how good or bad things are.
    The second half, when in cheating mode, will align the two matrices.
    """

    print("\nCheck the solution of the last weight matrix.")
    
    reorder = [None]*(neuron_count[layer_num+1])
    for i in range(neuron_count[layer_num+1]):
        gaps = []
        ratios = []
        for j in range(neuron_count[layer_num+1]):
            if np.all(np.abs(extracted_normal[:,i])) < 1e-9:
                extracted_normal[:,i] += 1e-9
            ratio = __cheat_A[layer_num][:,j] / extracted_normal[:,i]
            ratio = np.median(ratio)
            error = __cheat_A[layer_num][:,j] - ratio * extracted_normal[:,i]
            error = np.sum(error**2)/np.sum(__cheat_A[layer_num][:,j]**2)
            gaps.append(error)
            ratios.append(ratio)
        print("Neuron", i, "maps on to neuron", np.argmin(gaps), "with error", np.min(gaps)**.5, 'ratio', ratios[np.argmin(gaps)])
        print("Bias check", (__cheat_B[layer_num][np.argmin(gaps)]-extracted_bias[i]*ratios[np.argmin(gaps)]))

        reorder[np.argmin(gaps)] = i
        if do_fix and CHEATING:
            extracted_normal[:,i] *= np.abs(ratios[np.argmin(gaps)])
            extracted_bias[i] *= np.abs(ratios[np.argmin(gaps)])
        
        if min(gaps) > 1e-2:
            print("ERROR LAYER EXTRACTED INCORRECTLY")
            print("\tGAPS:", " ".join("%.04f"%x for x in gaps))
            print("\t  Got:", " ".join("%.04f"%x for x in extracted_normal[:,i]/extracted_normal[0,i]))
            print("\t Real:", " ".join("%.04f"%x for x in __cheat_A[layer_num][:,np.argmin(gaps)]/__cheat_A[layer_num][0,np.argmin(gaps)]))


    # Randomly assign the unused neurons.
    used = [x for x in reorder if x is not None]
    missed = list(set(range(len(reorder))) - set(used))
    for i in range(len(reorder)):
        if reorder[i] is None:
            reorder[i] = missed.pop()
        
            
    if CHEATING:
        extracted_normal = extracted_normal[:,reorder]
        extracted_bias = extracted_bias[reorder]

    return extracted_normal,extracted_bias

