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

from src.global_vars import *
from src.utils import run, basis, AcceptableFailure, cheat_get_inner_layers, which_is_zero

def get_grad(x, direction, eps=1e-6):
    """
    Finite differences to estimate the gradient.
    Uses just two coordinates---that's sufficient for most of the code.

    Can fail if we're right at a critical point and we get the left and right side.
           /
          X
         /
    -X--/

    """
    x = x[np.newaxis,:]
    a = run(x-eps*direction)
    b = run(x)
    g1 = (b-a)/eps
    return g1

def get_second_grad_unsigned(x, direction, eps, eps2):
    """
    Compute the second derivitive by computing the first derivitive twice.
    """
    grad_value = get_grad(x + direction*eps, direction, eps2)+get_grad(x - direction*eps, -direction, eps2)

    return grad_value[0]

MASK = np.array([1,-1,1,-1])
def get_second_grad_unsigned(x, direction, eps, eps2):
    """
    Same as the above but batched so it's more efficient.
    """
    x = np.array([x + direction * (eps - eps2),
                  x + direction * (eps),
                  x - direction * (eps - eps2),
                  x - direction * (eps)])

    out = run(x)
    
    return np.dot(out.flatten(), MASK)/eps

    

def get_ratios(critical_points, N, with_sign=True, eps=1e-5):
    """
    Compute the input weights to one neuron on the first layer.
    One of the core algorithms described in the paper.

    Given a set of critical point, compute the gradient for the first N directions.
    In practice N = range(DIM)

    Compute the second partial derivitive along each of the axes. This gives
    us the unsigned ratios corresponding to the ratio of the weights.

                      /
                  ^  /
                  | /
                  |/
             <----X---->  direction_1
                 /|
                / |
               /  V
              /  direction_2

    If we want to recover signs then we should also query on direction_1+direction_2
    And check to see if we get the correct solution.
    """
    ratios = []
    for j,point in enumerate(critical_points):
        ratio = []
        for i in N[j]:
            ratio.append(get_second_grad_unsigned(point, basis(i), eps, eps/3))
            
        if with_sign:
            both_ratio = []
            for i in N[j]:
                both_ratio.append(get_second_grad_unsigned(point, (basis(i) + basis(N[j][0]))/2, eps, eps/3))

            signed_ratio = []
            for i in range(len(ratio)):
                # When we have at least one y value already we need to orient this one
                # so that they point the same way.
                # We are given |f(x+d1)| and |f(x+d2)|
                # Compute |f(x+d1+d2)|.
                # Then either
                # |f(x+d1+d2)| = |f(x+d1)| + |f(x+d2)|
                # or
                # |f(x+d1+d2)| = |f(x+d1)| - |f(x+d2)|
                # or 
                # |f(x+d1+d2)| = |f(x+d2)| - |f(x+d1)|
                positive_error = abs(abs(ratio[0]+ratio[i])/2 - abs(both_ratio[i]))
                negative_error = abs(abs(ratio[0]-ratio[i])/2 - abs(both_ratio[i]))

                if positive_error > 1e-4 and negative_error > 1e-4:
                    print("Probably something is borked")
                    print("d^2(e(i))+d^2(e(j)) != d^2(e(i)+e(j))", positive_error, negative_error)
                    raise

                if positive_error < negative_error:
                    signed_ratio.append(ratio[i])
                else:
                    signed_ratio.append(-ratio[i])
        else:
            signed_ratio = ratio
        
        ratio = np.array(signed_ratio)

        #print(ratio)
        ratios.append(ratio)
        
    return ratios

def get_ratios_lstsq(LAYER, critical_points, N, known_T, debug=False, eps=1e-5):
    """
    Do the same thing as get_ratios, but works when we can't directly control where we want to query.
    
    This means we can't directly choose orthogonal directions, and so we're going
    to just pick random ones and then use least-squares to do it
    """
    #pickle.dump((LAYER, critical_points, N, known_T, debug, eps),
    #            open("/tmp/save.p","wb"))
    ratios = []
    for i,point in enumerate(critical_points):
        if CHEATING:
            layers = cheat_get_inner_layers(point)
            layer_vals = [np.min(np.abs(x)) for x in layers]
            which_layer = np.argmin(layer_vals)
            #print("real zero", np.argmin(np.abs(layers[0])))
            which_neuron = which_is_zero(which_layer, layers)
            #print("Which neuron?", which_neuron)

            real = A[which_layer][:,which_neuron]/A[which_layer][0,which_neuron]
        
        # We're going to create a system of linear equations
        # d_matrix is going to hold the inputs,
        # and ys is going to hold the resulting learned outputs
        d_matrix = []
        ys = []

        # Query on N+2 random points, so that we have redundency
        # for the least squares solution.
        for i in range(np.sum(known_T.forward(point) != 0)+2):
            # 1. Choose a random direction
            d = np.sign(np.random.normal(0,1,point.shape))
            d_matrix.append(d)

            # 2. See what the second partial derivitive at this value is
            ratio_val = get_second_grad_unsigned(point, d, eps, eps/3)

            # 3. Get the sign correct
            if len(ys) > 0:
                # When we have at least one y value already we need to orient this one
                # so that they point the same way.
                # We are given |f(x+d1)| and |f(x+d2)|
                # Compute |f(x+d1+d2)|.
                # Then either
                # |f(x+d1+d2)| = |f(x+d1)| + |f(x+d2)|
                # or
                # |f(x+d1+d2)| = |f(x+d1)| - |f(x+d2)|
                # or 
                # |f(x+d1+d2)| = |f(x+d2)| - |f(x+d1)|
                both_ratio_val = get_second_grad_unsigned(point, (d+d_matrix[0])/2, eps, eps/3)

                positive_error = abs(abs(ys[0]+ratio_val)/2 - abs(both_ratio_val))
                negative_error = abs(abs(ys[0]-ratio_val)/2 - abs(both_ratio_val))

                if positive_error > 1e-4 and negative_error > 1e-4:
                    print("Probably something is borked")
                    print("d^2(e(i))+d^2(e(j)) != d^2(e(i)+e(j))", positive_error, negative_error)
                    raise AcceptableFailure()

                
                if negative_error < positive_error:
                    ratio_val *= -1
            
            ys.append(ratio_val)

        d_matrix = np.array(d_matrix)
        # Now we need to compute the system of equations
        # We have to figure out what the vectors look like in hidden space,
        # so compute that precisely
        h_matrix = np.array(known_T.forward_at(point, d_matrix))

            
        # Which dimensions do we lose?
        column_is_zero = np.mean(np.abs(h_matrix)<1e-8,axis=0) > .5
        assert np.all((known_T.forward(point, with_relu=True) == 0) == column_is_zero)

        #print(h_matrix.shape)

        # Solve the least squares problem and get the solution
        # This is equal to solving for the ratios of the weight vector
        soln, *rest = np.linalg.lstsq(np.array(h_matrix, dtype=np.float32),
                                      np.array(ys, dtype=np.float32), 1e-5)
    
        # Set the columns we know to be wrong to NaN so that it's obvious
        # this isn't important but it helps us distinguish from genuine errors
        # and the kind that we can't avoic because of zero gradients
        soln[column_is_zero] = np.nan

        ratios.append(soln)
        
    return ratios
