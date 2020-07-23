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

import jax
import jax.experimental.optimizers
import jax.numpy as jnp

from src.global_vars import *
from src.utils import matmul, which_is_zero
from src.find_witnesses import do_better_sweep

def trim(hidden_layer, out, num_good):
    """
    Compute least squares in a robust-statistics manner.
    See Jagielski et al. 2018 S&P
    """
    lst, *rest = np.linalg.lstsq(hidden_layer, out)
    old = lst
    for _ in range(20):
        errs = np.abs(np.dot(hidden_layer, lst) - out)
        best_errs = np.argsort(errs)[:num_good]
        lst, *rest = np.linalg.lstsq(hidden_layer[best_errs], out[best_errs])
        if np.linalg.norm(old-lst) < 1e-9:
            return lst, best_errs
        old = lst
    return lst, best_errs

def improve_row_precision(args):
    """
    Improve the precision of an extracted row.
    We think we know where it is, but let's actually figure it out for sure.

    To do this, start by sampling a bunch of points near where we expect the line to be.
    This gives us a picture like this

                      X
                       X
                    
                   X
               X
                 X
                X

    Where some are correct and some are wrong.
    With some robust statistics, try to fit a line that fits through most of the points
    (in high dimension!)

                      X
                     / X
                    / 
                   X
               X  /
                 /
                X

    This solves the equation and improves the point for us.
    """
    (LAYER, known_T, known_A, known_B, row, did_again) = args
    print("Improve the extracted neuron number", row)

    print(np.sum(np.abs(known_A[:,row])))
    if np.sum(np.abs(known_A[:,row])) < 1e-8:
        return known_A[:,row], known_B[row]
        

    def loss(x, r):
        hidden = known_T.forward(x, with_relu=True, np=jnp)
        dotted = matmul(hidden, jnp.array(known_A)[:,r], jnp.array(known_B)[r], np=jnp)
                
        return jnp.sum(jnp.square(dotted))

    loss_grad = jax.jit(jax.grad(loss))
    loss = jax.jit(loss)

    extended_T = known_T.extend_by(known_A, known_B)

    def get_more_points(NUM):
        """
        Gather more points. This procedure is really kind of ugly and should probably be fixed.
        We want to find points that are near where we expect them to be.

        So begin by finding preimages to points that are on the line with gradient descent.
        This should be completely possible, because we have d_0 input dimensions but 
        only want to control one inner layer.
        """
        print("Gather some more actual critical points on the plane")
        stepsize = .1
        critical_points = []
        while len(critical_points) <= NUM:
            print("On this iteration I have ", len(critical_points), "critical points on the plane")
            points = np.random.normal(0, 1e3, size=(100,DIM,))
            
            lr = 10
            for step in range(5000):
                # Use JaX's built in optimizer to do this.
                # We want to adjust the LR so that we get a better solution
                # as we optimize. Probably there is a better way to do this,
                # but this seems to work just fine.

                # No queries involvd here.
                if step%1000 == 0:
                    lr *= .5
                    init, opt_update, get_params = jax.experimental.optimizers.adam(lr)
                
                    @jax.jit
                    def update(i, opt_state, batch):
                        params = get_params(opt_state)
                        return opt_update(i, loss_grad(batch, row), opt_state)
                    opt_state = init(points)
                
                if step%100 == 0:
                    ell = loss(points, row)
                    if CHEATING:
                        # This isn't cheating, but makes things prettier
                        print(ell)
                    if ell < 1e-5:
                        break
                opt_state = update(step, opt_state, points)
                points = opt_state.packed_state[0][0]
                
            for point in points:
                # For each point, try to see where it actually is.

                # First, if optimization failed, then abort.
                if loss(point, row) > 1e-5:
                    continue

                if LAYER > 0:
                    # If wee're on a deeper layer, and if a prior layer is zero, then abort
                    if min(np.min(np.abs(x)) for x in known_T.get_hidden_layers(point)) < 1e-4:
                        print("is on prior")
                        continue
                    
                    
                #print("Stepsize", stepsize)
                tmp = query_count
                solution = do_better_sweep(offset=point,
                                           low=-stepsize,
                                           high=stepsize,
                                           known_T=known_T)
                #print("qs", query_count-tmp)
                if len(solution) == 0:
                    stepsize *= 1.1
                elif len(solution) > 1:
                    stepsize /= 2
                elif len(solution) == 1:
                    stepsize *= 0.98
                    potential_solution = solution[0]

                    hiddens = extended_T.get_hidden_layers(potential_solution)


                    this_hidden_vec = extended_T.forward(potential_solution)
                    this_hidden = np.min(np.abs(this_hidden_vec))
                    if min(np.min(np.abs(x)) for x in this_hidden_vec) > np.abs(this_hidden)*0.9:
                        critical_points.append(potential_solution)
                    else:
                        print("Reject it")
        print("Finished with a total of", len(critical_points), "critical points")
        return critical_points


    critical_points_list = []
    for _ in range(1):
        NUM = sizes[LAYER]*2
        critical_points_list.extend(get_more_points(NUM))
        
        critical_points = np.array(critical_points_list)

        hidden_layer = known_T.forward(np.array(critical_points), with_relu=True)

        if CHEATING:
            out = np.abs(matmul(hidden_layer, A[LAYER],B[LAYER]))
            which_neuron = int(np.median(which_is_zero(0, [out])))
            print("NEURON NUM", which_neuron)

            crit_val_0 = out[:,which_neuron]
                
            print(crit_val_0)

            #print(list(np.sort(np.abs(crit_val_0))))
            print('probability ok',np.mean(np.abs(crit_val_0)<1e-8))

        crit_val_1 = matmul(hidden_layer, known_A[:,row], known_B[row])

        best = (None, 1e6)
        upto = 100

        for iteration in range(upto):
            if iteration%1000 == 0:
                print("ITERATION", iteration, "OF", upto)
            if iteration%2 == 0 or True:

                # Try 1000 times to make sure that we get at least one non-zero per axis
                for _ in range(1000):
                    randn = np.random.choice(len(hidden_layer), NUM+2, replace=False)
                    if np.all(np.any(hidden_layer[randn] != 0, axis=0)):
                        break

                hidden = hidden_layer[randn]
                soln,*rest = np.linalg.lstsq(hidden, np.ones(hidden.shape[0]))
                
                
            else:
                randn = np.random.choice(len(hidden_layer), min(len(hidden_layer),hidden_layer.shape[1]+20), replace=False)
                soln,_ = trim(hidden_layer[randn], np.ones(hidden_layer.shape[0])[randn], hidden_layer.shape[1])


            crit_val_2 = matmul(hidden_layer, soln, None)-1
            
            quality = np.median(np.abs(crit_val_2))

            if iteration%100 == 0:
                print('quality', quality, best[1])
            
            if quality < best[1]:
                best = (soln, quality)

            if quality < 1e-10: break
            if quality < 1e-10 and iteration > 1e4: break
            if quality < 1e-8 and iteration > 1e5: break

        soln, _ = best

        if CHEATING:
            print("Compare", np.median(np.abs(crit_val_0)))
        print("Compare",
              np.median(np.abs(crit_val_1)),
              best[1])

        if np.all(np.abs(soln) > 1e-10):
            break

    print('soln',soln)
    
    if np.any(np.abs(soln) < 1e-10):
        print("THIS IS BAD. FIX ME NOW.")
        exit(1)
    
    rescale = np.median(soln/known_A[:,row])
    soln[np.abs(soln) < 1e-10] = known_A[:,row][np.abs(soln) < 1e-10] * rescale

    if CHEATING:
        other = A[LAYER][:,which_neuron]
        print("real / mine / diff")
        print(other/other[0])
        print(soln/soln[0])
        print(known_A[:,row]/known_A[:,row][0])
        print(other/other[0] - soln/soln[0])

    
    if best[1] < np.mean(np.abs(crit_val_1)) or True:
        return soln, -1
    else:
        print("FAILED TO IMPROVE ACCURACY OF ROW", row)
        print(np.mean(np.abs(crit_val_2)), 'vs', np.mean(np.abs(crit_val_1)))
        return known_A[:,row], known_B[row]


def improve_layer_precision(LAYER, known_T, known_A, known_B):
    new_A = []
    new_B = []    

    out = map(improve_row_precision,
              [(LAYER, known_T, known_A, known_B, row, False) for row in range(neuron_count[LAYER+1])])
    new_A, new_B = zip(*out)

    new_A = np.array(new_A).T
    new_B = np.array(new_B)

    print("HAVE", new_A, new_B)

    return new_A, new_B
