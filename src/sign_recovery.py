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
import jax.numpy as jnp
import scipy.linalg
import scipy.signal
import time

from src.global_vars import *
from src.utils import run, get_polytope_at, get_hidden_at, AcceptableFailure, KnownT, matmul, cheat_get_inner_layers, which_is_zero
from src.hyperplane_normal import get_ratios_lstsq, get_ratios
from src.find_witnesses import do_better_sweep


def sign_to_int(signs):
    """
    Convert a list to an integer.
    [-1, 1, 1, -1], -> 0b0110 -> 6
    """
    return int("".join('0' if x == -1 else '1' for x in signs),2)

def is_on_following_layer(known_T, known_A, known_B, point):

    print("Check if the critical point is on the next layer")
    
    def is_on_prior_layer(query):
        print("Hidden think", known_T.get_hidden_layers(query))
        if CHEATING:
            print("Hidden real", cheat_get_inner_layers(query))
        if any(np.min(np.abs(layer)) < 1e-5 for layer in known_T.get_hidden_layers(query)):
            return True
        next_hidden = known_T.extend_by(known_A, known_B).forward(query)
        print(next_hidden)
        if np.min(np.abs(next_hidden)) < 1e-4:
            return True
        return False

    if is_on_prior_layer(point):
        print("It's not, because it's on an earlier layer")
        return False

    if CHEATING:
        ls = ([np.min(np.abs(x)) for x in cheat_get_inner_layers(point)])

    initial_signs = get_polytope_at(known_T, known_A, known_B, point)

    normal = get_ratios([point], [range(DIM)], eps=GRAD_EPS)[0].flatten()
    normal = normal / np.sum(normal**2)**.5

    for tol in range(10):
    
        random_dir = np.random.normal(size=DIM)
        perp_component = np.dot(random_dir,normal)/(np.dot(normal, normal)) * normal
        parallel_dir = random_dir - perp_component
        
        go_direction = parallel_dir/np.sum(parallel_dir**2)**.5

        _, high = binary_search_towards(known_T,
                                        known_A, known_B,
                                        point,
                                        initial_signs,
                                        go_direction)

        if CHEATING:
            print(cheat_get_inner_layers(point + go_direction * high/2)[np.argmin(ls)])

        point_in_same_polytope = point + (high * .999 - 1e-4) * go_direction

        print("high", high)

        solutions = do_better_sweep(point_in_same_polytope,
                                    normal,
                                    -1e-4 * high, 1e-4 * high,
                                    known_T=known_T)
        if len(solutions) >= 1:
            print("Correctly found", len(solutions))
        else:
            return False

        point_in_different_polytope = point + (high * 1.1 + 1e-1) * go_direction

        solutions = do_better_sweep(point_in_different_polytope,
                                    normal,
                                    -1e-4 * high, 1e-4 * high,
                                    known_T=known_T)
        if len(solutions) == 0:
            print("Correctly found", len(solutions))
        else:
            return False
        
        
    #print("I THINK IT'S ON THE NEXT LAYER")
    if CHEATING:
        soln = [np.min(np.abs(x)) for x in cheat_get_inner_layers(point)]
        print(soln)
        assert np.argmin(soln) == len(known_T.A)+1
        
    return True

def find_plane_angle(known_T,
                     known_A, known_B,
                     multiple_intersection_point,
                     sign_at_init,
                     init_step,
                     exponential_base=1.5):
    """
    Given an input that's at the multiple intersection point, figure out how
    to continue along the path after it bends.


                /       X    : multiple intersection point
       ......../..      ---- : layer N hyperplane
       .      /  .       |   : layer N+1 hyperplane that bends
       .     /   .    
    --------X-----------
       .    |    .
       .    |    .
       .....|.....
            |
            |

    We need to make sure to bend, and not turn onto the layer N hyperplane.

    To do this we will draw a box around the X and intersect with the planes 
    and determine the four coordinates. Then draw another box twice as big.
    
    The first layer plane will be the two points at a consistent angle.
    The second layer plane will have an inconsistent angle.

    Choose the inconsistent angle plane, and make sure we move to a new
    polytope and don't just go backwards to where we've already bene.
    """
    success = None
    camefrom = None

    prev_iter_intersections = []

    while True:
        x_dir_base = np.sign(np.random.normal(size=DIM))/DIM**.5
        y_dir_base = np.sign(np.random.normal(size=DIM))/DIM**.5
        # When the input dimension is odd we can't have two orthogonal
        # vectors from {-1,1}^DIM
        if np.abs(np.dot(x_dir_base, y_dir_base)) <= DIM%2 + 1e-8:
            break

    MAX = 35

    start = [10] if init_step > 10 else []
    for stepsize in start + list(range(init_step, MAX)):
        print("\tTry stepping away", stepsize)
        x_dir = x_dir_base * (exponential_base**(stepsize-10))
        y_dir = y_dir_base * (exponential_base**(stepsize-10))
                
        # Draw the box as shown in the diagram above, and compute where
        # the critical points are.
        top = do_better_sweep(multiple_intersection_point + x_dir,
                              y_dir, -1, 1,
                              known_T=known_T)
        bot = do_better_sweep(multiple_intersection_point - x_dir,
                              y_dir, -1, 1,
                              known_T=known_T)
        left = do_better_sweep(multiple_intersection_point + y_dir,
                               x_dir, -1, 1,
                              known_T=known_T)
        right = do_better_sweep(multiple_intersection_point - y_dir,
                                x_dir, -1, 1,
                              known_T=known_T)

        intersections = top + bot + left + right

        # If we only have two critical points, and we're taking a big step,
        # then something is seriously messed up.
        # This is not an acceptable error. Just abort out and let's try to
        # do the whole thing again.
        if len(intersections) == 2 and stepsize >= 10:
            raise AcceptableFailure()

        if CHEATING:
            print("\tHAVE BOX INTERSECT COUNT", len(intersections))
            print("\t",len(left), len(right), len(top), len(bot))

        if (len(intersections) == 0 and stepsize > 15):# or (len(intersections) == 3 and stepsize > 5):
            # Probably we're in just a constant flat 0 region
            # At this point we're basically dead in the water.
            # Just fail up and try again.
            print("\tIt looks like we're in a flat region, raise failure")
            raise AcceptableFailure()

        # If we somehow went from almost no critical points to more than 4,
        # then we've really messed up.
        # Just fail out and let's hope next time it doesn't happen.
        if len(intersections) > 4 and len(prev_iter_intersections) < 2:
            print("\tWe didn't get enough inner points")
            if exponential_base == 1.2:
                print("\tIt didn't work a second time")
                return None, None, 0
            else:
                print("\tTry with smaller step")
                return find_plane_angle(known_T,
                                        known_A, known_B,
                                        multiple_intersection_point,
                                        sign_at_init,
                                        init_step,
                                        exponential_base=1.2)

        # This is the good, expected code-path.
        # We've seen four intersections at least twice before, and now
        # we're seeing more than 4.
        if (len(intersections) > 4 or stepsize > 20) and len(prev_iter_intersections) >= 2:
            next_intersections = np.array(prev_iter_intersections[-1])
            intersections = np.array(prev_iter_intersections[-2])

            # Let's first figure out what points are responsible for the prior-layer neurons
            # being zero, and which are from the current-layer neuron being zero
            candidate = []
            for i,a in enumerate(intersections):
                for j,b in enumerate(intersections):
                    if i == j: continue
                    score = np.sum(((a+b)/2-multiple_intersection_point)**2)
                    a_to_b = b-a
                    a_to_b /= np.sum(a_to_b**2)**.5

                    variance = np.std((next_intersections-a)/a_to_b,axis=1)
                    best_variance = np.min(variance)

                    #print(i,j,score, best_variance)

                    candidate.append((best_variance, i, j))

            if sorted(candidate)[3][0] < 1e-8:
                # It looks like both lines are linear here
                # We can't distinguish what way is the next best way to go.
                print("\tFailed the box continuation finding procedure. (1)")
                print("\t",candidate)
                raise AcceptableFailure()

            # Sometimes life is just ugly, and nothing wants to work.
            # Just abort.
            err, index_0, index_1 = min(candidate)
            if err/max(candidate)[0] > 1e-5:
                return None, None, 0

            prior_layer_near_zero = np.zeros(4, dtype=np.bool)
            prior_layer_near_zero[index_0] = True
            prior_layer_near_zero[index_1] = True

            # Now let's walk through each of these points and check that everything looks sane.
            should_fail = False
            for critical_point, is_prior_layer_zero in zip(intersections,prior_layer_near_zero):
                vs = known_T.extend_by(known_A,known_B).get_hidden_layers(critical_point)
                #print("VS IS", vs)
                #print("Is prior", is_prior_layer_zero)
                #if CHEATING:
                #    print(cheat_get_inner_layers(critical_point))

                if is_prior_layer_zero:
                    # We expect the prior layer to be zero.
                    if all([np.min(np.abs(x)) > 1e-5 for x in vs]):
                        # If it looks like it's not actually zero, then brutally fail.
                        print("\tAbort 1: failed to find a valid box")
                        should_fail = True
                if any([np.min(np.abs(x)) < 1e-10 for x in vs]):
                    # We expect the prior layer to be zero.
                    if not is_prior_layer_zero:
                        # If it looks like it's not actually zero, then brutally fail.
                        print("\tAbort 2: failed to find a valid box")
                        should_fail = True
            if should_fail:
                return None, None, 0
                
                

            # Done with error checking, life is good here.
            # Find the direction that corresponds to the next direction we can move in
            # and continue our search from that point.
            for critical_point, is_prior_layer_zero in zip(intersections,prior_layer_near_zero):
                sign_at_crit = sign_to_int(get_polytope_at(known_T,
                                                           known_A, known_B,
                                                           critical_point))
                print("\tMove to", sign_at_crit, 'versus', sign_at_init, is_prior_layer_zero)
                if not is_prior_layer_zero:
                    if sign_at_crit != sign_at_init:
                        success = critical_point
                        if CHEATING:
                            print('\tinner at success', cheat_get_inner_layers(success))
                        print("\tSucceeded")
                    else:
                        camefrom = critical_point

            # If we didn't get a solution, then abort out.
            # Probably what happened here is that we got more than four points
            # on the box but didn't see exactly four points on the box twice before
            # this means we should decrease the initial step size and try again.
            if success is None:
                print("\tFailed the box continuation finding procedure. (2)")
                raise AcceptableFailure()
                #assert success is not None
            break
        if len(intersections) == 4:
            prev_iter_intersections.append(intersections)
    return success, camefrom, min(stepsize, MAX-3)

def binary_search_towards_slow(known_T,  known_A, known_B, start_point, initial_signs, go_direction, maxstep=1e6):
    low = 0
    high = maxstep
    while high-low > 1e-8:
        mid = (high+low)/2
        query_point = start_point + mid * go_direction
        
        next_signs = get_polytope_at(known_T, known_A, known_B,
                                     query_point)
        if initial_signs == next_signs:
            low = mid
        else:
            high = mid


    #print('check',np.abs(mid - can_go_dist))

    next_signs = get_polytope_at(known_T, known_A, known_B,
                                 start_point + low * go_direction)
    if next_signs != initial_signs:
        # It is extremely unlikely, but possible, for us to end up
        # skipping over the region of interest.
        # If this happens then don't step as far and try again.
        # This has only ever happend once, but just in case....
        print("Well this is awkward")
        return binary_search_towards(known_T,  known_A, known_B, start_point, initial_signs, go_direction, maxstep=maxstep/10)
    

    # If mid is at the end, it means it never binary searched.
    if mid > 1e6-1:
        return None, None
    else:
        a_bit_further = start_point + (high+1e-4)*go_direction
        return a_bit_further, high


PREV_GRAD = None
    
def binary_search_towards(known_T,  known_A, known_B, start_point, initial_signs, go_direction, maxstep=1e6):
    """
    Compute how far we can walk along the hyperplane until it is in a
    different polytope from a prior layer.

    It is okay if it's in a differnt polytope in a *later* layer, because
    it will still have the same angle.

    (but do it analytically by looking at the signs of the first layer)
    this requires no queries and could be done with math but instead
    of thinking I'm just going to run binary search.
    """
    global PREV_GRAD

    #_, slow_ans = binary_search_towards_slow(known_T,  known_A, known_B, start_point, initial_signs, go_direction, maxstep)
                               
    plus_T = known_T.extend_by(known_A, known_B)
    # this is the hidden state
    initial_hidden = np.array(plus_T.get_hidden_layers(start_point, flat=True))
    delta_hidden_np = (np.array(plus_T.get_hidden_layers(start_point + 1e-6 * go_direction, flat=True)) - initial_hidden) * 1e6
    #
    #can_go_dist_all = initial_hidden / delta_hidden

    if PREV_GRAD is None or PREV_GRAD[0] is not known_T or PREV_GRAD[1] is not known_A or PREV_GRAD[2] is not known_B:
        def get_grad(x, i):
            initial_hidden = plus_T.get_hidden_layers(x, flat=True, np=jnp)
            return initial_hidden[i]
        g = jax.jit(jax.grad(get_grad))


        def grads(start_point, go_direction):
            return jnp.array([jnp.dot(g(start_point, i), go_direction) for i in range(initial_hidden.shape[0])])
        
        PREV_GRAD = (known_T, known_A, known_B, jax.jit(grads))
    else:
        grads = PREV_GRAD[3]
    
    delta_hidden = grads(start_point, go_direction)

    can_go_dist_all = np.array(initial_hidden / delta_hidden)
    
    can_go_dist = -can_go_dist_all[can_go_dist_all<0]

    if len(can_go_dist) == 0:
        print("Can't go anywhere at all")
        raise AcceptableFailure()

    can_go_dist = np.min(can_go_dist)
    
    a_bit_further = start_point + (can_go_dist+1e-4)*go_direction
    return a_bit_further, can_go_dist

def follow_hyperplane(LAYER, start_point, known_T, known_A, known_B,
                      history=[], MAX_POINTS=1e3, only_need_positive=False):
    """
    This is the ugly algorithm that will let us recover sign for expansive networks.
    Assumes we have extracted up to layer K-1 correctly, and layer K up to sign.

    start_point is a neuron on layer K+1

    known_T is the transformation that computes up to layer K-1, with
    known_A and known_B being the layer K matrix up to sign.

    We're going to come up with a bunch of different inputs,
    each of which has the same critical point held constant at zero.
    """

    def choose_new_direction_from_minimize(previous_axis):
        """
        Given the current point which is at a critical point of the next
        layer neuron, compute which direction we should travel to continue
        with finding more points on this hyperplane.

        Our goal is going to be to pick a direction that lets us explore
        a new part of the space we haven't seen before.
        """

        print("Choose a new direction to travel in")
        if len(history) == 0:
            which_to_change = 0
            new_perp_dir = perp_dir
            new_start_point = start_point
            initial_signs = get_polytope_at(known_T, known_A, known_B, start_point)

            # If we're in the 1 region of the polytope then we try to make it smaller
            # otherwise make it bigger
            fn = min if initial_signs[0] == 1 else max
        else:
            neuron_values = np.array([x[1] for x in history])

            neuron_positive_count = np.sum(neuron_values>1,axis=0)
            neuron_negative_count = np.sum(neuron_values<-1,axis=0)

            mean_plus_neuron_value = neuron_positive_count/(neuron_positive_count + neuron_negative_count + 1)
            mean_minus_neuron_value = neuron_negative_count/(neuron_positive_count + neuron_negative_count + 1)

            # we want to find values that are consistently 0 or 1
            # So map 0 -> 0 and 1 -> 0 and the middle to higher values
            if only_need_positive:
                neuron_consistency = mean_plus_neuron_value
            else:
                neuron_consistency = mean_plus_neuron_value * mean_minus_neuron_value

            # Print out how much progress we've made.
            # This estimate is probably worse than Windows 95's estimated time remaining.
            # At least it's monotonic. Be thankful for that.
            print("Progress", "%.1f"%int(np.mean(neuron_consistency!=0)*100)+"%")
            print("Counts on each side of each neuron")
            print(neuron_positive_count)
            print(neuron_negative_count)

                
            # Choose the smallest value, which is the most consistent
            which_to_change = np.argmin(neuron_consistency)
            
            print("Try to explore the other side of neuron", which_to_change)

            if which_to_change != previous_axis:
                if previous_axis is not None and neuron_consistency[previous_axis] == neuron_consistency[which_to_change]:
                    # If the previous thing we were working towards has the same value as this one
                    # the don't change our mind and just keep going at that one
                    # (almost always--sometimes we can get stuck, let us get unstuck)
                    which_to_change = previous_axis
                    new_start_point = start_point
                    new_perp_dir = perp_dir
                else:
                    valid_axes = np.where(neuron_consistency == neuron_consistency[which_to_change])[0]

                    best = (np.inf, None, None)

                    for _, potential_hidden_vector, potential_point in history[-1:]:
                        for potential_axis in valid_axes:
                            value = potential_hidden_vector[potential_axis]
                            if np.abs(value) < best[0]:
                                best = (np.abs(value), potential_axis, potential_point)

                    _, which_to_change, new_start_point = best
                    new_perp_dir = perp_dir
                    
            else:
                new_start_point = start_point
                new_perp_dir = perp_dir


            # If we're in the 1 region of the polytope then we try to make it smaller
            # otherwise make it bigger
            fn = min if neuron_positive_count[which_to_change] > neuron_negative_count[which_to_change] else max
            arg_fn = np.argmin if neuron_positive_count[which_to_change] > neuron_negative_count[which_to_change] else np.argmax
            print("Changing", which_to_change, 'to flip sides because mean is', mean_plus_neuron_value[which_to_change])


        val = matmul(known_T.forward(new_start_point, with_relu=True), known_A, known_B)[which_to_change]

        initial_signs = get_polytope_at(known_T, known_A, known_B, new_start_point)

        # Now we're going to figure out what direction makes this biggest/smallest
        # this doesn't take any queries
        # There's probably an analytical way to do this.
        # But thinking is hard. Just try 1000 random angles.
        # There are no queries involved in this process.

        choices = []
        for _ in range(1000):
            random_dir = np.random.normal(size=DIM)
            perp_component = np.dot(random_dir,new_perp_dir)/(np.dot(new_perp_dir, new_perp_dir)) * new_perp_dir
            parallel_dir = random_dir - perp_component

            # This is the direction we're going to travel in.
            go_direction = parallel_dir/np.sum(parallel_dir**2)**.5

            try:
                a_bit_further, high = binary_search_towards(known_T,
                                                            known_A, known_B,
                                                            new_start_point,
                                                            initial_signs,
                                                            go_direction)
            except AcceptableFailure:
                continue
            if a_bit_further is None:
                continue
            
            # choose a direction that makes the Kth value go down by the most
            val = matmul(known_T.forward(a_bit_further[np.newaxis,:], with_relu=True), known_A, known_B)[0][which_to_change]

            #print('\t', val, high)

            choices.append([val,
                            new_start_point + high*go_direction])


        best_value, multiple_intersection_point = fn(choices, key=lambda x: x[0])
        
        print('Value', best_value)
        return new_start_point, multiple_intersection_point, which_to_change

    ###################################################
    ### Actual code to do the sign recovery starts. ###
    ###################################################

    start_box_step = 0
    points_on_plane = []

    if CHEATING:
        layer = np.abs(cheat_get_inner_layers(np.array(start_point))[LAYER+1]) 
        print("Layer", layer)
        which_is_zero = np.argmin(layer)

    current_change_axis = 0
    
    while True:
        print("\n\n")
        print("-----"*10)

        if CHEATING:
            layer = np.abs(cheat_get_inner_layers(np.array(start_point))[LAYER+1])
            #print('layer',LAYER+1, layer)
            #print('all inner layers')
            #for e in cheat_get_inner_layers(np.array(start_point)):
            #    print(e)
            which_is_zero_2 = np.argmin(np.abs(layer))

            if which_is_zero_2 != which_is_zero:
                print("STARTED WITH", which_is_zero, "NOW IS", which_is_zero_2)
                print(layer)
                raise

        # Keep track of where we've been, so we can go to new places.
        which_polytope = get_polytope_at(known_T, known_A, known_B, start_point, False) # [-1 1 -1]
        hidden_vector = get_hidden_at(known_T, known_A, known_B, LAYER, start_point, False)
        sign_at_init = sign_to_int(which_polytope) # 0b010 -> 2

        print("Number of collected points", len(points_on_plane))
        if len(points_on_plane) > MAX_POINTS:
            return points_on_plane, False

        neuron_values = np.array([x[1] for x in history])

        neuron_positive_count = np.sum(neuron_values>1,axis=0)
        neuron_negative_count = np.sum(neuron_values<-1,axis=0)

        if (np.all(neuron_positive_count > 0) and np.all(neuron_negative_count > 0)) or \
           (only_need_positive and np.all(neuron_positive_count > 0)):
            print("Have all the points we need (1)")
            print(query_count)
            print(neuron_positive_count)
            print(neuron_negative_count)

            neuron_values = np.array([get_hidden_at(known_T, known_A, known_B, LAYER, x, False) for x in points_on_plane])
            
            neuron_positive_count = np.sum(neuron_values>1,axis=0)
            neuron_negative_count = np.sum(neuron_values<-1,axis=0)

            print(neuron_positive_count)
            print(neuron_negative_count)
            
            return points_on_plane, True
    
        # 1. find a way to move along the hyperplane by computing the normal
        # direction using the ratios function. Then find a parallel direction.
        
        try:
            #perp_dir = get_ratios([start_point], [range(DIM)], eps=1e-4)[0].flatten()
            perp_dir = get_ratios_lstsq(0, [start_point], [range(DIM)], KnownT([], []), eps=1e-5)[0].flatten()

        except AcceptableFailure:
            print("Failed to compute ratio at start point. Something very bad happened.")
            return points_on_plane, False

        # Record these points.
        history.append((which_polytope,
                        hidden_vector,
                        np.copy(start_point)))
        
        # We can't just pick any parallel direction. If we did, then we would
        # not end up covering much of the input space.
    
        # Instead, we're going to figure out which layer-1 hyperplanes are "visible"
        # from the current point. Then we're going to try and go reach all of them.
    
        # This is the point at which the first and second layers intersect.
        start_point, multiple_intersection_point, new_change_axis = choose_new_direction_from_minimize(current_change_axis)
        
        if new_change_axis != current_change_axis:
            start_point, multiple_intersection_point, current_change_axis = choose_new_direction_from_minimize(None)

        #if CHEATING:
        #    print("INIT MULTIPLE", cheat_get_inner_layers(multiple_intersection_point))
            
        # Refine the direction we're going to travel in---stay numerically stable.
        towards_multiple_direction = multiple_intersection_point - start_point
        step_distance = np.sum(towards_multiple_direction**2)**.5

        print("Distance we need to step:", step_distance)        
        
        if step_distance > 1 or True:
            mid_point = 1e-4 * towards_multiple_direction/np.sum(towards_multiple_direction**2)**.5 + start_point

            random_dir = np.random.normal(size=DIM)
            
            mid_points = do_better_sweep(mid_point, perp_dir/np.sum(perp_dir**2)**.5,
                                         low=-1e-3,
                                         high=1e-3,
                                         known_T=known_T)

            if len(mid_points) > 0:
                mid_point = mid_points[np.argmin(np.sum((mid_point-mid_points)**2,axis=1))]

                towards_multiple_direction = mid_point - start_point
                towards_multiple_direction = towards_multiple_direction/np.sum(towards_multiple_direction**2)**.5

                initial_signs = get_polytope_at(known_T, known_A, known_B, start_point)
                _, high = binary_search_towards(known_T,
                                                known_A, known_B,
                                                start_point,
                                                initial_signs,
                                                towards_multiple_direction)
                
                multiple_intersection_point = towards_multiple_direction * high + start_point
                    
    
        # Find the angle of the next hyperplane
        # First, take random steps away from the intersection point
        # Then run the search algorithm to find some intersections
        # what we find will either be a layer-1 or layer-2 intersection.

        print("Now try to find the continuation direction")
        success = None
        while success is None:
            if start_box_step < 0:
                start_box_step = 0
                print("VERY BAD FAILURE")
                print("Choose a new random point to start from")
                which_point = np.random.randint(0, len(history))
                start_point = history[which_point][2]
                print("New point is", which_point)
                current_change_axis = np.random.randint(0, sizes[LAYER+1])
                print("New axis to change", current_change_axis)
                break

            print("\tStart the box step with size", start_box_step)
            try:
                success, camefrom, stepsize = find_plane_angle(known_T,
                                                               known_A, known_B,
                                                               multiple_intersection_point,
                                                               sign_at_init,
                                                               start_box_step)
            except AcceptableFailure:
                # Go back to the top and try with a new start point
                print("\tOkay we need to try with a new start point")
                start_box_step = -10

            start_box_step -= 2

        if success is None:
            continue

        val = matmul(known_T.forward(multiple_intersection_point, with_relu=True), known_A, known_B)[new_change_axis]
        print("Value at multiple:", val)
        val = matmul(known_T.forward(success, with_relu=True), known_A, known_B)[new_change_axis]
        print("Value at success:", val)

        if stepsize < 10:
            new_move_direction = success - multiple_intersection_point
    
            # We don't want to be right next to the multiple intersection point.
            # So let's binary search to find how far away we can go while remaining in this polytope.
            # Then we'll go half as far as we can maximally go.
            
            initial_signs = get_polytope_at(known_T, known_A, known_B, success)
            print("polytope at initial", sign_to_int(initial_signs))        
            low = 0
            high = 1
            while high-low > 1e-2:
                mid = (high+low)/2
                query_point = multiple_intersection_point + mid * new_move_direction
                next_signs = get_polytope_at(known_T, known_A, known_B, query_point)
                print("polytope at", mid, sign_to_int(next_signs), "%x"%(sign_to_int(next_signs)^sign_to_int(initial_signs)))
                if initial_signs == next_signs:
                    low = mid
                else:
                    high = mid
            print("GO TO", mid)
        
            success = multiple_intersection_point + (mid/2) * new_move_direction
    
            val = matmul(known_T.forward(success, with_relu=True), known_A, known_B)[new_change_axis]
            print("Value at moved success:", val)
        
        print("Adding the points to the set of known good points")

        points_on_plane.append(start_point)
            
        if camefrom is not None:
            points_on_plane.append(camefrom)
        #print("Old start point", start_point)
        #print("Set to success", success)
        start_point = success
        start_box_step = max(stepsize-1,0)
        
    return points_on_plane, False

def is_solution_map(args):
    bounds, extra_tuple = args
    r = []
    for i in range(bounds[0], bounds[1]):
        r.append(is_solution((i, extra_tuple)))
    return r


def is_solution(input_tuple):
    signs, (known_A0, known_B0, LAYER, known_hidden_so_far, K, responses) = input_tuple
    new_signs = np.array([-1 if x == '0' else 1 for x in bin((1<<K)+signs)[3:]])

    if CHEATING:
        if signs%1001 == 0:
            print('tick',signs)
    else:
        if signs%100001 == 0:
            # This isn't cheating, but makes things prettier
            print('tick',signs)

    guess_A0 = known_A0 * new_signs
    guess_B0 = known_B0 * new_signs
    # We're going to set up a system of equations here
    # The matrix is going to have a bunch of rows (equal to number of equations)
    # each row is of the form
    # [h_0 h_1 h_2 h_3 h_4 ... h_n 1]
    # where h_n is the hidden vector after multiplying by the guessed matrix.
    # and 1 is the weight for the bias term
        
    inputs = matmul(known_hidden_so_far, guess_A0, guess_B0)
    inputs[inputs < 0] = 0

    if responses is None:
        responses = np.ones((inputs.shape[0], 1))
    else:
        inputs = np.concatenate([inputs, np.ones((inputs.shape[0],1))], axis=1)
        pass
        
    solution, res, _, _ = scipy.linalg.lstsq(inputs, responses)

    bias = np.dot(inputs, solution)-responses
    res = np.std(bias)

    #print("Recovered vector", solution.flatten())

    if res > 1e-2:
        return (res, new_signs, solution), 0

    bias = bias.mean(axis=0)

    #solution = np.concatenate([solution, [-bias]])[:, np.newaxis]
    mat = (solution/solution[0][0])[:-1,:]
    if np.any(np.isnan(mat)) or np.any(np.isinf(mat)):
        print("Invalid solution")
        return (res, new_signs, solution), 0
    else:
        s = solution/solution[0][0]
        s[np.abs(s)<1e-14] = 0

        return (res, new_signs, solution), 1

def solve_contractive_sign(known_T, weight, bias, LAYER):

    print("Solve the extraction problem for contractive networks")
        
    def get_preimage(hidden):
        preimage = hidden
        
        for i,(my_A,my_B) in reversed(list(enumerate(zip(known_T.A+[weight], known_T.B+[bias])))):
            if i == 0:
                res = scipy.optimize.lsq_linear(my_A.T, preimage-my_B,
                                                bounds=(-np.inf, np.inf))
            else:
                res = scipy.optimize.lsq_linear(my_A.T, preimage-my_B,
                                                bounds=(0, np.inf))
            
            preimage = res.x
        return preimage[np.newaxis,:]

    hidden = np.zeros((sizes[LAYER+1]))

    preimage = get_preimage(hidden)

    extended_T = known_T.extend_by(weight,bias)
    
    standard_out = run(preimage)

    signs = []

    for axis in range(len(hidden)):
        h = np.array(hidden)
        h[axis] = 10
        preimage_plus = get_preimage(h)
        h[axis] = -10
        preimage_minus = get_preimage(h)

        print("Confirm preimage")

        if np.any(extended_T.forward(preimage) > 1e-5):
            raise AcceptableFailure()
        
        out_plus = run(preimage_plus)
        out_minus = run(preimage_minus)

        print(standard_out, out_plus, out_minus)

        inverted_if_small = np.sum(np.abs(out_plus-standard_out))
        not_inverted_if_small = np.sum(np.abs(out_minus-standard_out))

        print("One of these should be small",
              inverted_if_small,
              not_inverted_if_small)

        if inverted_if_small < not_inverted_if_small:
            signs.append(-1)
        else:
            signs.append(1)
    return signs

def solve_layer_sign(known_T, known_A0, known_B0, critical_points, LAYER,
                     already_checked_critical_points=False,
                     only_need_positive=False, l1_mask=None):
    """
    Compute the signs for one layer of the network.

    known_T is the transformation that computes up to layer K-1, with
    known_A and known_B being the layer K matrix up to sign.
    """
    
    def get_critical_points():
        print("Init")
        print(critical_points)
        for point in critical_points:
            print("Tick")
            if already_checked_critical_points or is_on_following_layer(known_T, known_A0, known_B0, point):
                print("Found layer N point at ", point, already_checked_critical_points)
                yield point

    get_critical_point = get_critical_points()


    print("Start looking for critical point")
    MAX_POINTS = 200
    which_point = next(get_critical_point)
    print("Done looking for critical point")

    initial_points = []
    history = []
    pts = []
    if already_checked_critical_points:
        for point in get_critical_point:
            initial_points.append(point)
            pts.append(point)
            which_polytope = get_polytope_at(known_T, known_A0, known_B0, point, False) # [-1 1 -1]
            hidden_vector = get_hidden_at(known_T, known_A0, known_B0, LAYER, point, False)
            if CHEATING:
                layers = cheat_get_inner_layers(point)
                print('have',[(np.argmin(np.abs(x)),np.min(np.abs(x))) for x in layers])
            history.append((which_polytope,
                            hidden_vector,
                            np.copy(point)))
    

    while True:
        if not already_checked_critical_points:
            history = []
            pts = []

        prev_count = -10
        good = False
        while len(pts) > prev_count+2:
            print("======"*10)
            print("RESTART SEARCH", len(pts), prev_count)
            print(which_point)
            prev_count = len(pts)
            more_points, done = follow_hyperplane(LAYER, which_point,
                                        known_T,
                                        known_A0, known_B0,
                                        history=history,
                                         only_need_positive=only_need_positive)
            pts.extend(more_points)
            if len(pts) >= MAX_POINTS:
                print("Have enough; break")
                break

            if len(pts) == 0:
                break

            neuron_values = known_T.extend_by(known_A0, known_B0).forward(pts)                
            
            neuron_positive_count = np.sum(neuron_values>1,axis=0)
            neuron_negative_count = np.sum(neuron_values<-1,axis=0)
            print("Counts")
            print(neuron_positive_count)
            print(neuron_negative_count)

            print("SHOULD BE DONE?", done, only_need_positive)
            if done and only_need_positive:
                good = True
                break
            if np.all(neuron_positive_count > 0) and np.all(neuron_negative_count > 0) or \
               (only_need_positive and np.all(neuron_positive_count > 0)):
                print("Have all the points we need (2)")
                good = True
                break
            
        if len(pts) < MAX_POINTS/2 and good == False:
            print("======="*10)
            print("Select a new point to start from")
            print("======="*10)
            if already_checked_critical_points:
                print("CHOOSE FROM", len(initial_points), initial_points)
                which_point = initial_points[np.random.randint(0,len(initial_points)-1)]
            else:
                which_point = next(get_critical_point)
        else:
            print("Abort")
            break
            
    critical_points = np.array(pts)#sorted(list(set(map(tuple,pts))))


    print("Now have critical points", len(critical_points))

    if CHEATING:
        layer = [[np.min(np.abs(x)) for x in cheat_get_inner_layers(x[np.newaxis,:])][LAYER+1] for x in critical_points]
    
        #print("Which layer is zero?", sorted(layer))
        layer = np.abs(cheat_get_inner_layers(np.array(critical_points))[LAYER+1])

        print(layer)
        
        which_is_zero = np.argmin(layer,axis=1)
        print("Which neuron is zero?", which_is_zero)

        which_is_zero = which_is_zero[0]

    print("Query count", query_count)
        

    K = neuron_count[LAYER+1]
    MAX = (1<<K)
    if already_checked_critical_points:
        bounds = [(MAX-1, MAX)]
    else:
        bounds = []
        for i in range(1024):
            bounds.append(((MAX*i)//1024, (MAX*(i+1))//1024))

    print("Created a list")
    
    known_hidden_so_far = known_T.forward(critical_points, with_relu=True)
    debug = False

    start_time = time.time()

    extra_args_tup = (known_A0, known_B0, LAYER, known_hidden_so_far, K, None)

    all_res = pool[0].map(is_solution_map, [(bound, extra_args_tup) for bound in bounds])

    end_time = time.time()

    print("Done map, now collect results")
    print("Took", end_time-start_time, 'seconds')

    all_res = [x for y in all_res for x in y]
    
    scores = [r[0] for r in all_res]
    solution_attempts = sum([r[1] for r in all_res])
    total_attempts = len(all_res)
    
    print("Attempts at solution:", (solution_attempts), 'out of', total_attempts)

    
    std = np.std([x[0] for x in scores])
    print('std',std)
    print('median', np.median([x[0] for x in scores]))
    print('min', np.min([x[0] for x in scores]))


    return min(scores,key=lambda x: x[0])[1], critical_points
