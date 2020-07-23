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
import jax.numpy as jnp
import numpy as np
import time
import networkx as nx
import collections

from src.global_vars import *
from src.find_witnesses import do_better_sweep
from src.hyperplane_normal import get_ratios_lstsq

from src.utils import AcceptableFailure, GatherMoreData, matmul, KnownT, cheat_get_inner_layers, which_is_zero
import src.sign_recovery as sign_recovery


@jax.jit
def process_block(ratios, other_ratios):
    """
    Let jax efficiently compute pairwise similarity by blocking things.
    """
    differences = jnp.abs(ratios[:,jnp.newaxis,:] - other_ratios[jnp.newaxis,:,:])
    differences = differences / jnp.abs(ratios[:,jnp.newaxis,:]) + differences / jnp.abs(other_ratios[jnp.newaxis,:,:])

    close = differences < BLOCK_ERROR_TOL * jnp.log(ratios.shape[1])

    pairings = jnp.sum(close, axis=2) >= max(MIN_SAME_SIZE,BLOCK_MULTIPLY_FACTOR*(jnp.log(ratios.shape[1])-2))

    return pairings

def graph_solve(all_ratios, all_criticals, expected_neurons, LAYER, debug=False):
    # 1. Load the critical points and ratios we precomputed

    all_ratios = np.array(all_ratios, dtype=np.float64)
    all_ratios_f32 = np.array(all_ratios, dtype=np.float32)
    all_criticals = np.array(all_criticals, dtype=np.float64)

    # Batch them to be sensibly sized
    ratios_group = [all_ratios_f32[i:i+1000] for i in range(0,len(all_ratios),1000)]
    criticals_group = [all_criticals[i:i+1000] for i in range(0,len(all_criticals),1000)]
                    
    # 2. Compute the similarity pairwise between the ratios we've computed

    print("Go up to", len(criticals_group))
    now = time.time()
    all_pairings = [[] for _ in range(sum(map(len,ratios_group)))]
    for batch_index,(criticals,ratios) in enumerate(zip(criticals_group, ratios_group)):
        print(batch_index)

        # Compute the all-pairs similarity
        axis = list(range(all_ratios.shape[1]))
        random.shuffle(axis)
        axis = axis[:20]
        for dim in axis:
            # We may have an error on one of the directions, so let's try all of them
            scaled_all_ratios =  all_ratios_f32 / all_ratios_f32[:,dim:dim+1]
            scaled_ratios = ratios / ratios[:,dim:dim+1]

            batch_pairings = process_block(scaled_ratios, scaled_all_ratios)
            
            # To get the offset, Compute the cumsum of the length up to batch_index
            batch_offset = sum(map(len,ratios_group[:batch_index]))
            # And now create the graph matching ratios that are similar
            for this_batch_i,global_j in zip(*np.nonzero(np.array(batch_pairings))):
                all_pairings[this_batch_i + batch_offset].append(global_j)
    print(time.time()-now)

    graph = nx.Graph()
    # Add the edges to the graph, removing self-loops
    graph.add_edges_from([(i,j) for i,js in enumerate(all_pairings) for j in js if abs(i-j) > 1]) 
    components = list(nx.connected_components(graph))

    sorted_components = sorted(components, key=lambda x: -len(x))

    if CHEATING:
        print('Total (unmatched) examples found:', sorted(collections.Counter(which_is_zero(LAYER, cheat_get_inner_layers(all_criticals))).items()))

        #for crit,rat in zip(all_criticals,all_ratios):
        #    if which_is_zero(LAYER, cheat_get_inner_layers(crit)) == 6:
        #        print(" ".join("%.6f"%abs(x) if not np.isnan(x) else "     nan" for x in rat))
            
        #cc = which_is_zero(LAYER, cheat_get_inner_layers(all_criticals))
        #print("THREES")
        #
        #threes = []
        #print("Pair", process_block
        #      [all_ratios[x] for x in range(len(all_criticals)) if cc[x] == 3]

            

    if len(components) == 0:
        print("No components found")
        raise AcceptableFailure()
    print("Graph search found", len(components), "different components with the following counts", list(map(len,sorted_components)))

    if CHEATING:
        which_neurons = [collections.Counter(which_is_zero(LAYER, cheat_get_inner_layers(all_criticals[list(orig_component)]))) for orig_component in sorted_components]
        first_index_of = [-1]*expected_neurons

        for i,items in enumerate(which_neurons):
            for item in items.keys():
                if first_index_of[item] == -1:
                    first_index_of[item] = i

        print('These components corresopnd to', which_neurons)
        print("Withe the corresponding index in the list:", first_index_of)

    previous_num_components = np.inf
    
    while previous_num_components > len(sorted_components):
        previous_num_components = len(sorted_components)
        candidate_rows = []
        candidate_components = []

        datas = [all_ratios[list(component)] for component in sorted_components]
        results = pool[0].map(ratio_normalize, datas)

        candidate_rows = [x[0] for x in results]
        candidate_components = sorted_components

        candidate_rows = np.array(candidate_rows)

        new_pairings = [[] for _ in range(len(candidate_rows))]
        
        # Re-do the pairings
        for dim in range(all_ratios.shape[1]):
            scaled_ratios = candidate_rows / candidate_rows[:,dim:dim+1]

            batch_pairings = process_block(scaled_ratios, scaled_ratios)
            
            # And now create the graph matching ratios that are similar
            for this_batch_i,global_j in zip(*np.nonzero(np.array(batch_pairings))):
                new_pairings[this_batch_i].append(global_j)
            
        graph = nx.Graph()
        # Add the edges to the graph, ALLOWING self-loops this time
        graph.add_edges_from([(i,j) for i,js in enumerate(new_pairings) for j in js]) 
        components = list(nx.connected_components(graph))

        components = [sum([list(candidate_components[y]) for y in comp],[]) for comp in components]

        sorted_components = sorted(components, key=lambda x: -len(x))

        print("After re-doing the graph, the component counts is", len(components), "with items", list(map(len,sorted_components)))

        if CHEATING:
            which_neurons = [collections.Counter(which_is_zero(LAYER, cheat_get_inner_layers(all_criticals[list(orig_component)]))) for orig_component in sorted_components]
            first_index_of = [-1]*expected_neurons
        
            for i,items in enumerate(which_neurons):
                for item in items.keys():
                    if first_index_of[item] == -1:
                        first_index_of[item] = i
            
            print('Corresponding to', which_neurons)
            print("First index:", first_index_of)
        
            print("Expected neurons", expected_neurons)


    print("Processing each connected component in turn.")
            
    resulting_examples = []
    resulting_rows = []

    skips_because_of_nan = 0
    failure = None
    
    for c_count, component in enumerate(sorted_components):
        if debug:
            print("\n")
            if c_count >= expected_neurons:
                print("WARNING: This one might be a duplicate!")
        print("On component", c_count, "with indexs", component)
        if debug and CHEATING:
            inner = cheat_get_inner_layers(all_criticals[list(component)])
            print('Corresponding to (cheating) ', which_is_zero(LAYER, inner))

        possible_matrix_rows = all_ratios[list(component)]
        
        guessed_row, normalize_axis, normalize_error = ratio_normalize(possible_matrix_rows)

        print('The guessed error in the computation is',normalize_error, 'with', len(component), 'witnesses')
        if normalize_error > .01 and len(component) <= 5:
            print("Component size less than 5 with high error; this isn't enough to be sure")
            continue
        
        print("Normalize on axis", normalize_axis)

        if len(resulting_rows):
            scaled_resulting_rows = np.array(resulting_rows)
            #print(scaled_resulting_rows.shape)
            scaled_resulting_rows /= scaled_resulting_rows[:,normalize_axis:normalize_axis+1]
            delta = np.abs(scaled_resulting_rows - guessed_row[np.newaxis,:])
            if min(np.nanmax(delta, axis=1)) < 1e-2:
                print("Likely have found this node before")
                raise


        if CHEATING:
            # Check our work against the ground truth entries in the corresponding matrix
            layers = cheat_get_inner_layers(all_criticals[list(component)[0]])
            layer_vals = [np.min(np.abs(x)) for x in layers]
            which_layer = np.argmin(layer_vals)
        
            M = A[which_layer]
            which_neuron = which_is_zero(which_layer, layers)
            print("Neuron corresponds to", which_neuron)
            if which_layer != LAYER:
                which_neuron = 0
                normalize_axis = 0

            actual_row = M[:,which_neuron]/M[normalize_axis,which_neuron]
            actual_row = actual_row[:guessed_row.shape[0]]
    
            do_print_err = np.any(np.isnan(guessed_row))
    
            if which_layer == LAYER:
                error = np.max(np.abs(np.abs(guessed_row)-np.abs(actual_row)))
            else:
                error = 1e6
            print('max error', "%0.8f"%error, len(component))
            if (error > 1e-4 * len(guessed_row) and debug) or do_print_err:
                print('real ', " ".join("%2.3f"%x for x in actual_row))
                print('guess', " ".join("%2.3f"%x for x in guessed_row))
                print('gap', " ".join("%2.3f"%(np.abs(x-y)) for x,y in zip(guessed_row,actual_row)))
                #print("scale", scale)
                print("--")
                for row in possible_matrix_rows:
                    print('posbl', " ".join("%2.3f"%x for x in row/row[normalize_axis]))
                print("--")
                
                scale = 10**int(np.round(np.log(np.nanmedian(np.abs(possible_matrix_rows)))/np.log(10)))
                possible_matrix_rows /= scale
                for row in possible_matrix_rows:
                    print('posbl', " ".join("%2.3f"%x for x in row))
        if np.any(np.isnan(guessed_row)) and c_count < expected_neurons:
            print("Got NaN, need more data",len(component)/sum(map(len,components)),1/sizes[LAYER+1])
            if len(component) >= 3:
                if c_count < expected_neurons:
                    failure = GatherMoreData([all_criticals[x] for x in component])
                skips_because_of_nan += 1
            continue

        guessed_row[np.isnan(guessed_row)] = 0

        if c_count < expected_neurons and len(component) >= 3:
            resulting_rows.append(guessed_row)
            resulting_examples.append([all_criticals[x] for x in component])
        else:
            print("Don't add it to the set")


    # We set failure when something went wrong but we want to defer crashing
    # (so that we can use the partial solution)

    if len(resulting_rows)+skips_because_of_nan < expected_neurons and len(all_ratios) < DEAD_NEURON_THRESHOLD:
        print("We have not explored all neurons. Do more random search", len(resulting_rows), skips_because_of_nan, expected_neurons)
        raise AcceptableFailure(partial_solution=(np.array(resulting_rows), np.array(resulting_examples)))
    else:
        print("At this point, we just assume the neuron must be dead")
        while len(resulting_rows) < expected_neurons:
            resulting_rows.append(np.zeros_like((resulting_rows[0])))
            resulting_examples.append([np.zeros_like(resulting_examples[0][0])])

    # Here we know it's a GatherMoreData failure, but we want to only do this
    # if there was enough data for everything else
    if failure is not None:
        print("Need to raise a previously generated failure.")
        raise failure


    print("Successfully returning a solution attempt.\n")
    return resulting_examples, resulting_rows

def ratio_normalize(possible_matrix_rows):
    # We get a set of a bunch of numbers
    # a1 b1 c1 d1 e1 f1 g1 
    # a2 b2 c2 d2 e2 f2 g2
    # such that some of them are nan
    # We want to compute the pairwise ratios ignoring the nans

    now = time.time()
    ratio_evidence = [[[] for _ in range(possible_matrix_rows.shape[1])] for _ in range(possible_matrix_rows.shape[1])]

    for row in possible_matrix_rows:
        for i in range(len(row)):
            for j in range(len(row)):
                ratio_evidence[i][j].append(row[i]/row[j])

    if len(ratio_evidence) > 100:
        ratio_evidence = np.array(ratio_evidence, dtype=np.float32)
    else:
        ratio_evidence = np.array(ratio_evidence, dtype=np.float64)
        
    medians = np.nanmedian(ratio_evidence, axis=2)
    errors = np.nanstd(ratio_evidence, axis=2) / np.sum(~np.isnan(ratio_evidence), axis=2)**.5
    errors += 1e-2 * (np.sum(~np.isnan(ratio_evidence), axis=2) == 1)
    errors /= np.abs(medians)
    errors[np.isnan(errors)] = 1e6

    ratio_evidence = medians

    last_nan_count = 1e8
    last_total_cost = 1e8

    while (np.sum(np.isnan(ratio_evidence)) < last_nan_count or last_total_cost < np.sum(errors)*.9) and False:
        last_nan_count = np.sum(np.isnan(ratio_evidence))
        last_total_cost = np.sum(errors)
        print('.')
        print("Takenc", time.time()-now)
        print('nan count', last_nan_count)
        print('total cost', last_total_cost)

        cost_i_over_j = ratio_evidence[:,:,np.newaxis]
        cost_j_over_k = ratio_evidence
        cost_i_over_k = cost_i_over_j * cost_j_over_k
        del cost_i_over_j, cost_j_over_k
        print(cost_i_over_k.shape, cost_i_over_k.dtype)

        error_i_over_j = errors[:,:,np.newaxis]
        error_j_over_k = errors
        error_i_over_k = error_i_over_j + error_j_over_k

        best_indexs = np.nanargmin(error_i_over_k,axis=1)
        best_errors = np.nanmin(error_i_over_k,axis=1)
        del error_i_over_j, error_j_over_k, error_i_over_k

        cost_i_over_k_new = []
        for i in range(len(best_indexs)):
            cost_i_over_k_new.append(cost_i_over_k[i].T[np.arange(len(best_indexs)),best_indexs[i]])

        cost_i_over_k = np.array(cost_i_over_k_new)
        
        which = best_errors<errors
        ratio_evidence = cost_i_over_k*which + ratio_evidence*(1-which)
        errors = best_errors


    # Choose the column with the fewest nans to return
    nancount = np.sum(np.isnan(ratio_evidence), axis=0)

    #print("Column nan count", nancount)
    
    column_ok = np.min(nancount) == nancount

    best = (None, np.inf)

    cost_i_over_j = ratio_evidence[:,:,np.newaxis]
    cost_j_over_k = ratio_evidence
    cost_i_over_k = cost_i_over_j * cost_j_over_k
    
    cost_i_j_k = cost_i_over_k
    # cost from i through j to k
    
    for column in range(len(column_ok)):
        if not column_ok[column]:
            continue

        quality = np.nansum(np.abs(cost_i_j_k[:,column,:] - ratio_evidence))
        #print('q',quality)
        if quality < best[1]:
            best = (column, quality)

    column, best_error = best
    
    return ratio_evidence[:,column], column, best_error

def gather_ratios(critical_points, known_T, check_fn, LAYER, COUNT):
    this_layer_critical_points = []
    print("Gathering", COUNT, "critical points")
    for point in critical_points:
        if LAYER > 0:
            if any(np.any(np.abs(x) < 1e-5) for x in known_T.get_hidden_layers(point)):
                continue
            if CHEATING:
                if np.any(np.abs(cheat_get_inner_layers(point)[0]) < 1e-10):
                    print(cheat_get_inner_layers(point))
                    print("Looking at one I don't need to")
            
            
        if LAYER > 0 and np.sum(known_T.forward(point) != 0) <= 1:
            print("Not enough hidden values are active to get meaningful data")
            continue

        if not check_fn(point):
            #print("Check function rejected it")
            continue
        if CHEATING:
            print("What layer is this neuron on (by cheating)?",
                  [(np.min(np.abs(x)), np.argmin(np.abs(x))) for x in cheat_get_inner_layers(point)])

        tmp = query_count
        for EPS in [GRAD_EPS, GRAD_EPS/10, GRAD_EPS/100]:
            try:
                normal = get_ratios_lstsq(LAYER, [point], [range(DIM)], known_T, eps=EPS)[0].flatten()
                #normal = get_ratios([point], [range(DIM)], eps=EPS)[0].flatten()
                break
            except AcceptableFailure:
                print("Try again with smaller eps")
                pass
        #print("LSTSQ Delta queries", query_count-tmp)

        this_layer_critical_points.append((normal, point))
        
        # coupon collector: we need nlogn points.
        print("Up to", len(this_layer_critical_points), 'of', COUNT)
        if len(this_layer_critical_points) >= COUNT:
            break

    return this_layer_critical_points

def compute_layer_values(critical_points, known_T, LAYER):
    if LAYER == 0:
        COUNT = neuron_count[LAYER+1] * 3
    else:
        COUNT = neuron_count[LAYER+1] * np.log(sizes[LAYER+1]) * 3


    # type: [(ratios, critical_point)]
    this_layer_critical_points = []

    partial_weights = None
    partial_biases = None

    def check_fn(point):
        if partial_weights is None:
            return True
        hidden = matmul(known_T.forward(point, with_relu=True), partial_weights.T, partial_biases)
        if np.any(np.abs(hidden) < 1e-4):
            return False
        
        return True

    
    print()
    print("Start running critical point search to find neurons on layer", LAYER)
    while True:
        print("At this iteration I have", len(this_layer_critical_points), "critical points")

        def reuse_critical_points():
            for witness in critical_points:
                yield witness
        
        this_layer_critical_points.extend(gather_ratios(reuse_critical_points(), known_T, check_fn,
                                                         LAYER, COUNT))

        print("Query count after that search:", query_count)
        print("And now up to ", len(this_layer_critical_points), "critical points")

        ## filter out duplicates
        filtered_points = []

        # Let's not add points that are identical to onees we've already done.
        for i,(ratio1,point1) in enumerate(this_layer_critical_points):
            for ratio2,point2 in this_layer_critical_points[i+1:]:
                if np.sum((point1 - point2)**2)**.5 < 1e-10:
                    break
            else:
                filtered_points.append((ratio1, point1))
        
        this_layer_critical_points = filtered_points

        print("After filtering duplicates we're down to ", len(this_layer_critical_points), "critical points")
        

        print("Start trying to do the graph solving")
        try:
            critical_groups, extracted_normals = graph_solve([x[0] for x in this_layer_critical_points],
                                                             [x[1] for x in this_layer_critical_points],
                                                             neuron_count[LAYER+1],
                                                             LAYER=LAYER,
                                                             debug=True)
            break
        except GatherMoreData as e:
            print("Graph solving failed because we didn't explore all sides of at least one neuron")
            print("Fall back to the hyperplane following algorithm in order to get more data")
            
            def mine(r):
                while len(r) > 0:
                    print("Yielding a point")
                    yield r[0]
                    r = r[1:]
                print("No more to give!")
    
            prev_T = KnownT(known_T.A[:-1], known_T.B[:-1])
            
            _, more_critical_points = sign_recovery.solve_layer_sign(prev_T, known_T.A[-1], known_T.B[-1], mine(e.data),
                                                                     LAYER-1, already_checked_critical_points=True,
                                                                     only_need_positive=True)

            print("Add more", len(more_critical_points))
            this_layer_critical_points.extend(gather_ratios(more_critical_points, known_T, check_fn,
                                                             LAYER, 1e6))
            print("Done adding")
            
            COUNT = neuron_count[LAYER+1]
        except AcceptableFailure as e:
            print("Graph solving failed; get more points")
            COUNT = neuron_count[LAYER+1]
            if 'partial_solution' in dir(e):

                if len(e.partial_solution[0]) > 0:
                    partial_weights, corresponding_examples = e.partial_solution
                    print("Got partial solution with shape", partial_weights.shape)
                    if CHEATING:
                        print("Corresponding to", np.argmin(np.abs(cheat_get_inner_layers([x[0] for x in corresponding_examples])[LAYER]),axis=1))
    
                    partial_biases = []
                    for weight, examples in zip(partial_weights, corresponding_examples):

                        hidden = known_T.forward(examples, with_relu=True)
                        print("hidden", np.array(hidden).shape)
                        bias = -np.median(np.dot(hidden, weight))
                        partial_biases.append(bias)
                    partial_biases = np.array(partial_biases)
                    
                
    print("Number of critical points per cluster", [len(x) for x in critical_groups])
    
    point_per_class = [x[0] for x in critical_groups]

    extracted_normals = np.array(extracted_normals).T

    # Compute the bias because we know wx+b=0
    extracted_bias = [matmul(known_T.forward(point_per_class[i], with_relu=True), extracted_normals[:,i], c=None) for i in range(neuron_count[LAYER+1])]

    # Don't forget to negate it.
    # That's important.
    # No, I definitely didn't forget this line the first time around.
    extracted_bias = -np.array(extracted_bias)

    # For the failed-to-identify neurons, set the bias to zero
    extracted_bias *= np.any(extracted_normals != 0,axis=0)[:,np.newaxis]
        

    if CHEATING:
        # Compute how far we off from the true matrix
        real_scaled = A[LAYER]/A[LAYER][0]
        extracted_scaled = extracted_normals/extracted_normals[0]
        
        mask = []
        reorder_rows = []
        for i in range(len(extracted_bias)):
            which_idx = np.argmin(np.sum(np.abs(real_scaled - extracted_scaled[:,[i]]),axis=0))
            reorder_rows.append(which_idx)
            mask.append((A[LAYER][0,which_idx]))
    
        print('matrix norm difference', np.sum(np.abs(extracted_normals*mask - A[LAYER][:,reorder_rows])))
    else:
        mask = [1]*len(extracted_bias)
    

    return extracted_normals, extracted_bias, mask
