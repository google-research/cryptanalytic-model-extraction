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
import random
import traceback
import time
import numpy.linalg
import pickle
import multiprocessing as mp
import os
import signal

import numpy as np

from src.utils import matmul, KnownT, check_quality, SAVED_QUERIES, run
from src.find_witnesses import sweep_for_critical_points
import src.refine_precision as refine_precision
import src.layer_recovery as layer_recovery
import src.sign_recovery as sign_recovery
from src.global_vars import *

#####################################################################
##          MAIN FUNCTION. This is where it all happens.           ##
#####################################################################

def run_full_attack():
    global query_count, SAVED_QUERIES

    extracted_normals = []
    extracted_biases = []
    
    known_T = KnownT(extracted_normals, extracted_biases)        
    
    for layer_num in range(0,len(A)-1):
        # For each layer of the network ...

        # First setup the critical points generator
        critical_points = sweep_for_critical_points(PARAM_SEARCH_AT_LOCATION, known_T)

        # Extract weights corresponding to those critical points
        extracted_normal, extracted_bias, mask = layer_recovery.compute_layer_values(critical_points,
                                                                                     known_T, 
                                                                                     layer_num)

        # Report how well we're doing
        check_quality(layer_num, extracted_normal, extracted_bias)

        # Now, make them more precise
        extracted_normal, extracted_bias = refine_precision.improve_layer_precision(layer_num,
                                                                                    known_T, extracted_normal, extracted_bias)
        print("Query count", query_count)

        # And print how well we're doing
        check_quality(layer_num, extracted_normal, extracted_bias)

        # New generator
        critical_points = sweep_for_critical_points(1e1)

        # Solve for signs
        if layer_num == 0 and sizes[1] <= sizes[0]:
            extracted_sign = sign_recovery.solve_contractive_sign(known_T, extracted_normal, extracted_bias, layer_num)
        elif layer_num > 0 and sizes[1] <= sizes[0] and all(sizes[x+1] <= sizes[x]/2 for x in range(1,len(sizes)-1)):
            try:
                extracted_sign = sign_recovery.solve_contractive_sign(known_T, extracted_normal, extracted_bias, layer_num)
            except AcceptableFailure as e:
                print("Contractive solving failed; fall back to noncontractive method")
                if layer_num == len(A)-2:
                    print("Solve final two")
                    break

                extracted_sign, _ = sign_recovery.solve_layer_sign(known_T, extracted_normal, extracted_bias, critical_points,
                                                                   layer_num,
                                                                   l1_mask=np.int32(np.sign(mask)))
                
        else:
            if layer_num == len(A)-2:
                print("Solve final two")
                break
            
            extracted_sign, _ = sign_recovery.solve_layer_sign(known_T, extracted_normal, extracted_bias, critical_points,
                                                               layer_num,
                                                               l1_mask=np.int32(np.sign(mask)))

        print("Extracted", extracted_sign)
        print('real sign', np.int32(np.sign(mask)))

        print("Total query count", query_count)

        # Correct signs
        extracted_normal *= extracted_sign
        extracted_bias *= extracted_sign
        extracted_bias = np.array(extracted_bias, dtype=np.float64)

        # Report how we're doing
        extracted_normal, extracted_bias = check_quality(layer_num, extracted_normal, extracted_bias, do_fix=True)

        extracted_normals.append(extracted_normal)
        extracted_biases.append(extracted_bias)
    
    known_T = KnownT(extracted_normals, extracted_biases)

    for a,b in sorted(query_count_at.items(),key=lambda x: -x[1]):
        print('count', b, '\t', 'line:', a, ':', self_lines[a-1].strip())

    # And then finish up
    if len(extracted_normals) == len(sizes)-2:
        print("Just solve final layer")
        N = int(len(SAVED_QUERIES)/1000) or 1
        ins, outs = zip(*SAVED_QUERIES[::N])
        solve_final_layer(known_T, np.array(ins), np.array(outs))
    else:
        print("Solve final two")
        solve_final_two_layers(known_T, extracted_normal, extracted_bias)


def solve_final_two_layers(known_T, known_A0, known_B0):
    ## Recover the final two layers through brute forcing signs, then least squares
    ## Yes, this is mostly a copy of solve_layer_sign. I am repeating myself. Sorry.
    LAYER = len(sizes)-2
    filtered_inputs = []
    filtered_outputs = []

    # How many unique points to use. This seems to work. Tweak if needed...
    # (In checking consistency of the final layer signs)
    N = int(len(SAVED_QUERIES)/100) or 1
    ins, outs = zip(*SAVED_QUERIES[::N])
    filtered_inputs, filtered_outputs = zip(*SAVED_QUERIES[::N])
    print('Total query count', len(SAVED_QUERIES))
    print("Solving on", len(filtered_inputs))

    inputs, outputs = np.array(filtered_inputs), np.array(filtered_outputs)
    known_hidden_so_far = known_T.forward(inputs, with_relu=True)

    K = sizes[LAYER]
    print("K IS", K)
    shuf = list(range(1<<K))[::-1]

    print("Here before start", known_hidden_so_far.shape)

    start_time = time.time()

    extra_args_tup = (known_A0, known_B0, LAYER-1, known_hidden_so_far, K, -outputs)
    def shufpp(s):
        for elem in s:
            yield elem, extra_args_tup

    # Brute force all sign assignments...
    all_res = pool[0].map(sign_recovery.is_solution, shufpp(shuf))

    end_time = time.time()

    scores = [r[0] for r in all_res]
    solution_attempts = sum([r[1] for r in all_res])
    total_attempts = len(all_res)

    print("Attempts at solution:", (solution_attempts), 'out of', total_attempts)
    print("Took", end_time-start_time, 'seconds')
    
    std = np.std([x[0] for x in scores])
    print('std',std)
    print('median', np.median([x[0] for x in scores]))
    print('min', np.min([x[0] for x in scores]))

    score, recovered_signs, final = min(scores,key=lambda x: x[0])
    print('recover', recovered_signs)

    known_A0 *= recovered_signs
    known_B0 *= recovered_signs

    out = known_T.extend_by(known_A0, known_B0)

    return solve_final_layer(out, inputs, outputs)

def solve_final_layer(known_T, inputs, outputs):
    if CHEATING:
        for i,(normal,bias) in enumerate(zip(known_T.A, known_T.B)):
            print()
            print("LAYER", i)
            check_quality(i, normal, bias)
    
    outputs = run(inputs)
    hidden = known_T.forward(inputs, with_relu=True)

    hidden = np.concatenate([hidden, np.ones((hidden.shape[0], 1))], axis=1)

    solution = np.linalg.lstsq(hidden, outputs)

    vector = solution[0]

    At = known_T.A+[vector[:-1]]
    Bt = known_T.B+[vector[-1]]

    print("SAVING", "/tmp/extracted-%s.p"%"-".join(map(str,sizes)))
    
    pickle.dump([At,
                 Bt],
                open("/tmp/extracted-%s.p"%"-".join(map(str,sizes)),"wb"))

    from src.global_vars import __cheat_A, __cheat_B
    pickle.dump([__cheat_A, __cheat_B],
                open("/tmp/real-%s.p"%"-".join(map(str,sizes)),"wb"))


    def loss(x):
        return (run(x, inner_A=At, inner_B=Bt)-run(x, inner_A=__cheat_A, inner_B=__cheat_B))
    ls = []
    for _ in range(1):
        print(_)
        inp = np.random.normal(0, 1, (100000, A[0].shape[0]))
        ls.extend(loss(inp).flatten())

    print("\n\n")

    print("Finally we are done.\n")
    
    print('Maximum logit loss on the unit sphere',np.max(np.abs(ls)))
    print("\nfin")

def set_timeout(time):
    def handler(a, b):
        print("Timed out.")
        print("I assume something bad happened. Did it?")
        exit(2)
    
    signal.signal(signal.SIGALRM, handler)
    
    # Set a 1 hour timelimit for experiments.
    signal.alarm(time)
    
        
if __name__ == "__main__":
    # We use mp.Pool to make some of our operations faster
    # Figure out how many threads we can use and create the pool now
    pool.append(mp.Pool(MPROC_THREADS//4))

    set_timeout(60*60)

    print("START EXTRACTION ATTACK")
    # Make it so
    run_full_attack()
