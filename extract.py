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

from src.tracker import Tracker, Logger
from src.utils import matmul, KnownT, check_quality, run, AcceptableFailure, self_lines
from src.find_witnesses import sweep_for_critical_points
import src.refine_precision as refine_precision
import src.layer_recovery as layer_recovery
import src.sign_recovery as sign_recovery
from src.global_vars import *


logger = Logger()

#####################################################################
##          MAIN FUNCTION. This is where it all happens.           ##
#####################################################################


def run_full_attack():
    extracted_normals = []
    extracted_biases = []

    known_T = KnownT(extracted_normals, extracted_biases)

    for layer_num in range(0, len(A) - 1):
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
                                                                                    known_T, extracted_normal,
                                                                                    extracted_bias)
        logger.log("Query count", Tracker().query_count, level=Logger.INFO)

        # And print how well we're doing
        check_quality(layer_num, extracted_normal, extracted_bias)

        # New generator
        critical_points = sweep_for_critical_points(1e1)

        # Solve for signs
        if layer_num == 0 and sizes[1] <= sizes[0]:
            extracted_sign = sign_recovery.solve_contractive_sign(known_T, extracted_normal, extracted_bias, layer_num)
        elif layer_num > 0 and sizes[1] <= sizes[0] and all(
                sizes[x + 1] <= sizes[x] / 2 for x in range(1, len(sizes) - 1)):
            try:
                extracted_sign = sign_recovery.solve_contractive_sign(known_T, extracted_normal, extracted_bias,
                                                                      layer_num)
            except AcceptableFailure as e:
                logger.log("Contractive solving failed; fall back to noncontractive method", level=Logger.INFO)
                if layer_num == len(A) - 2:
                    logger.log("Solve final two", level=Logger.INFO)
                    break

                extracted_sign, _ = sign_recovery.solve_layer_sign(known_T, extracted_normal, extracted_bias,
                                                                   critical_points,
                                                                   layer_num,
                                                                   l1_mask=np.int32(np.sign(mask)))

        else:
            if layer_num == len(A) - 2:
                logger.log("Solve final two", level=Logger.INFO)
                break

            extracted_sign, _ = sign_recovery.solve_layer_sign(known_T, extracted_normal, extracted_bias,
                                                               critical_points,
                                                               layer_num,
                                                               l1_mask=np.int32(np.sign(mask)))

        logger.log("Extracted", extracted_sign, level=Logger.INFO)
        logger.log('real sign', np.int32(np.sign(mask)), level=Logger.INFO)

        logger.log("Total query count", Tracker().query_count, level=Logger.INFO)

        # Correct signs
        extracted_normal *= extracted_sign
        extracted_bias *= extracted_sign
        extracted_bias = np.array(extracted_bias, dtype=np.float64)

        # Report how we're doing
        extracted_normal, extracted_bias = check_quality(layer_num, extracted_normal, extracted_bias, do_fix=True)

        extracted_normals.append(extracted_normal)
        extracted_biases.append(extracted_bias)

    known_T = KnownT(extracted_normals, extracted_biases)

    for a, b in sorted(Tracker().query_count_at.items(), key=lambda x: -x[1]):
        logger.log('count', b, '\t', 'line:', a, ':', self_lines[a - 1].strip(), level=Logger.INFO)

    # And then finish up
    if len(extracted_normals) == len(sizes) - 2:
        logger.log("Just solve final layer", level=Logger.INFO)
        N = int(Tracker().nr_of_queries / 1000) or 1
        ins, outs = zip(*Tracker().saved_queries[::N])
        solve_final_layer(known_T, np.array(ins), np.array(outs))
    else:
        logger.log("Solve final two", level=Logger.INFO)
        solve_final_two_layers(known_T, extracted_normal, extracted_bias)


def solve_final_two_layers(known_T, known_A0, known_B0):
    ## Recover the final two layers through brute forcing signs, then least squares
    ## Yes, this is mostly a copy of solve_layer_sign. I am repeating myself. Sorry.
    LAYER = len(sizes) - 2
    filtered_inputs = []
    filtered_outputs = []

    # How many unique points to use. This seems to work. Tweak if needed...
    # (In checking consistency of the final layer signs)
    N = int(Tracker().nr_of_queries / 100) or 1
    ins, outs = zip(*Tracker().saved_queries[::N])
    filtered_inputs, filtered_outputs = zip(*Tracker().saved_queries[::N])
    logger.log('Total query count', Tracker().nr_of_queries, level=Logger.INFO)
    logger.log("Solving on", len(filtered_inputs), level=Logger.INFO)

    inputs, outputs = np.array(filtered_inputs), np.array(filtered_outputs)
    known_hidden_so_far = known_T.forward(inputs, with_relu=True)

    K = sizes[LAYER]
    logger.log("K IS", K, level=Logger.INFO)
    shuf = list(range(1 << K))[::-1]

    logger.log("Here before start", known_hidden_so_far.shape, level=Logger.INFO)

    start_time = time.time()

    extra_args_tup = (known_A0, known_B0, LAYER - 1, known_hidden_so_far, K, -outputs)

    def shufpp(s):
        for elem in s:
            yield elem, extra_args_tup

    # Brute force all sign assignments...
    all_res = pool[0].map(sign_recovery.is_solution, shufpp(shuf))

    end_time = time.time()

    scores = [r[0] for r in all_res]
    solution_attempts = sum([r[1] for r in all_res])
    total_attempts = len(all_res)

    logger.log("Attempts at solution:", (solution_attempts), 'out of', level=Logger.INFO)
    logger.log("Took", end_time - start_time, 'seconds', level=Logger.INFO)

    std = np.std([x[0] for x in scores])
    logger.log('std', std, level=Logger.INFO)
    logger.log('median', np.median([x[0] for x in scores]), level=Logger.INFO)
    logger.log('min', np.min([x[0] for x in scores]), level=Logger.INFO)

    score, recovered_signs, final = min(scores, key=lambda x: x[0])
    logger.log('recover', recovered_signs, level=Logger.INFO)

    known_A0 *= recovered_signs
    known_B0 *= recovered_signs

    out = known_T.extend_by(known_A0, known_B0)

    return solve_final_layer(out, inputs, outputs)


def solve_final_layer(known_T, inputs, outputs):
    if CHEATING:
        for i, (normal, bias) in enumerate(zip(known_T.A, known_T.B)):
            logger.log('', level=Logger.INFO)
            logger.log("LAYER", i, level=Logger.INFO)
            check_quality(i, normal, bias)

    outputs = run(inputs)
    hidden = known_T.forward(inputs, with_relu=True)

    hidden = np.concatenate([hidden, np.ones((hidden.shape[0], 1))], axis=1)

    solution = np.linalg.lstsq(hidden, outputs)

    vector = solution[0]

    At = known_T.A + [vector[:-1]]
    Bt = known_T.B + [vector[-1]]

    logger.log("SAVING", "./models/extracted-%s.p" % "-".join(map(str, sizes)), level=Logger.INFO)

    pickle.dump([At,
                 Bt],
                open("./models/extracted-%s.p" % "-".join(map(str, sizes)), "wb"))

    from src.global_vars import __cheat_A, __cheat_B
    pickle.dump([__cheat_A, __cheat_B],
                open("./models/real-%s.p" % "-".join(map(str, sizes)), "wb"))

    def loss(x):
        return (run(x, inner_A=At, inner_B=Bt) - run(x, inner_A=__cheat_A, inner_B=__cheat_B))

    ls = []
    for _ in range(1):
        print(_)
        inp = np.random.normal(0, 1, (100000, A[0].shape[0]))
        ls.extend(loss(inp).flatten())

    logger.log("\n\n", level=Logger.INFO)

    logger.log("Finally we are done.\n", level=Logger.INFO)

    max_loss = np.max(np.abs(ls))

    res = open("results.txt", "a")
    res.write(str(max_loss))
    res.close()

    logger.log('Maximum logit loss on the unit sphere', max_loss, level=Logger.INFO)
    logger.log("\nfin", level=Logger.INFO)


def set_timeout(time):
    def handler(a, b):
        logger.log("Timed out.", level=Logger.ERROR)
        logger.log("I assume something bad happened. Did it?", level=Logger.ERROR)
        exit(2)

    signal.signal(signal.SIGALRM, handler)

    # Set a 1 hour timelimit for experiments.
    signal.alarm(time)


if __name__ == "__main__":
    # We use mp.Pool to make some of our operations faster
    # Figure out how many threads we can use and create the pool now
    pool.append(mp.Pool(MPROC_THREADS // 4))

    logger.log("START EXTRACTION ATTACK", level=Logger.INFO)
    # Make it so
    run_full_attack()
