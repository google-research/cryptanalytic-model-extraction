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

import collections
import time

import jax
import jax.numpy as jnp
import networkx as nx

import src.sign_recovery as sign_recovery
from src.global_vars import *
from src.hyperplane_normal import get_ratios_lstsq
from src.tracker import Logger, Tracker
from src.utils import AcceptableFailure, GatherMoreData, matmul, KnownT, cheat_get_inner_layers, which_is_zero

logger = Logger()


@jax.jit
def process_block(ratios, other_ratios):
    """
    Let jax efficiently compute pairwise similarity by blocking things.
    """
    differences = jnp.abs(ratios[:, jnp.newaxis, :] - other_ratios[jnp.newaxis, :, :])
    differences = differences / jnp.abs(ratios[:, jnp.newaxis, :]) + differences / jnp.abs(
        other_ratios[jnp.newaxis, :, :])

    close = differences < BLOCK_ERROR_TOL * jnp.log(ratios.shape[1])

    pairings = jnp.sum(close, axis=2) >= max(MIN_SAME_SIZE, BLOCK_MULTIPLY_FACTOR * (np.log(ratios.shape[1]) - 2))

    return pairings


def graph_solve(all_ratios, all_criticals, expected_neurons, LAYER, debug=False):
    # 1. Load the critical points and ratios we precomputed

    all_ratios = np.array(all_ratios, dtype=np.float64)
    all_ratios_f32 = np.array(all_ratios, dtype=np.float32)
    all_criticals = np.array(all_criticals, dtype=np.float64)

    # Batch them to be sensibly sized
    ratios_group = [all_ratios_f32[i:i + 1000] for i in range(0, len(all_ratios), 1000)]
    criticals_group = [all_criticals[i:i + 1000] for i in range(0, len(all_criticals), 1000)]

    # 2. Compute the similarity pairwise between the ratios we've computed

    logger.log("Go up to", len(criticals_group), level=Logger.INFO)
    now = time.time()
    all_pairings = [[] for _ in range(sum(map(len, ratios_group)))]
    for batch_index, (criticals, ratios) in enumerate(zip(criticals_group, ratios_group)):
        logger.log(batch_index, level=Logger.INFO)

        # Compute the all-pairs similarity
        axis = list(range(all_ratios.shape[1]))
        random.shuffle(axis)
        axis = axis[:20]
        for dim in axis:
            # We may have an error on one of the directions, so let's try all of them
            scaled_all_ratios = all_ratios_f32 / all_ratios_f32[:, dim:dim + 1]
            scaled_ratios = ratios / ratios[:, dim:dim + 1]

            batch_pairings = process_block(scaled_ratios, scaled_all_ratios)

            # To get the offset, Compute the cumsum of the length up to batch_index
            batch_offset = sum(map(len, ratios_group[:batch_index]))
            # And now create the graph matching ratios that are similar
            for this_batch_i, global_j in zip(*np.nonzero(np.array(batch_pairings))):
                all_pairings[this_batch_i + batch_offset].append(global_j)
    print(time.time() - now)

    graph = nx.Graph()
    # Add the edges to the graph, removing self-loops
    graph.add_edges_from([(i, j) for i, js in enumerate(all_pairings) for j in js if abs(i - j) > 1])
    components = list(nx.connected_components(graph))

    sorted_components = sorted(components, key=lambda x: -len(x))

    if CHEATING:
        logger.log('Total (unmatched) examples found:',
                   sorted(collections.Counter(which_is_zero(LAYER, cheat_get_inner_layers(all_criticals))).items()),
                   level=Logger.INFO)

    if len(components) == 0:
        logger.log('No components found', level=Logger.ERROR)
        raise AcceptableFailure()
    logger.log("Graph search found", len(components), "different components with the following counts",
               list(map(len, sorted_components)), level=Logger.INFO)

    if CHEATING:
        which_neurons = [
            collections.Counter(which_is_zero(LAYER, cheat_get_inner_layers(all_criticals[list(orig_component)]))) for
            orig_component in sorted_components]
        first_index_of = [-1] * expected_neurons

        for i, items in enumerate(which_neurons):
            for item in items.keys():
                if first_index_of[item] == -1:
                    first_index_of[item] = i

        logger.log('These components correspond to', which_neurons, level=Logger.INFO)
        logger.log('With the corresponding index in the list:', first_index_of, level=Logger.INFO)

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
            scaled_ratios = candidate_rows / candidate_rows[:, dim:dim + 1]

            batch_pairings = process_block(scaled_ratios, scaled_ratios)

            # And now create the graph matching ratios that are similar
            for this_batch_i, global_j in zip(*np.nonzero(np.array(batch_pairings))):
                new_pairings[this_batch_i].append(global_j)

        graph = nx.Graph()
        # Add the edges to the graph, ALLOWING self-loops this time
        graph.add_edges_from([(i, j) for i, js in enumerate(new_pairings) for j in js])
        components = list(nx.connected_components(graph))

        components = [sum([list(candidate_components[y]) for y in comp], []) for comp in components]

        sorted_components = sorted(components, key=lambda x: -len(x))

        logger.log("After re-doing the graph, the component counts is", len(components), "with items",
                   list(map(len, sorted_components)), level=Logger.INFO)

        if CHEATING:
            which_neurons = [
                collections.Counter(which_is_zero(LAYER, cheat_get_inner_layers(all_criticals[list(orig_component)])))
                for orig_component in sorted_components]
            first_index_of = [-1] * expected_neurons

            for i, items in enumerate(which_neurons):
                for item in items.keys():
                    if first_index_of[item] == -1:
                        first_index_of[item] = i
            logger.log('Corresponding to', which_neurons, level=Logger.INFO)
            logger.log("First index:", first_index_of, level=Logger.INFO)
            logger.log("Expected neurons", expected_neurons, level=Logger.INFO)

    logger.log("Processing each connected component in turn.", level=Logger.INFO)

    resulting_examples = []
    resulting_rows = []

    skips_because_of_nan = 0
    failure = None

    for c_count, component in enumerate(sorted_components):
        if debug:
            logger.log("\n", level=Logger.DEBUG)
            if c_count >= expected_neurons:
                logger.log("WARNING: This one might be a duplicate!", level=Logger.DEBUG)

        logger.log("On component", c_count, "with indexs", component, level=Logger.INFO)
        if debug and CHEATING:
            inner = cheat_get_inner_layers(all_criticals[list(component)])
            logger.log('Corresponding to (cheating) ', which_is_zero(LAYER, inner), level=Logger.DEBUG)

        possible_matrix_rows = all_ratios[list(component)]

        guessed_row, normalize_axis, normalize_error = ratio_normalize(possible_matrix_rows)

        logger.log('The guessed error in the computation is', normalize_error, 'with', len(component), 'witnesses',
                   level=Logger.INFO)
        if normalize_error > .01 and len(component) <= 5:
            logger.log("Component size less than 5 with high error; this isn't enough to be sure",
                       level=Logger.INFO)
            continue
        logger.log("Normalize on axis", normalize_axis, level=Logger.INFO)

        if len(resulting_rows):
            scaled_resulting_rows = np.array(resulting_rows)
            # print(scaled_resulting_rows.shape)
            scaled_resulting_rows /= scaled_resulting_rows[:, normalize_axis:normalize_axis + 1]
            delta = np.abs(scaled_resulting_rows - guessed_row[np.newaxis, :])
            if min(np.nanmax(delta, axis=1)) < 1e-2:
                logger.log("Likely have found this node before", level=Logger.ERROR)
                raise AcceptableFailure()

        if CHEATING:
            # Check our work against the ground truth entries in the corresponding matrix
            layers = cheat_get_inner_layers(all_criticals[list(component)[0]])
            layer_vals = [np.min(np.abs(x)) for x in layers]
            which_layer = np.argmin(layer_vals)

            M = A[which_layer]
            which_neuron = which_is_zero(which_layer, layers)
            logger.log("Neuron corresponds to", which_neuron, level=Logger.INFO)
            if which_layer != LAYER:
                which_neuron = 0
                normalize_axis = 0

            actual_row = M[:, which_neuron] / M[normalize_axis, which_neuron]
            actual_row = actual_row[:guessed_row.shape[0]]

            do_print_err = np.any(np.isnan(guessed_row))

            if which_layer == LAYER:
                error = np.max(np.abs(np.abs(guessed_row) - np.abs(actual_row)))
            else:
                error = 1e6
            logger.log('max error', "%0.8f" % error, len(component), level=Logger.INFO)
            if (error > 1e-4 * len(guessed_row) and debug) or do_print_err:

                logger.log('real ', " ".join("%2.3f" % x for x in actual_row), level=Logger.INFO)
                logger.log('guess', " ".join("%2.3f" % x for x in guessed_row), level=Logger.INFO)
                logger.log('gap', " ".join("%2.3f" % (np.abs(x - y)) for x, y in zip(guessed_row, actual_row)),
                           level=Logger.INFO)
                logger.log("--", level=Logger.INFO)
                for row in possible_matrix_rows:
                    logger.log('posbl', " ".join("%2.3f" % x for x in row / row[normalize_axis]), level=Logger.INFO)
                logger.log("--", level=Logger.INFO)

                scale = 10 ** int(np.round(np.log(np.nanmedian(np.abs(possible_matrix_rows))) / np.log(10)))
                possible_matrix_rows /= scale
                for row in possible_matrix_rows:
                    logger.log('posbl', " ".join("%2.3f" % x for x in row), level=Logger.INFO)
        if np.any(np.isnan(guessed_row)) and c_count < expected_neurons:
            logger.log("Got NaN, need more data", len(component) / sum(map(len, components)), 1 / sizes[LAYER + 1],
                       level=Logger.INFO)
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
            logger.log("Don't add it to the set", level=Logger.INFO)

    # We set failure when something went wrong but we want to defer crashing
    # (so that we can use the partial solution)

    if len(resulting_rows) + skips_because_of_nan < expected_neurons and len(all_ratios) < DEAD_NEURON_THRESHOLD:
        logger.log("We have not explored all neurons. Do more random search", len(resulting_rows), skips_because_of_nan,
                   expected_neurons, level=Logger.INFO)
        raise AcceptableFailure(partial_solution=(np.array(resulting_rows), np.array(resulting_examples)))
    else:
        logger.log("At this point, we just assume the neuron must be dead", level=Logger.INFO)
        while len(resulting_rows) < expected_neurons:
            resulting_rows.append(np.zeros_like((resulting_rows[0])))
            resulting_examples.append([np.zeros_like(resulting_examples[0][0])])

    # Here we know it's a GatherMoreData failure, but we want to only do this
    # if there was enough data for everything else
    if failure is not None:
        logger.log("Need to raise a previously generated failure.", level=Logger.INFO)
        raise failure

    logger.log("Successfully returning a solution attempt.\n", level=Logger.INFO)
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
                ratio_evidence[i][j].append(row[i] / row[j])

    if len(ratio_evidence) > 100:
        ratio_evidence = np.array(ratio_evidence, dtype=np.float32)
    else:
        ratio_evidence = np.array(ratio_evidence, dtype=np.float64)

    medians = np.nanmedian(ratio_evidence, axis=2)
    errors = np.nanstd(ratio_evidence, axis=2) / np.sum(~np.isnan(ratio_evidence), axis=2) ** .5
    errors += 1e-2 * (np.sum(~np.isnan(ratio_evidence), axis=2) == 1)
    errors /= np.abs(medians)
    errors[np.isnan(errors)] = 1e6

    ratio_evidence = medians

    last_nan_count = 1e8
    last_total_cost = 1e8

    while (np.sum(np.isnan(ratio_evidence)) < last_nan_count or last_total_cost < np.sum(errors) * .9) and False:
        last_nan_count = np.sum(np.isnan(ratio_evidence))
        last_total_cost = np.sum(errors)

        logger.log(".", level=Logger.INFO)
        logger.log("Takenc", time.time() - now, level=Logger.INFO)
        logger.log('nan count', last_nan_count, level=Logger.INFO)
        logger.log('total cost', last_total_cost, level=Logger.INFO)

        cost_i_over_j = ratio_evidence[:, :, np.newaxis]
        cost_j_over_k = ratio_evidence
        cost_i_over_k = cost_i_over_j * cost_j_over_k
        del cost_i_over_j, cost_j_over_k

        logger.log(cost_i_over_k.shape, cost_i_over_k.dtype, level=Logger.INFO)

        error_i_over_j = errors[:, :, np.newaxis]
        error_j_over_k = errors
        error_i_over_k = error_i_over_j + error_j_over_k

        best_indexs = np.nanargmin(error_i_over_k, axis=1)
        best_errors = np.nanmin(error_i_over_k, axis=1)
        del error_i_over_j, error_j_over_k, error_i_over_k

        cost_i_over_k_new = []
        for i in range(len(best_indexs)):
            cost_i_over_k_new.append(cost_i_over_k[i].T[np.arange(len(best_indexs)), best_indexs[i]])

        cost_i_over_k = np.array(cost_i_over_k_new)

        which = best_errors < errors
        ratio_evidence = cost_i_over_k * which + ratio_evidence * (1 - which)
        errors = best_errors

    # Choose the column with the fewest nans to return
    nancount = np.sum(np.isnan(ratio_evidence), axis=0)

    # print("Column nan count", nancount)

    column_ok = np.min(nancount) == nancount

    best = (None, np.inf)

    cost_i_over_j = ratio_evidence[:, :, np.newaxis]
    cost_j_over_k = ratio_evidence
    cost_i_over_k = cost_i_over_j * cost_j_over_k

    cost_i_j_k = cost_i_over_k
    # cost from i through j to k

    for column in range(len(column_ok)):
        if not column_ok[column]:
            continue

        quality = np.nansum(np.abs(cost_i_j_k[:, column, :] - ratio_evidence))
        # print('q',quality)
        if quality < best[1]:
            best = (column, quality)

    column, best_error = best

    return ratio_evidence[:, column], column, best_error


def gather_ratios(critical_points, known_T, check_fn, LAYER, COUNT):
    this_layer_critical_points = []
    logger.log("Gathering", COUNT, "critical points", level=Logger.INFO)
    for point in critical_points:
        if LAYER > 0:
            if any(np.any(np.abs(x) < 1e-5) for x in known_T.get_hidden_layers(point)):
                continue
            if CHEATING:
                if np.any(np.abs(cheat_get_inner_layers(point)[0]) < 1e-10):
                    logger.log(cheat_get_inner_layers(point), level=Logger.INFO)
                    logger.log("Looking at one I don't need to", level=Logger.INFO)

        if LAYER > 0 and np.sum(known_T.forward(point) != 0) <= 1:
            logger.log("Not enough hidden values are active to get meaningful data", level=Logger.INFO)
            continue

        if not check_fn(point):
            # print("Check function rejected it")
            continue
        if CHEATING:
            logger.log("What layer is this neuron on (by cheating)?",
                       [(np.min(np.abs(x)), np.argmin(np.abs(x))) for x in cheat_get_inner_layers(point)],
                       level=Logger.INFO)

        tmp = Tracker().query_count
        for EPS in [GRAD_EPS, GRAD_EPS / 10, GRAD_EPS / 100]:
            try:
                normal = get_ratios_lstsq(LAYER, [point], [range(DIM)], known_T, eps=EPS)[0].flatten()
                # normal = get_ratios([point], [range(DIM)], eps=EPS)[0].flatten()
                break
            except AcceptableFailure:
                logger.log("Try again with smaller eps", level=Logger.INFO)
                pass
        # print("LSTSQ Delta queries", query_count-tmp)

        this_layer_critical_points.append((normal, point))

        # coupon collector: we need nlogn points.
        logger.log("Up to", len(this_layer_critical_points), 'of', COUNT, level=Logger.INFO)
        if len(this_layer_critical_points) >= COUNT:
            break

    return this_layer_critical_points


def compute_layer_values(critical_points, known_T, LAYER):
    if LAYER == 0:
        COUNT = neuron_count[LAYER + 1] * 3
    else:
        COUNT = neuron_count[LAYER + 1] * np.log(sizes[LAYER + 1]) * 3

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

    logger.log("", level=Logger.INFO)
    logger.log("Start running critical point search to find neurons on layer", LAYER, level=Logger.INFO)
    while True:
        logger.log("At this iteration I have", len(this_layer_critical_points), "critical points", level=Logger.INFO)

        def reuse_critical_points():
            for witness in critical_points:
                yield witness

        this_layer_critical_points.extend(gather_ratios(reuse_critical_points(), known_T, check_fn,
                                                        LAYER, COUNT))

        logger.log("Query count after that search:", Tracker().query_count, level=Logger.INFO)
        logger.log("And now up to ", len(this_layer_critical_points), "critical points", level=Logger.INFO)

        ## filter out duplicates
        filtered_points = []

        # Let's not add points that are identical to onees we've already done.
        for i, (ratio1, point1) in enumerate(this_layer_critical_points):
            for ratio2, point2 in this_layer_critical_points[i + 1:]:
                if np.sum((point1 - point2) ** 2) ** .5 < 1e-10:
                    break
            else:
                filtered_points.append((ratio1, point1))

        this_layer_critical_points = filtered_points

        logger.log("After filtering duplicates we're down to ", len(this_layer_critical_points), "critical points",
                   level=Logger.INFO)

        logger.log("Start trying to do the graph solving", level=Logger.INFO)
        try:
            critical_groups, extracted_normals = graph_solve([x[0] for x in this_layer_critical_points],
                                                             [x[1] for x in this_layer_critical_points],
                                                             neuron_count[LAYER + 1],
                                                             LAYER=LAYER,
                                                             debug=True)
            break
        except GatherMoreData as e:
            logger.log("Graph solving failed because we didn't explore all sides of at least one neuron",
                       level=Logger.INFO)
            logger.log("Fall back to the hyperplane following algorithm in order to get more data", level=Logger.INFO)

            def mine(r):
                while len(r) > 0:
                    logger.log("Yielding a point", level=Logger.INFO)
                    yield r[0]
                    r = r[1:]
                logger.log("No more to give!", level=Logger.INFO)

            prev_T = KnownT(known_T.A[:-1], known_T.B[:-1])

            _, more_critical_points = sign_recovery.solve_layer_sign(prev_T, known_T.A[-1], known_T.B[-1], mine(e.data),
                                                                     LAYER - 1, already_checked_critical_points=True,
                                                                     only_need_positive=True)

            logger.log("Add more", len(more_critical_points), level=Logger.INFO)
            this_layer_critical_points.extend(gather_ratios(more_critical_points, known_T, check_fn,
                                                            LAYER, 1e6))
            logger.log("Done adding", level=Logger.INFO)

            COUNT = neuron_count[LAYER + 1]
        except AcceptableFailure as e:
            logger.log("Graph solving failed; get more points", level=Logger.INFO)
            COUNT = neuron_count[LAYER + 1]
            if 'partial_solution' in dir(e):

                if len(e.partial_solution[0]) > 0:
                    partial_weights, corresponding_examples = e.partial_solution
                    logger.log("Got partial solution with shape", partial_weights.shape, level=Logger.INFO)
                    if CHEATING:
                        logger.log("Corresponding to",
                                   np.argmin(
                                       np.abs(cheat_get_inner_layers([x[0] for x in corresponding_examples])[LAYER]),
                                       axis=1), level=Logger.INFO)

                    partial_biases = []
                    for weight, examples in zip(partial_weights, corresponding_examples):
                        hidden = known_T.forward(examples, with_relu=True)
                        logger.log("hidden", np.array(hidden).shape, level=Logger.INFO)
                        bias = -np.median(np.dot(hidden, weight))
                        partial_biases.append(bias)
                    partial_biases = np.array(partial_biases)

    logger.log("Number of critical points per cluster", [len(x) for x in critical_groups], level=Logger.INFO)

    point_per_class = [x[0] for x in critical_groups]

    extracted_normals = np.array(extracted_normals).T

    # Compute the bias because we know wx+b=0
    extracted_bias = [matmul(known_T.forward(point_per_class[i], with_relu=True), extracted_normals[:, i], c=None) for i
                      in range(neuron_count[LAYER + 1])]

    # Don't forget to negate it.
    # That's important.
    # No, I definitely didn't forget this line the first time around.
    extracted_bias = -np.array(extracted_bias)

    # For the failed-to-identify neurons, set the bias to zero
    extracted_bias *= np.any(extracted_normals != 0, axis=0)[:, np.newaxis]

    if CHEATING:
        # Compute how far we off from the true matrix
        real_scaled = A[LAYER] / A[LAYER][0]
        extracted_scaled = extracted_normals / extracted_normals[0]

        mask = []
        reorder_rows = []
        for i in range(len(extracted_bias)):
            which_idx = np.argmin(np.sum(np.abs(real_scaled - extracted_scaled[:, [i]]), axis=0))
            reorder_rows.append(which_idx)
            mask.append((A[LAYER][0, which_idx]))

        logger.log('matrix norm difference', np.sum(np.abs(extracted_normals * mask - A[LAYER][:, reorder_rows])),
                   level=Logger.INFO)
    else:
        mask = [1] * len(extracted_bias)

    return extracted_normals, extracted_bias, mask
