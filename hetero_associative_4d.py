# Copyright [2023] Luis Alberto Pineda Cortés & Rafael Morales Gamboa.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import math
import random
import numpy as np
import commons
from associative import AssociativeMemory

class HeteroAssociativeMemory4D:
    def __init__(self, n: int, p: int, m: int, q: int,
        es: commons.ExperimentSettings, fold, nm_qd = None, pq_qd = None,
        prototypes = None):
        """
        Parameters
        ----------
        n : int
            The size of the first domain (of properties).
        m : int
            The size of the first range (of representation).
        p : int
            The size of the second domain (of properties).
        q : int
            The size of the second range (of representation).
        es: Experimental Settings
            Includes the values for iota, kappa, xi y sigma.

        prototypes: A list of arrays of prototpyes for the domains
            defined by (n,m), and (p,q), or a list of None.
        """
        self._n = n
        self._m = m+1 # +1 to handle partial functions.
        self._p = p
        self._q = q+1 # +1 to handle partial functions.
        self._xi = es.xi
        self._absolute_max = 2**16 - 1
        self._sigma = es.sigma
        self._iota = es.iota
        self._kappa = es.kappa
        self._relation = np.zeros((self._n, self._p, self._m, self._q), dtype=int)
        self._iota_relation = np.zeros((self._n, self._p, self._m, self._q), dtype=int)
        self._prototypes = self.validate_prototypes(prototypes)
        self._entropies = np.zeros((self._n, self._p), dtype=np.double)
        self._means = np.zeros((self._n, self._p), dtype=np.double)
        self._updated = True
        self._es = es
        self._fold = fold
        # In order to accept partial functions, the borders (_m-1 and _q-1)
        # should not be zero.
        self._set_margins()

        # Set quantizers/dequantizers per dimension.
        self.qudeqs = [nm_qd, pq_qd]

        self.classifiers = [None, None]

        print(f'Relational memory {self.model_name} {{n: {self.n}, p: {self.p}, ' +
            f'm: {self.m}, q: {self.q}, ' +
            f'xi: {self.xi}, iota: {self.iota}, ' +
            f'kappa: {self.kappa}, sigma: {self.sigma}}}, has been created')

    def _get_classifier(self, dim):
        classifier = self.classifiers[dim]
        if classifier is not None:
            return classifier

        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError('TensorFlow is required only for prototype-based recall.') from exc

        dataset = commons.datasets[dim]
        model_prefix = commons.model_name(dataset, self._es)
        filename = commons.classifier_filename(model_prefix, self._es, self._fold)
        classifier = tf.keras.models.load_model(filename)
        self.classifiers[dim] = classifier
        return classifier

    def __str__(self):
        return f'{{n: {self.n}, p: {self.p}, m: {self.m}, q: {self.q},\n{self.rel_string}}}'

    @property
    def model_name(self):
        return commons.d4_model_name
    
    @property
    def n(self):
        return self._n

    @property
    def p(self):
        return self._p

    @property
    def m(self):
        return self._m-1

    @property
    def q(self):
        return self._q-1

    @property
    def relation(self):
        return self._relation[:, :, :self.m, :self.q]

    @property
    def absolute_max_value(self):
        return self._absolute_max

    @property
    def entropies(self):
        if not self._updated:
            self._updated = self.update()
        return self._entropies

    @property
    def entropy(self):
        """Return the entropy of the Hetero Associative Memory."""
        return np.mean(self.entropies)

    @property
    def means(self):
        if not self._updated:
            self._updated = self.update()
        return self._means

    @property
    def mean(self):
        return np.mean(self.means)

    @property
    def iota_relation(self):
        return self._full_iota_relation[:, :, :self.m, :self.q]

    @property
    def _full_iota_relation(self):
        if not self._updated:
            self._updated = self.update()
        return self._iota_relation

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        if sigma < 0:
            raise ValueError('Sigma must be a non negative number.')
        self._sigma = sigma

    @property
    def kappa(self):
        return self._kappa

    @kappa.setter
    def kappa(self, kappa):
        if kappa < 0:
            raise ValueError('Kappa must be a non negative number.')
        self._kappa = kappa

    @property
    def iota(self):
        return self._iota

    @iota.setter
    def iota(self, iota):
        if iota < 0:
            raise ValueError('Iota must be a non negative number.')
        self._iota = iota
        self._updated = False

    @property
    def xi(self):
        return self._xi

    @xi.setter
    def xi(self, x):
        if x < 0:
            raise ValueError('Xi must be a non negative number.')
        self._xi = x

    @property
    def exp_settings_2d(self):
        # Iota has already been applied to the 4D relation.
        return commons.ExperimentSettings(iota = 0.0, kappa = self.kappa, xi = math.sqrt(self.xi), sigma = self.sigma)
    
    @property
    def fullness(self):
        count = np.count_nonzero(self.relation)
        total = self.n*self.m*self.p*self.q
        return count*1.0/total
    
    @property
    def rel_string(self):
        return self.relation_to_string(self.relation)

    def is_undefined(self, value, dim):
        return value == self.undefined(dim)

    def undefined(self, dim: int):
        return self.m if dim == 0 else self.q

    def is_partial(self, f, dim):
        u = self.undefined(dim)
        for v in f:
            if v == u:
                return True
        return False
    
    def undefined_function(self, dim):
        return np.full(self.cols(dim), self.undefined(dim), dtype=int)

    def alt(self, dim):
        return (dim + 1) % 2

    def cols(self, dim):
        return self.n if dim == 0 else self.p

    def rows(self, dim):
        return self.m if dim == 0 else self.q

    def register(self, cue_a, cue_b, weights_a = None, weights_b = None) -> None:
        if weights_a is None:
            weights_a = np.full(len(cue_a), fill_value=1)
        if weights_b is None:
            weights_b = np.full(len(cue_b), fill_value=1)
        cue_a = self.validate(cue_a, 0)
        cue_b = self.validate(cue_b, 1)
        r_io = self.vectors_to_relation(cue_a, cue_b, weights_a, weights_b)
        self.abstract(r_io)

    def recognize(self, cue_a, cue_b, weights_a = None, weights_b = None):
        if weights_a is None:
            weights_a = np.full(len(cue_a), fill_value=1)
        if weights_b is None:
            weights_b = np.full(len(cue_b), fill_value=1)
        recognized, weights = self.recog_full_weights(cue_a, cue_b, weights_a, weights_b, final = False)
        mean_weight = np.sum(weights)
        recognized = recognized and (self._kappa*self.mean <= mean_weight)
        return recognized, mean_weight

    def recog_full_weights(self, cue_a, cue_b, weights_a = None, weights_b = None, final = True):
        if weights_a is None:
            weights_a = np.full(len(cue_a), fill_value=1)
        if weights_b is None:
            weights_b = np.full(len(cue_b), fill_value=1)
        cue_a = self.validate(cue_a, 0)
        cue_b = self.validate(cue_b, 1)
        r_io = self.vectors_to_relation(cue_a, cue_b, weights_a, weights_b)
        implication = self.containment(r_io)
        recognized = np.count_nonzero(implication == 0) <= self._xi
        weights = self._weights(r_io)
        if final:
            recognized = recognized and (self._kappa*self.mean <= np.sum(weights))
        return recognized, weights

    def recall_from_left(self, cue, method = commons.recall_with_sampling_n_search,
            euc = None, weights = None, label = None):
        if weights is None:
            weights = np.full(len(cue), fill_value=1)
        return self.recall(cue, method, euc, weights, label, 0)

    def recall_from_right(self, cue, method = commons.recall_with_sampling_n_search,
            euc = None, weights = None, label = None):
        if weights is None:
            weights = np.full(len(cue), fill_value=1)
        return self.recall(cue, method, euc, weights, label, 1)

    def recall(self, cue, method, euc, weights, label, dim):
        cue = self.validate(cue, dim)
        if euc is not None:
            euc = self.validate(euc, self.alt(dim))
        projection = self.project(cue, weights, dim)
        recognized = (np.count_nonzero(np.sum(projection, axis=1) == 0) == 0)
        if not recognized:
            r_io = self.undefined_function(self.alt(dim))
            weight = 0.0
            stats = [0, 0, 0.0, 0.0]
        else:
            r_io, weights, stats = \
                    self.optimal_recall(cue, method, euc, weights, label, projection, dim)
            if r_io is None:
                recognized = False
                r_io = self.undefined_function(self.alt(dim))
                weight = 0.0
            else:
                weight = np.mean(weights)
                r_io = self.revalidate(r_io, self.alt(dim))
        return r_io, recognized, weight, projection, stats

    def optimal_recall(self, cue, method, euc, cue_weights, label, projection, dim):
        if method == commons.recall_with_sampling_n_search:
            return self.sample_n_search_recall(cue, cue_weights, projection, dim)
        elif method == commons.recall_with_protos:
            return self.prototypes_recall(cue, cue_weights, label, projection, dim)
        elif method == commons.recall_with_correct_proto:
            return self.correct_proto_recall(cue, cue_weights, label, projection, dim)
        elif method == commons.recall_with_cue:
            return self.cue_recall(euc, projection, dim)
        else:
            raise ValueError(f'Incorrect value for method: {method}')
            
    def sample_n_search_recall(self, cue, cue_weights, projection, dim):
        sampling_iterations = 0
        giving_ups = 0
        last_update = 0
        r_io, weights = self.reduce(projection, self.alt(dim))
        distance = self.distance_recall(cue, cue_weights, r_io, weights, dim)
        visited = [r_io]
        q_io, q_ws = r_io, weights
        for k in range(commons.sample_size):
            j = 0
            while self.already_visited(q_io, visited) and (j < commons.early_threshold):
                q_io, q_ws = self.reduce(projection, self.alt(dim))
                j += 1
            if j == commons.early_threshold:
                giving_ups += 1
                continue
            visited.append(q_io)
            d = self.distance_recall(cue, cue_weights, q_io, q_ws, dim)
            if d < distance:
                r_io = q_io
                weights = q_ws
                distance = d
                sampling_iterations += 1
                last_update = k
        distance2 = distance
        better_found = True
        k = commons.sample_size
        search_iterations = 0
        sampling_io = r_io
        sampling_ws = weights
        while not commons.sampling_without_search and better_found:
            neighbors = self.neighborhood(projection, r_io, self.alt(dim))
            better_found = False
            while neighbors:
                t = neighbors.pop()
                i = t[0]
                v = t[1]
                q_io = np.array([r_io[j] if j != i else v for j in range(self.cols(self.alt(dim)))])
                if self.already_visited(q_io, visited):
                    giving_ups += 1
                    continue
                visited.append(q_io)
                q_ws = self.weights_in_projection(projection, q_io, self.alt(dim))
                k += 1
                d = self.distance_recall(cue, cue_weights, q_io, q_ws, dim)
                if d < distance2:
                    distance2 = d
                    search_iterations += 1
                    last_update = k
                    better_found = True
                    break
            if better_found:
                r_io = q_io
                weights = q_ws
        diffs, length = self.functions_distance(sampling_io, sampling_ws, r_io, weights)
        return r_io, weights, [sampling_iterations, search_iterations,
                last_update, giving_ups, distance2, (distance2- distance), diffs, length]
    
    def prototypes_recall(self, cue, cue_weights, label, projection, dim):
        sampling_iterations = 0
        last_update = 0
        # giving_ups = 0
        giving_ups = 0
        coherence = self.protos_coherence(projection, self.alt(dim))
        if np.sum(coherence) == 0:
            return None, None, [sampling_iterations, last_update, np.nan]
        p = self.choose_from_distrib(coherence)
        s_projection = self.adjust_by_proto(projection, p, self.alt(dim))
        r_io, weights = self.reduce(s_projection, self.alt(dim))
        distance = self.distance_recall(cue, cue_weights, r_io, weights, dim)
        visited = [r_io]
        q_io, q_ws = r_io, weights
        for k in range(commons.sample_size):
            j = 0
            while self.already_visited(q_io, visited) and (j < commons.early_threshold):
                p = self.choose_from_distrib(coherence)
                s_projection = self.adjust_by_proto(projection, p, self.alt(dim))
                q_io, q_ws = self.reduce(s_projection, self.alt(dim))
                j += 1
            if self.already_visited(q_io, visited):
                giving_ups += 1
                continue
            visited.append(q_io)
            d = self.distance_recall(cue, cue_weights, q_io, q_ws, dim)
            if d < distance:
                r_io = q_io
                weights = q_ws
                distance = d
                sampling_iterations += 1
                last_update = k
        return r_io, weights, [sampling_iterations, last_update, giving_ups, distance]

    def correct_proto_recall(self, proto, proto_weights, label, projection, dim):
        sampling_iterations = 0
        last_update = 0
        giving_ups = 0
        am = AssociativeMemory.from_relation(projection, self.exp_settings_2d)
        recognized, _ = am.recognize(proto, validate=False)
        if not recognized:
            return None, None, [sampling_iterations, last_update, np.nan]
        s_projection = self.adjust_by_proto(projection, label, self.alt(dim))
        r_io, weights = self.reduce(s_projection, self.alt(dim))
        distance = self.distance_recall(proto, proto_weights, r_io, weights, dim)
        visited = [r_io]
        q_io, q_ws = r_io, weights
        for k in range(commons.sample_size):
            j = 0
            while self.already_visited(q_io, visited) and (j < commons.early_threshold):
                q_io, q_ws = self.reduce(s_projection, self.alt(dim))
                j += 1
            if j == commons.early_threshold:
                giving_ups += 1
                continue
            visited.append(q_io)
            d = self.distance_recall(proto, proto_weights, q_io, q_ws, dim)
            if d < distance:
                r_io = q_io
                weights = q_ws
                distance = d
                sampling_iterations += 1
                last_update = k
        return r_io, weights, [sampling_iterations, last_update, giving_ups, distance]

    def cue_recall(self, euc, projection, dim):
        am = AssociativeMemory.from_relation(projection, self.exp_settings_2d)
        recognized, _ = am.recognize(euc, validate=False)
        if not recognized:
            return None, None, [0.0]
        s = self.rows(self.alt(dim)) * self.sigma
        s_projection = self.adjust(projection, euc, s)
        r_io, weights = self.reduce(s_projection, self.alt(dim))
        return r_io, weights, [0.0]
    
    def abstract(self, r_io):
        self._relation = np.where(
            self._relation == self.absolute_max_value,
            self._relation, self._relation + r_io)
        self._updated = False

    def containment(self, r_io):
        return np.where((r_io == 0) | (self._full_iota_relation != 0), 1, 0)

    def project(self, cue, weights, dim):
        integration = np.zeros((self.cols(self.alt(dim)), self.rows(self.alt(dim))), dtype=float)
        sum_weights = np.sum(weights)
        if sum_weights == 0:
            return integration
        first = True
        w = cue.size*weights/sum_weights
        for i in range(cue.size):
            k = cue[i]
            if self.is_undefined(k, dim):
                continue
            projection = (self._full_iota_relation[i, :, k, :self.q] if dim == 0
                else self._full_iota_relation[:, i, :self.m, k])
            if first:
                integration = w[i]*projection
                first = False
            else:
                integration = np.where((integration == 0) | (projection == 0),
                        0, integration + w[i]*projection)
        return integration

    def distance_recall(self, cue, cue_weights, q_io, q_ws, dim, label=None):
        projection = self.project(q_io, q_ws, self.alt(dim))
        if label is not None:
            projection = self.adjust_by_proto(projection, label, dim)
        # booleans = np.where(projection > 0, True, False)
        # r_cue = self.to_relation(cue, dim)
        # r_io = ~r_cue | booleans
        # distance = float('inf') \
        #         if np.count_nonzero(~r_io) > 0 \
        #         else self.calculate_distance(cue, cue_weights, projection, dim)
        distance = self.calculate_distance(cue, cue_weights, projection, dim)
        return distance

    def calculate_distance(self, cue, cue_weights, p_io, dim):
        distance = 0.0
        ws = 0.0
        for v, w, column in zip(cue, cue_weights, p_io):
            if self.is_undefined(v, dim):
                continue
            ws += w
            s = np.sum(column)
            ps = column if s == 0.0 else column/np.sum(column)
            d = np.dot(np.square(np.arange(self.rows(dim))-v),ps)*w
            distance += d
        return distance / np.sum(ws)
    
    def functions_distance(self, p_io, p_ws, q_io, q_ws):
        abs = np.abs(p_io - q_io)
        diff = np.sum(abs)
        length = np.max(abs)
        return diff, length

    def protos_coherence(self, projection, dim):
        frequencies = self.prototypes_frequencies(projection, dim)
        coherence = []
        adjustment = -math.log2(0.9999)
        for label in commons.all_labels:
            freqs = frequencies[label]
            s = np.sum(freqs)
            probs = freqs if s == 0 else freqs/s
            entropy = np.sum(np.multiply(-probs, np.log2(np.where(probs == 0.0, 1.0, probs))))
            coherence.append(probs[label]/(entropy+adjustment))
        return np.array(coherence)

    # Reduces a relation to a function
    def reduce(self, relation, dim, excluded = None):
        cols = self.cols(dim)
        v = np.array([self.choose(column, dim) for column in relation]) \
            if excluded is None else \
                np.array([self.choose(column, dim, exc) for column, exc in zip(relation, excluded)])
        weights = []
        for i in range(cols):
            if self.is_undefined(v[i], dim):
                weights.append(0)
            else:
                weights.append(relation[i, v[i]])
        return v, np.array(weights)

    def choose(self, column, dim, excluded = None):
        """Choose a value from the column given a cue
        
        It assumes the column as a probabilistic distribution.
        """
        s = column.sum()
        if s == 0:
            return self.undefined(dim)
        if (excluded is not None):
            if s > column[excluded]:
                s -= column[excluded]
            else:
                excluded = None
        r = s*random.random()
        for j in range(column.size):
            if (excluded is not None) and (j == excluded):
                continue
            if r <= column[j]:
                return j
            r -= column[j]
        return self.undefined(dim)

    def choose_from_distrib(self, distribution):
        """Choose a value using the probability distribution"""
        s = distribution.sum()
        if s == 0:
            raise ValueError('Invalid probability distribution')
        r = s*random.random()
        chosen = None
        for j in range(distribution.size):
            if r <= distribution[j]:
                chosen = j
                break
            r -= distribution[j]
        return chosen

    def neighborhood(self, projection, r_io, dim):
        neigh = []
        rows = self.rows(dim)-1
        for i in range(self.cols(dim)):
            column = projection[i]
            value = r_io[i]
            if (value < rows) and (column[value+1] > 0):
                neigh.append((i, value+1))
            if (0 < value) and (column[value-1] > 0):
                neigh.append((i, value-1))
        random.shuffle(neigh)
        return neigh

    def adjust(self, projection, cue, s):
        if cue is None:
            return projection
        s_projection = []
        for column, mean in zip(projection, cue):
            adjusted = self.ponderate(column, mean, s)
            s_projection.append(adjusted)
        return np.array(s_projection)

    def ponderate(self, column, mean, s):
        norm = np.array([self.normpdf(i, mean, s)/self.normpdf(0, 0, s) for i in range(column.size)])
        return norm*column
    
    def _weights(self, r_io):
        r = r_io/np.sum(r_io)
        weights = np.sum(r[:, :, :self.m, :self.q] * self.relation, axis=(2,3))
        return weights
        
    def weights_in_projection(self, projection, q_io, dim):
        return projection[np.arange(self.cols(dim)), q_io]
    
    def update(self):
        self._update_entropies()
        self._update_means()
        self._update_iota_relation()
        return True

    def _update_entropies(self):
        for i in range(self.n):
            for j in range(self.p):
                relation = self.relation[i, j, :, :]
                total = np.sum(relation)
                if total > 0:
                    matrix = relation/total
                else:
                    matrix = relation.copy()
                matrix = np.multiply(-matrix, np.log2(np.where(matrix == 0.0, 1.0, matrix)))
                self._entropies[i, j] = np.sum(matrix)
        print(f'Entropy updated to mean = {np.mean(self._entropies)}, ' 
              + f'stdev = {np.std(self._entropies)}')

    def _update_means(self):
        for i in range(self.n):
            for j in range(self.p):
                r = self.relation[i, j, :, :]
                count = np.count_nonzero(r)
                count = 1 if count == 0 else count
                self._means[i,j] = np.sum(r)/count

    def _update_iota_relation(self):
        for i in range(self.n):
            for j in range(self.p):
                matrix = self.relation[i, j, :, :]
                s = np.sum(matrix)
                if s == 0:
                    self._iota_relation[i, j, :self.m, :self.q] = \
                        np.zeros((self.m, self.q), dtype=int)
                else:
                    count = np.count_nonzero(matrix)
                    threshold = self.iota*s/count
                    self._iota_relation[i, j, :self.m, :self.q] = \
                        np.where(matrix < threshold, 0, matrix)
        turned_off = np.count_nonzero(self._relation) - np.count_nonzero(self._iota_relation)
        print(f'Iota relation updated, and {turned_off} cells have been turned off')

    def validate(self, cue, dim):
        """ It asumes vector is an array of floats, and np.nan
            is used to register an undefined value, but it also
            considerers any negative number or out of range number
            as undefined.
        """
        expected_length = self.cols(dim)
        if (len(cue.shape) < 1) or (len(cue.shape) > 2):
            raise ValueError(f'Unexpected shape of cue(s): {cue.shape}.')
        if len(cue.shape) == 1:
            if cue.size != expected_length:
                raise ValueError('Invalid lenght of the input data. Expected ' +
                        f'{expected_length} and given {cue.size}')
        elif cue.shape[1] != expected_length:
            raise ValueError(f'Expected shape (n, {expected_length}) ' +
                    f'but got shape {cue.shape}')
        threshold = self.rows(dim)
        undefined = self.undefined(dim)
        v = np.where(cue < 0, 0, cue)
        v = np.where(threshold <= v, self.rows(dim)-1, v)
        v = np.nan_to_num(v, copy=True, nan=undefined)
        v = v.round()
        return v.astype('int')

    def revalidate(self, memory, dim):
        v = np.where(memory == self.undefined(dim), np.nan, memory)
        return v

    def validate_prototypes(self, prototypes):
        if prototypes is None:
            return [None, None]
        protos = []
        for dim in range(2):
            protos.append(None if prototypes[dim] is None \
                    else self.validate(prototypes[dim], dim))
        return protos

    def vectors_to_relation(self, cue_a, cue_b, weights_a, weights_b):
        relation = np.zeros((self._n, self._p, self._m, self._q), dtype=int)
        for i in range(self.n):
            k = cue_a[i]
            for j in range(self.p):
                label = cue_b[j]
                w = math.sqrt(weights_a[i]*weights_b[j])
                relation[i, j, k, label] = int(w)
        return relation

    def to_relation(self, cue, dim):
        relation = np.zeros((self.cols(dim), self.rows(dim)+1), dtype=bool)
        relation[range(self.cols(dim)), cue] = True
        return relation[:, :self.rows(dim)]

    def _set_margins(self):
        """ Set margins to one.

        Margins are tuples (i, j, k, l) where either k = self.m or l = self.q.
        """
        self._relation[:, :, self.m, :] = np.full((self._n, self._p, self._q), 1, dtype=int)
        self._relation[:, :, :, self.q] = np.full((self._n, self._p, self._m), 1, dtype=int)
        self._iota_relation[:, :, self.m, :] = np.full((self._n, self._p, self._q), 1, dtype=int)
        self._iota_relation[:, :, :, self.q] = np.full((self._n, self._p, self._m), 1, dtype=int)

    def prototypes_frequencies(self, projection, dim):
        frequencies = []
        classifier = self._get_classifier(dim)
        am = AssociativeMemory.from_relation(projection, self.exp_settings_2d)
        for lbl in commons.all_labels:
            proto = self._prototypes[dim][lbl]
            counts = np.zeros(commons.n_labels, dtype=int)
            recognized, _ = am.recognize(proto, validate=False)
            if recognized:
                s = self.rows(dim) * self.sigma
                s_projection = self.adjust(projection, proto, s)
                if (np.count_nonzero(np.sum(s_projection, axis=1) == 0) == 0):
                    memories = []
                    for i in range(commons.presence_iterations):
                        r_io, _ = self.reduce(s_projection, dim)
                        if not self.is_partial(r_io, dim):
                            memories.append(r_io)
                    if len(memories) > 0:
                        memories = self.rsize_recalls(np.array(memories), dim)
                        classification = np.argmax(classifier(memories, training=False), axis=1)
                        labels, freqs = np.unique(classification, return_counts=True)
                        for lbl, freq in zip(labels, freqs):
                            counts[lbl] = freq
            frequencies.append(counts)
        gc.collect()
        return np.array(frequencies, dtype=float)
    
    def rsize_recalls(self, recalls, dim):
        return self.qudeqs[dim].dequantize(recalls, self.rows(dim))

    def relation_to_string(self, a, p = ''):
        if a.ndim == 1:
            return f'{p}{a}'
        s = f'{p}[\n'
        for b in a:
            ss = self.relation_to_string(b, p + ' ')
            s = f'{s}{ss}\n'
        s = f'{s}{p}]'
        return s

    def already_visited(self, r_io, visited):
        for q_io in visited:
            if np.array_equal(r_io, q_io):
                return True
        return False
    
    def adjust_by_proto(self, relation, label, dim):
        proto = self._prototypes[dim][label]
        s = self.rows(dim) * self.sigma
        q = self.adjust(relation, proto, s)
        return q

    def normpdf(self, x, mean, sd):
        var = float(sd)**2
        denom = (2*math.pi*var)**.5
        num = math.exp(-(float(x)-float(mean))**2/(2*var))
        return num/denom

