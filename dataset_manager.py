# Copyright [2020] Luis Alberto Pineda Cortés, Rafael Morales Gamboa.
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

import random
import numpy as np
from datasets import load_dataset
import commons

# Compatibility values for legacy callers that still inspect dataset geometry.
columns = commons.text_embedding_dim
rows = 1

_TRAINING_SEGMENT = 0
_FILLING_SEGMENT = 1
_TESTING_SEGMENT = 2

def get_training(dataset, fold):
    return _get_segment(dataset, _TRAINING_SEGMENT, fold)

def get_filling(dataset, fold):
    return _get_segment(dataset, _FILLING_SEGMENT, fold)

def get_testing(dataset, fold, noised = False):
    return _get_segment(dataset, _TESTING_SEGMENT, fold, noised)


def get_training_pairs(fold):
    return _get_pair_segment(_TRAINING_SEGMENT, fold)


def get_filling_pairs(fold):
    return _get_pair_segment(_FILLING_SEGMENT, fold)


def get_testing_pairs(fold, noised = False):
    return _get_pair_segment(_TESTING_SEGMENT, fold, noised)

def _get_segment(dataset, segment, fold, noised = False):
    if dataset not in commons.datasets:
        raise ValueError(f'{dataset} is not a valid dataset view')

    pairs = _get_pair_segment(segment, fold, noised)
    column = commons.dataset_columns[dataset]
    data = np.array(pairs[column], dtype=object)
    labels = np.array(pairs['pair_ids'], dtype=int)
    return data, labels

def _get_pair_segment(segment, fold, noised = False):
    if noised:
        commons.print_warning('Text datasets do not support noised mode; returning clean data.')

    if _get_pair_segment.data is None:
        _get_pair_segment.data = _load_dataset()

    return _get_data_in_range(segment, _get_pair_segment.data, fold)


_get_pair_segment.data = None

def noised(data, percent):
    raise NotImplementedError('Noise injection is not defined for text datasets.')

def _noised(image, percent):
    raise NotImplementedError('Noise injection is not defined for text datasets.')

def _load_dataset():
    left_source = commons.dataset_sources[commons.left_dataset]
    right_source = commons.dataset_sources[commons.right_dataset]
    left_column = commons.dataset_columns[commons.left_dataset]
    right_column = commons.dataset_columns[commons.right_dataset]

    left_data = load_dataset(left_source[0], split=left_source[1])
    right_data = left_data if left_source == right_source else load_dataset(right_source[0], split=right_source[1])

    total = min(len(left_data), len(right_data))
    pairs = []
    for row_id in range(total):
        left_value = _clean_text(left_data[row_id][left_column])
        right_value = _clean_definition(right_data[row_id][right_column])
        if not left_value or not right_value:
            continue

        pairs.append(
            {
                commons.left_dataset: left_value,
                commons.right_dataset: right_value,
                'pair_id': row_id,
            }
        )

    if not pairs:
        raise ValueError('The configured Hugging Face datasets produced no valid text pairs.')

    random.Random(commons.hf_split_seed).shuffle(pairs)
    return pairs
def _preprocessed_dataset(dirname):
    raise NotImplementedError('Legacy image preprocessing is no longer used.')

def _save_dataset(dirname, data, noised, labels):
    raise NotImplementedError('Legacy image preprocessing is no longer used.')

def _load_mnist_like(dirname, kind='train'):
    raise NotImplementedError('Legacy MNIST loading is no longer used.')

def _shuffle(data, noised, labels):
    raise NotImplementedError('Legacy image shuffling is no longer used.')

def _split_by_labels(data, noised, labels):
    raise NotImplementedError('Legacy label splitting is no longer used.')

def _get_data_in_range(segment, data_per_label, fold):
    if fold < 0 or fold >= commons.n_folds:
        raise ValueError(f'Fold must be in [0, {commons.n_folds - 1}]')

    total = len(data_per_label)
    training = total * commons.nn_training_percent
    filling = total * commons.am_filling_percent
    testing = total * commons.am_testing_percent
    step = total / commons.n_folds
    i = int(fold * step)
    j = int(i + training) % total
    k = int(j + filling) % total
    l = int(k + testing) % total

    if segment == _TRAINING_SEGMENT:
        start, end = i, j
    elif segment == _FILLING_SEGMENT:
        start, end = j, k
    elif segment == _TESTING_SEGMENT:
        start, end = k, l
    else:
        raise ValueError(f'Unknown segment: {segment}')

    segment_pairs = get_data_in_range(data_per_label, start, end)
    return {
        commons.left_dataset: [pair[commons.left_dataset] for pair in segment_pairs],
        commons.right_dataset: [pair[commons.right_dataset] for pair in segment_pairs],
        'pair_ids': [pair['pair_id'] for pair in segment_pairs],
    }

def get_data_in_range(data: list, i: int, j: int):
    if j > i:
        return data[i:j]
    else:
        pre = data[i:]
        pos = data[:j]
        if len(pre) == 0:
            return pos
        elif len(pos) == 0:
            return pre
        else:
            return pre + pos


def _clean_text(value):
    if value is None:
        return ''
    return str(value).strip()


def _clean_definition(definition):
    definition = _clean_text(definition)
    if '.' not in definition:
        return definition

    parts = [part.strip() for part in definition.split('.') if part.strip()]
    if not parts:
        return definition
    return max(parts, key=len)

