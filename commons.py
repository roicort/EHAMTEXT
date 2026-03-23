# Copyright [2020] Luis Alberto Pineda Cortés, Gibrán Fuentes Pineda,
# Rafael Morales Gamboa.
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

import csv
import os

# os.environ['CUDA_VISIBLE_DEVICES']='0'
import re
import sys
import random
import string
import time
import numpy as np

left_hf_dataset_id = os.getenv('EHAM_LEFT_HF_DATASET', 'npvinHnivqn/EnglishDictionary')
right_hf_dataset_id = os.getenv('EHAM_RIGHT_HF_DATASET', left_hf_dataset_id)
left_hf_dataset_split = os.getenv('EHAM_LEFT_HF_SPLIT', 'train')
right_hf_dataset_split = os.getenv('EHAM_RIGHT_HF_SPLIT', left_hf_dataset_split)
hf_split_seed = int(os.getenv('EHAM_HF_SPLIT_SEED', '42'))
left_dataset = os.getenv('EHAM_LEFT_VIEW', 'word')
right_dataset = os.getenv('EHAM_RIGHT_VIEW', 'definition')

text_embedding_dim = 1024
embedding_batch_size = 64
pair_ids_prefix = 'pair_ids-'

datasets = [left_dataset, right_dataset]
datasets_to_domains = {left_dataset: text_embedding_dim, right_dataset: text_embedding_dim}
datasets_to_codomains = {left_dataset: 16, right_dataset: 16}

dataset_columns = {
    left_dataset: os.getenv('EHAM_LEFT_COLUMN', 'word'),
    right_dataset: os.getenv('EHAM_RIGHT_COLUMN', 'definition'),
}

dataset_sources = {
    left_dataset: (left_hf_dataset_id, left_hf_dataset_split),
    right_dataset: (right_hf_dataset_id, right_hf_dataset_split),
}

dataset_max_seq_len = {
    left_dataset: 64,
    right_dataset: 512,
}


def alt(dataset):
    if dataset == left_dataset:
        return right_dataset
    elif dataset == right_dataset:
        return left_dataset
    else:
        raise ValueError(f'{dataset} is not a valid dataset')


def domains():
    doms = {}
    for d in datasets:
        doms[d] = domain(d)
    return doms


def codomains():
    codoms = {}
    for d in datasets:
        codoms[d] = codomain(d)
    return codoms


d3_model_name = '3DEHAM'
d4_model_name = '4DEHAM'
d3_with_distance = False

sample_size = 2 * max(datasets_to_domains.values()) - 1
early_threshold = sample_size
presence_iterations = 2 * sample_size
mean_matches = 1
stdv_matches = 0
use_percentiles = True
minimum_percentile = 0.5
maximum_percentile = 99.5
project_same = 0
project_logistic = 1
project_maximum = 2
project_prototype = 3
projection_transform = project_same

recall_with_sampling_n_search = 0
recall_with_protos = 1
recall_with_correct_proto = 2
recall_with_cue = 3
sampling_without_search = False

sequence_length = 10
sequence_recall_fill = 64
sequence_recall_method = recall_with_sampling_n_search

# Directory where all results are stored.
data_path = 'data'
run_path = 'runs'
idx_digits = 3
prep_data_fname = 'prep_data.npy'
pred_noised_data_fname = 'prep_noised_data.npy'
prep_labels_fname = 'prep_labels.npy'

image_path = 'images'
testing_path = 'test'
memories_path = 'memories'
prototypes_path = 'prototypes'
dreams_path = 'dreams'

data_prefix = 'data-'
labels_prefix = 'labels-'
features_prefix = 'features-'
memories_prefix = 'memories-'
noised_prefix = 'mem_noised-'
proto_prefix = 'proto-'
prototypes_prefix = 'prototypes-'
mem_conf_prefix = 'mem_confrix-'
model_prefix = 'model-'
recognition_prefix = 'recognition-'
recog_noised_prefix = 'recog_noised-'
weights_prefix = 'weights-'
weights_noised_prefix = 'weights-noised-'
classification_prefix = 'classification-'
stats_prefix = 'model_stats-'
learn_params_prefix = 'learn_params-'
memory_parameters_prefix = 'mem_params'
chosen_prefix = 'chosen-'
distance_prefix = 'distance-'
fstats_prefix = 'feature_stats-'
sequence_prefix = 'seq-'

balanced_data = 'balanced'
seed_data = 'seed'
learning_data_seed = 'seed_balanced'
learning_data_learned = 'learned'

# Categories suffixes.
original_suffix = '-original'
training_suffix = '-training'
filling_suffix = '-filling'
testing_suffix = '-testing'
noised_suffix = '-noised'
prod_noised_suffix = '-prod_noised'
memories_suffix = '-memories'
proto_suffix = '-proto'

proto_kind_constructed = 'constructed'
proto_kind_extracted = 'extracted'
proto_kind_fill_recalled = 'recalled'
proto_kind_test_recalled = 'tested'

proto_kinds = [
    proto_kind_constructed,
    proto_kind_extracted,
    proto_kind_fill_recalled,
    proto_kind_test_recalled,
]

proto_labels = {
    proto_kind_constructed: 'Constructed',
    proto_kind_extracted: 'Randomly extracted',
    proto_kind_fill_recalled: 'Recalled with filling corpus',
    proto_kind_test_recalled: 'Recalled with testing corpus',
}
proto_formats = ['r--o', 'b:v', 'g-s', 'y-.d']


def __getattr__(name):
    if name == 'constructed_suffix':
        return proto_kind_suffix(proto_kind_constructed)
    elif name == 'extracted_suffix':
        return proto_kind_suffix(proto_kind_extracted)
    elif name == 'recall_filled_suffix':
        return proto_kind_suffix(proto_kind_fill_recalled)
    elif name == 'recall_tested_suffix':
        return proto_kind_suffix(proto_kind_test_recalled)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Model suffixes.
encoder_suffix = '-encoder'
classifier_suffix = '-classifier'
decoder_suffix = '-decoder'
memory_suffix = '-memory'

data_suffix = '_X'
labels_suffix = '_Y'
matrix_suffix = '-confrix'

search_suffix = '-search'
protos_suffix = '-protos'
correct_proto_suffix = '-correct_proto'
cue_suffix = '-cue'

means_suffix = '-means'
stdvs_suffix = '-stdvs'


def recall_suffix(n: int, proto_kind_suffix=None):
    if (n < 0) or (n >= len(recall_suffix.suffixes)):
        raise ValueError(f'There is no suffix with {n} index.')
    suffix = recall_suffix.suffixes[n]
    if (n == recall_with_protos) or (n == recall_with_correct_proto):
        if proto_kind_suffix is None:
            raise ValueError(f'Suffix cannot be None for recall method {n}')
        else:
            suffix += proto_kind_suffix
    return suffix


recall_suffix.suffixes = [
    search_suffix,
    protos_suffix,
    correct_proto_suffix,
    cue_suffix,
]

agreed_suffix = '-agr'
original_suffix = '-ori'
amsystem_suffix = '-ams'
nnetwork_suffix = '-rnn'
learning_suffixes = [
    [original_suffix],
    [agreed_suffix],
    [amsystem_suffix],
    [nnetwork_suffix],
    [original_suffix, amsystem_suffix],
]


def dataset_suffix(dataset):
    return '-' + dataset


n_folds = 1
n_jobs = 4
random_string_length = 30
dreaming_cycles = 6

nn_training_percent = 0.70
am_filling_percent = 0.20
am_testing_percent = 0.10
# Proportion of testing data used for exploring (preliminary results)
exploration_ratio = 1.0
noise_percent = 50

n_labels = 10
labels_per_memory = 1
all_labels = list(range(n_labels))

precision_idx = 0
recall_idx = 1
accuracy_idx = 2
entropy_idx = 3
no_response_idx = 4
no_correct_response_idx = 5
correct_response_idx = 6
n_behaviours = 7

memory_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
memory_fills = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0]
n_best_memory_sizes = 3
n_samples = 10
learned_data_groups = 6

iota_default = 0.0
kappa_default = 0.0
xi_default = 0.0
sigma_default = 0.1
params_defaults = [iota_default, kappa_default, xi_default, sigma_default]
iota_idx = 0
kappa_idx = 1
xi_idx = 2
sigma_idx = 3


class ExperimentSettings:
    def __init__(self, *, params=None, iota=None, kappa=None, xi=None, sigma=None):
        if params is None:
            self.mem_params = np.array(params_defaults, dtype=float)
        else:
            # If not None, it must be a one dimensional array.
            assert isinstance(params, np.ndarray)
            assert params.ndim == 1
            # The dimension must have four elements
            # iota, kappa, xi, sigma
            shape = params.shape
            assert shape[0] == 4
            self.mem_params = np.array(params, dtype=float, copy=True)
        if iota is not None:
            self.mem_params[iota_idx] = iota
        if kappa is not None:
            self.mem_params[kappa_idx] = kappa
        if xi is not None:
            self.mem_params[xi_idx] = xi
        if sigma is not None:
            self.mem_params[sigma_idx] = sigma

    @property
    def xi(self):
        return self.mem_params[xi_idx]

    @property
    def iota(self):
        return self.mem_params[iota_idx]

    @property
    def kappa(self):
        return self.mem_params[kappa_idx]

    @property
    def sigma(self):
        return self.mem_params[sigma_idx]

    def __str__(self):
        s = '{Parameters: ' + str(self.mem_params) + '}'
        return s


def domain(dataset):
    return datasets_to_domains[dataset]


def codomain(dataset):
    return datasets_to_codomains[dataset]


def print_warning(*s):
    print('WARNING:', *s, file=sys.stderr)


def print_error(*s):
    print('ERROR:', *s, file=sys.stderr)


counters_times = {}


def set_counter():
    name = get_random_string()
    counters_times[name] = time.time()
    return name


def print_counter(n, every, step=1, symbol='.', prefix='', name=None):
    if n == 0:
        return
    e = n % every
    s = n % step
    if (e != 0) and (s != 0):
        return
    counter = symbol
    if e == 0:
        suffix = ''
        if (name is not None) and (name in counters_times.keys()):
            t = time.time()
            suffix = f' {t- counters_times[name]}'
            counters_times[name] = t
        counter = ' ' + prefix + str(n) + suffix + ' '
    print(counter, end='', flush=True)


def get_random_string():
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    random_string = ''.join(random.choice(letters) for i in range(random_string_length))
    return random_string


def int_suffix(n, prefix=None):
    prefix = '-' if prefix is None else '-' + prefix + '_'
    return prefix + str(n).zfill(3)


def float_suffix(x, prefix=None):
    prefix = '' if prefix is None else '-' + prefix + '_'
    return prefix + f'{x:.2f}'


def extended_suffix(extended):
    return '-ext' if extended else ''


def numeric_suffix(prefix, value):
    return '-' + prefix + '_' + str(value).zfill(3)


def fold_suffix(fold):
    return '' if fold is None else int_suffix(fold, 'fld')


def learned_suffix(learned):
    return int_suffix(learned, 'lrn')


def stage_suffix(stage):
    return int_suffix(stage, 'stg')


def msize_suffix(msize):
    return int_suffix(msize, 'msz')


def sigma_suffix(sigma):
    return float_suffix(sigma, 'sgm')


def label_suffix(label):
    return int_suffix(label, 'lbl')


def dream_depth_suffix(cycle):
    return numeric_suffix('dph', cycle)


def proto_kind_suffix(kind):
    if kind in proto_kinds:
        return '-' + kind
    raise ValueError(f'{kind} is not a prototype kind')


def get_name_w_suffix(prefix):
    suffix = ''
    return prefix + suffix


def get_full_name(prefix, es):
    if es is None:
        return prefix
    name = get_name_w_suffix(prefix)
    return name


# Currently, names include nothing about experiment settings.
def model_name(dataset, es):
    return model_prefix + dataset


def stats_model_name(dataset, es):
    return stats_prefix + dataset


def classification_name(dataset, es):
    return classification_prefix + dataset


def data_name(dataset, es):
    return data_prefix + dataset


def features_name(dataset, es):
    return features_prefix + dataset


def labels_name(dataset, es):
    return labels_prefix + dataset


def recall_labels_name(dataset, es):
    return memories_prefix + labels_prefix + dataset


def memories_name(dataset, es):
    return memories_prefix + data_prefix + dataset


def prototypes_name(dataset, es):
    return prototypes_prefix + dataset


def confrix_name(dataset, es):
    return mem_conf_prefix + dataset


def noised_memories_name(dataset, es):
    return noised_prefix + dataset


def recognition_name(dataset, es):
    return recognition_prefix + dataset


def noised_recog_name(dataset, es):
    return recog_noised_prefix + dataset


def weights_name(dataset, es):
    return weights_prefix + dataset


def noised_weights_name(dataset, es):
    return weights_noised_prefix + dataset


def learn_params_name(dataset, es):
    return learn_params_prefix + dataset


def distance_name(dataset, es):
    return distance_prefix + dataset


def fstats_name(dataset, es):
    return fstats_prefix + dataset


def mem_params_name(es):
    return memory_parameters_prefix


def dirname(path):
    match = re.search('[^/]*$', path)
    if match is None:
        return path
    tuple = os.path.splitext(match.group(0))
    return os.path.dirname(path) if tuple[1] else path


def create_directory(path):
    try:
        os.makedirs(path)
        print(f'Directory {path} created.')
    except FileExistsError:
        pass


def filename(name_prefix, es=None, fold=None, extension=''):
    """Returns a file name in run_path directory with a given extension and an index"""
    # Create target directory & all intermediate directories if don't exists
    create_directory(run_path)
    return (
        run_path + '/' + get_full_name(name_prefix, es) + fold_suffix(fold) + extension
    )


def csv_filename(name_prefix, es=None, fold=None):
    return filename(name_prefix, es, fold, '.csv')


def data_filename(name_prefix, es=None, fold=None):
    return filename(name_prefix, es, fold, '.npy')


def json_filename(name_prefix, es):
    return filename(name_prefix, es, extension='.json')


def pickle_filename(name_prefix, es=None, fold=None):
    return filename(name_prefix, es, fold, '.pkl')


def picture_filename(name_prefix, es=None, fold=None):
    return filename(name_prefix, es, fold, extension='.svg')


def image_filename(prefix, idx, label, classif=None, suffix='', es=None, fold=None):
    name_prefix = os.path.join(
        image_path,
        prefix,
        str(label).zfill(3)
        + '_'
        + str(idx).zfill(5)
        + (('_' + str(classif).zfill(3)) if classif is not None else '')
        + suffix,
    )
    return filename(name_prefix, es, fold, extension='.png')


def learned_data_filename(suffix, es, fold):
    prefix = learning_data_learned + suffix + data_suffix
    return data_filename(prefix, es, fold)


def learned_labels_filename(suffix, es, fold):
    prefix = learning_data_learned + suffix + labels_suffix
    return data_filename(prefix, es, fold)


def seed_data_filename():
    return data_filename(learning_data_seed + data_suffix)


def seed_labels_filename():
    return data_filename(learning_data_seed + labels_suffix)


def model_filename(name_prefix, es, fold):
    return filename(name_prefix, es, fold)


def encoder_filename(name_prefix, es, fold):
    return filename(name_prefix + encoder_suffix, es, fold)


def classifier_filename(name_prefix, es, fold):
    return filename(name_prefix + classifier_suffix, es, fold)


def decoder_filename(name_prefix, es, fold):
    return filename(name_prefix + decoder_suffix, es, fold)


def memory_confrix_filename(name_prefix, fill, es):
    prefix = name_prefix + int_suffix(fill, 'fll')
    return data_filename(prefix, es)


def recog_filename(name_prefix, es, fold):
    return csv_filename(name_prefix, es, fold)


def testing_image_filename(dir, idx, label, es, fold):
    return image_filename(dir, idx, label, suffix=original_suffix, es=es, fold=fold)


def prod_testing_image_filename(dir, idx, label, es, fold):
    return image_filename(dir, idx, label, suffix=testing_suffix, es=es, fold=fold)


def memory_image_filename(dir, name, idx, label, classif, es, fold):
    dirname = os.path.join(dir, name) if len(name) > 0 else dir
    return image_filename(dirname, idx, label, classif, memory_suffix, es, fold)


def dream_image_filename(dir, initial_label, depth, label):
    name = (
        image_path
        + '/'
        + dir
        + '/'
        + sequence_prefix
        + label_suffix(initial_label)
        + dream_depth_suffix(depth)
        + label_suffix(label)
    )
    return filename(name, extension='.png')


def mean_idx(m):
    return m


def std_idx(m):
    return m + 1


def padding_cropping(data, n_frames):
    frames, _ = data.shape
    df = frames - n_frames
    if df < 0:
        return []
    elif df == 0:
        return [data]
    else:
        features = []
        for i in range(df + 1):
            features.append(data[i : i + n_frames, :])
        return features


def print_csv(data):
    writer = csv.writer(sys.stdout)
    if np.ndim(data) == 1:
        writer.writerow(data)
    else:
        writer.writerows(data)
