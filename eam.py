# Copyright [2020-23] Luis Alberto Pineda Cortés, Gibrán Fuentes Pineda,
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

"""Entropic Hetero-Associative Memory Experiments

Usage:
    eam -h | --help
    eam (-n [<dataset>] | -f [<dataset>] | -d [<dataset>] | -s [<dataset>] | -c [<dataset>] | -e | -r | -p <kind> | -P <kind> | -q | -u)
        [--relsmean=MEAN] [--relsstdv=STDV] [--runpath=PATH] [ -l (en | es) ]

Options:
    -h    Show this screen.

    -f    Generates features for one dataset view or for both when dataset is omitted.
    -c    Characterizes features for one dataset view or for both when dataset is omitted.
    -d    Calculates feature distances for one dataset view or for both when dataset is omitted.
    -s    Runs separated tests for one dataset view or for both when dataset is omitted.
    -e    Evaluates recognition of hetero-associations.
    -r    Evaluates hetero-recalling using search.
    -p    Validates hetero-recalling using prototypes.
  -P    Validates hetero-recalling using correct prototype.
  -q    Validates hetero-recalling using cue.
  -u    Generates sequences of memories
  --relsmean=MEAN   Average number of relations per data element.
  --relsstdv=STDV   Standard deviation of the number of relations per data element.
  --runpath=PATH    Path to directory where everything will be saved [default: runs]
  -l    Chooses Language for graphs.
"""

import os
import sys
import time
import gc
import math
import gettext
import json
import random
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn
from docopt import docopt

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import png
except ImportError:
    png = None

import commons
import qudeq
import neural_net
from associative import AssociativeMemory
from hetero_associative_4d import HeteroAssociativeMemory4D as HeteroAssociativeMemory
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

sys.setrecursionlimit(10000)

# This allows the production of graphs in multiple languages
# (English is the default).
gettext.bindtextdomain('eam', 'locale')
gettext.textdomain('eam')
_ = gettext.gettext

# Categories in binary confussion matrix
TP = (0, 0)
FN = (0, 1)
FP = (1, 0)
TN = (1, 1)


def plot_metric_pair_graph(
    metric_a_mean,
    metric_b_mean,
    ent_mean,
    metric_a_std,
    metric_b_std,
    dataset,
    es,
    metric_a_label,
    metric_b_label,
    acc_mean=None,
    acc_std=None,
    prefix='',
    xlabels=None,
    xtitle=None,
    ytitle=None,
):
    plt.figure(figsize=(6.4, 4.8))
    full_length = 100.0
    step = 0.1
    if xlabels is None:
        xlabels = commons.memory_sizes
    main_step = full_length / len(xlabels)
    x = np.arange(0, full_length, main_step)
    xmax = full_length - main_step + step
    ymax = full_length + 2

    metric_a_mean = np.nan_to_num(metric_a_mean, copy=False)
    metric_b_mean = np.nan_to_num(metric_b_mean, copy=False)
    metric_a_std = np.nan_to_num(metric_a_std, copy=False)
    metric_b_std = np.nan_to_num(metric_b_std, copy=False)

    plt.errorbar(
        x,
        metric_a_mean,
        fmt='r-o',
        yerr=metric_a_std,
        capsize=3,
        label=metric_a_label,
    )
    plt.errorbar(
        x,
        metric_b_mean,
        fmt='b--s',
        yerr=metric_b_std,
        capsize=3,
        label=metric_b_label,
    )
    if (acc_mean is not None) and (acc_std is not None):
        plt.errorbar(
            x, acc_mean, fmt='g--d', yerr=acc_std, capsize=3, label=_('Accuracy')
        )
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.xticks(x, xlabels)
    if xtitle is None:
        xtitle = _('Range Quantization Levels')
    if ytitle is None:
        ytitle = _('Percentage')

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend(loc='best')
    plt.grid(True)

    entropy_labels = [str(e) for e in np.around(ent_mean, decimals=1)]

    cmap = mpl.colors.LinearSegmentedColormap.from_list('mycolors', ['cyan', 'purple'])
    z = [[0, 0], [0, 0]]
    levels = np.arange(0.0, xmax, step)
    cs3 = plt.contourf(z, levels, cmap=cmap)

    cbar = plt.colorbar(cs3, orientation='horizontal')
    cbar.set_ticks(x)
    cbar.ax.set_xticklabels(entropy_labels)
    cbar.set_label(_('Entropy'))

    fname = prefix + 'graph_metrics-' + dataset + _('-english')
    graph_filename = commons.picture_filename(fname, es)
    plt.savefig(graph_filename, dpi=600)
    plt.close()


def plot_prerec_graph(
    pre_mean,
    rec_mean,
    ent_mean,
    pre_std,
    rec_std,
    dataset,
    es,
    acc_mean=None,
    acc_std=None,
    prefix='',
    xlabels=None,
    xtitle=None,
    ytitle=None,
):
    """Plots a precision and recall graph.

    It can also include accuracy, and it can be used to to graph any
    measure in percentages.
    """
    pre_mean = np.nan_to_num(pre_mean, copy=False, nan=100.0)
    plot_metric_pair_graph(
        pre_mean,
        rec_mean,
        ent_mean,
        pre_std,
        rec_std,
        dataset,
        es,
        _('Precision'),
        _('Recall'),
        acc_mean=acc_mean,
        acc_std=acc_std,
        prefix=prefix,
        xlabels=xlabels,
        xtitle=xtitle,
        ytitle=ytitle,
    )


def plot_behs_graph(
    no_response, no_correct, correct, dataset, es, xtags=None, prefix=''
):
    """Plots the behaviours graph.

    A behaviour graph is a stacked bars graph that representes three exclusive types
    of all possible responses: no response, no correct response, and correct response.
    """
    plt.clf()
    print('Behaviours: ')
    print(f'No response: {no_response}')
    print(f'No correct response: {no_correct}')
    print(f'Correct response: {correct}')
    for i in range(len(no_response)):
        total = (no_response[i] + no_correct[i] + correct[i]) / 100.0
        no_response[i] /= total
        no_correct[i] /= total
        correct[i] /= total
    full_length = 100.0
    step = 0.1
    if xtags is None:
        xtags = commons.memory_sizes
    main_step = full_length / len(xtags)
    x = np.arange(0.0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step
    ymax = full_length
    width = 5  # the width of the bars: can also be len(x) sequence

    plt.bar(x, correct, width, label=_('Correct response'))
    cumm = np.array(correct)
    plt.bar(x, no_correct, width, bottom=cumm, label=_('No correct response'))
    cumm += np.array(no_correct)
    plt.bar(x, no_response, width, bottom=cumm, label=_('No response'))

    plt.xlim(-width, xmax + width)
    plt.ylim(0.0, ymax)
    plt.xticks(x, xtags)

    plt.xlabel(_('Range Quantization Levels'))
    plt.ylabel(_('Labels'))

    plt.legend(loc='best')
    plt.grid(axis='y')

    fname = prefix + 'graph_behaviours-' + dataset + _('-english')
    graph_filename = commons.picture_filename(fname, es)
    plt.savefig(graph_filename, dpi=600)
    plt.close()


def plot_histo_bar(
    frequencies, dataset, es, xtags=None, xlabel='Categories', label=None, name=''
):
    """Plots a histogram graph."""
    plt.clf()
    # full_length = 100.0
    # step = 0.1
    x = range(len(frequencies))
    if xtags is None:
        xtags = x
    # main_step = full_length/len(frequencies)
    # x = np.arange(0.0, full_length, main_step)
    frequencies = 100 * frequencies / np.sum(frequencies)
    # One main step less because levels go on sticks, not
    # on intervals.
    # xmax = full_length - main_step + step
    ymax = 100.0
    # width = 5       # the width of the bars: can also be len(x) sequence
    # plt.bar(x, frequencies, width, label=label)
    plt.bar(x, frequencies, label=label)
    # plt.xlim(-width, xmax + width)
    plt.ylim(0.0, ymax)
    plt.xticks(range(len(frequencies)), xtags)

    plt.xlabel(xlabel)
    plt.ylabel('Percentage')

    plt.legend(loc='best')
    plt.grid(axis='y')

    fname = name + '-histogram-' + dataset + _('-english')
    graph_filename = commons.picture_filename(fname, es)
    plt.savefig(graph_filename, dpi=600)
    plt.close()


def plot_features_graph(domain, means, stdevs, labels, dataset, es):
    """Draws the characterist shape of features per label.

    The graph is a dots and lines graph with error bars denoting standard deviations.
    """
    ymin = np.PINF
    ymax = np.NINF
    for i in commons.all_labels:
        for j in range(2):
            yn = (means[j, i] - stdevs[j, i]).min()
            yx = (means[j, i] + stdevs[j, i]).max()
            ymin = ymin if ymin < yn else yn
            ymax = ymax if ymax > yx else yx
    main_step = 100.0 / domain
    xrange = np.arange(0, 100, main_step)
    fmts = commons.proto_formats
    for i in commons.all_labels:
        plt.clf()
        plt.figure(figsize=(12, 5))
        for j in range(len(labels)):
            plt.errorbar(
                xrange,
                means[j, i],
                fmt=fmts[j],
                yerr=stdevs[j, i],
                ecolor='silver',
                elinewidth=0.5,
                capsize=3,
                label=labels[j],
            )
        plt.xlim(0, 100)
        plt.ylim(ymin, ymax)
        plt.xticks(xrange, labels='')
        plt.xlabel(_('Features'))
        plt.ylabel(_('Values'))
        plt.legend(loc='best')
        plt.grid(axis='y')
        filename = (
            commons.features_name(dataset, es) + '-' + str(i).zfill(3) + _('-english')
        )
        plt.savefig(commons.picture_filename(filename, es), dpi=600)
    plt.close()


def plot_legacy_confusion_matrix(
    matrix, tags, dataset, es, prefix='', vmin=0.0, vmax=None
):
    """Plots a legacy confusion matrix of labels vs predictions."""
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))
    if vmax is None:
        vmax = np.max(matrix)
    seaborn.heatmap(
        matrix,
        xticklabels=tags,
        yticklabels=tags,
        vmin=vmin,
        vmax=vmax,
        annot=False,
        cmap='Blues',
    )
    plt.xlabel(_('Prediction'))
    plt.ylabel(_('Label'))
    fname = prefix + commons.matrix_suffix + '-' + dataset + _('-english')
    filename = commons.picture_filename(fname, es)
    plt.savefig(filename, dpi=600)
    plt.close()


def plot_confusion_matrix(matrix, tags, dataset, es, prefix='', vmin=0.0, vmax=None):
    """Backward-compatible alias for legacy confusion matrices."""
    plot_legacy_confusion_matrix(
        matrix,
        tags,
        dataset,
        es,
        prefix=prefix,
        vmin=vmin,
        vmax=vmax,
    )


def plot_relation_heatmap(
    relation, prefix, xlabel='Characteristics', ylabel='Values', es=None, fold=None
):
    """Plots a generic relation matrix as a heat map."""
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))
    seaborn.heatmap(np.transpose(relation), annot=False, cmap='coolwarm')
    plt.xlabel(_(xlabel))
    plt.ylabel(_(ylabel))
    if es is None:
        es = commons.ExperimentSettings()
    filename = commons.picture_filename(prefix, es, fold)
    plt.savefig(filename, dpi=600)
    plt.close()


def plot_projection_relation(relation, prefix, es=None, fold=None):
    """Plots a projection/relation matrix produced during recall."""
    plot_relation_heatmap(relation, prefix, es=es, fold=fold)


def plot_distances(distances, prefix, es=None, fold=None):
    """Plots a matrix of distances between categories (labels) in a dataset."""
    plot_relation_heatmap(
        distances,
        prefix,
        xlabel='Label',
        ylabel='Label',
        es=es,
        fold=fold,
    )


def plot_relation(
    relation, prefix, xlabel='Characteristics', ylabel='Values', es=None, fold=None
):
    """Backward-compatible alias for relation heat maps."""
    plot_relation_heatmap(
        relation,
        prefix,
        xlabel=xlabel,
        ylabel=ylabel,
        es=es,
        fold=fold,
    )


def features_distance(f, g):
    """Calculates euclidean distance between two arrays of features."""
    return np.linalg.norm(f - g)


def row_cosine(a, b):
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    denom = np.where((a_norm * b_norm) == 0.0, 1.0, a_norm * b_norm)
    return np.sum(a * b, axis=1) / denom


def cosine_to_score(cosine):
    return np.clip((cosine + 1.0) / 2.0, 0.0, 1.0)


def add_roundtrip_metrics(metrics, recalled_embeddings, targets, dataset):
    metrics['mean_cosine_roundtrip'] = np.nan
    metrics['cosine_roundtrip_score'] = 0.0

    if len(recalled_embeddings) == 0:
        return

    roundtrip_embeddings = neural_net.roundtrip_embeddings(recalled_embeddings, dataset)
    valid = np.all(np.isfinite(roundtrip_embeddings), axis=1)
    valid &= np.all(np.isfinite(targets), axis=1)
    if np.any(valid):
        roundtrip_embeddings = roundtrip_embeddings[valid]
        valid_targets = targets[valid]
        roundtrip_cosines = row_cosine(roundtrip_embeddings, valid_targets)
        metrics['mean_cosine_roundtrip'] = float(np.mean(roundtrip_cosines))
        metrics['cosine_roundtrip_score'] = float(
            cosine_to_score(metrics['mean_cosine_roundtrip'])
        )


def stats_measures(filling_features, filling_labels, testing_features, testing_labels):
    """Calculates mean and standard deviation for each feature element per label."""
    filling_fpl = {}
    testing_fpl = {}
    for label in range(commons.n_labels):
        filling_fpl[label] = []
        testing_fpl[label] = []
    for features, label in zip(filling_features, filling_labels):
        filling_fpl[label].append(features)
    for features, label in zip(testing_features, testing_labels):
        testing_fpl[label].append(features)
    means = np.zeros((commons.n_labels + 1, 2), dtype=float)
    stdvs = np.zeros((commons.n_labels + 1, 2), dtype=float)
    for label in range(commons.n_labels):
        means[label, 0] = np.mean(filling_fpl[label])
        means[label, 1] = np.mean(testing_fpl[label])
        stdvs[label, 0] = np.std(filling_fpl[label])
        stdvs[label, 1] = np.std(testing_fpl[label])
    means[commons.n_labels, 0] = np.mean(filling_features)
    means[commons.n_labels, 1] = np.mean(testing_features)
    stdvs[commons.n_labels, 0] = np.std(filling_features)
    stdvs[commons.n_labels, 1] = np.std(testing_features)
    return means, stdvs


def distance_matrices(
    filling_features, filling_labels, testing_features, testing_labels
):
    ff_dist = {}
    ft_dist = {}
    for l1 in range(commons.n_labels):
        for l2 in range(commons.n_labels):
            ff_dist[(l1, l2)] = []
            ft_dist[(l1, l2)] = []
    f_len = len(filling_labels)
    t_len = len(testing_labels)
    counter = 0
    for i in range(f_len):
        for j in range(f_len):
            if i != j:
                l1 = filling_labels[i]
                l2 = filling_labels[j]
                d = features_distance(filling_features[i], filling_features[j])
                ff_dist[(l1, l2)].append(d)
        for j in range(t_len):
            l1 = filling_labels[i]
            l2 = testing_labels[j]
            d = features_distance(filling_features[i], testing_features[j])
            ft_dist[(l1, l2)].append(d)
        commons.print_counter(counter, 1000, 100)
        counter += 1
    print(' end.')
    ff_means = np.zeros((commons.n_labels, commons.n_labels), dtype=float)
    ff_stdvs = np.zeros((commons.n_labels, commons.n_labels), dtype=float)
    ft_means = np.zeros((commons.n_labels, commons.n_labels), dtype=float)
    ft_stdvs = np.zeros((commons.n_labels, commons.n_labels), dtype=float)
    for l1 in range(commons.n_labels):
        for l2 in range(commons.n_labels):
            mean = np.mean(ff_dist[(l1, l2)])
            stdv = np.std(ff_dist[(l1, l2)])

            ff_means[l1, l2] = mean
            ff_stdvs[l1, l2] = stdv
            mean = np.mean(ft_dist[(l1, l2)])
            stdv = np.std(ft_dist[(l1, l2)])
            ft_means[l1, l2] = mean
            ft_stdvs[l1, l2] = stdv
    means = np.concatenate((ff_means, ft_means), axis=1)
    stdvs = np.concatenate((ff_stdvs, ft_stdvs), axis=1)
    return means, stdvs


def features_per_fold(dataset, es, fold):
    suffix = commons.filling_suffix
    filling_features_filename = commons.features_name(dataset, es) + suffix
    filling_features_filename = commons.data_filename(
        filling_features_filename, es, fold
    )
    filling_labels_filename = commons.labels_name(dataset, es) + suffix
    filling_labels_filename = commons.data_filename(filling_labels_filename, es, fold)

    suffix = commons.testing_suffix
    testing_features_filename = commons.features_name(dataset, es) + suffix
    testing_features_filename = commons.data_filename(
        testing_features_filename, es, fold
    )
    testing_labels_filename = commons.labels_name(dataset, es) + suffix
    testing_labels_filename = commons.data_filename(testing_labels_filename, es, fold)

    filling_features = np.load(filling_features_filename)
    filling_labels = np.load(filling_labels_filename)
    testing_features = np.load(testing_features_filename)
    testing_labels = np.load(testing_labels_filename)
    return filling_features, filling_labels, testing_features, testing_labels


def match_labels(features, labels, half=False):
    right_features = []
    right_labels = []
    used_idx = set()
    last = 0
    if len(labels[commons.left_dataset]) < len(labels[commons.right_dataset]):
        smaller_ds = commons.left_dataset
        larger_ds = commons.right_dataset
    else:
        smaller_ds = commons.right_dataset
        larger_ds = commons.left_dataset
    # Assuming ten clases on each dataset.
    midx = round(len(labels[smaller_ds]) * 4.0 / 9.0)
    matching_labels = labels[smaller_ds][:midx] if half else labels[smaller_ds]
    counter = 0
    print(
        f'Matching {len(labels[smaller_ds])} in {smaller_ds} '
        + f' with {len(labels[larger_ds])} in {larger_ds}:'
    )
    for left_label in matching_labels:
        while last in used_idx:
            last += 1
        i = last
        found = False
        for right_feats, right_label in zip(
            features[larger_ds][i:], labels[larger_ds][i:]
        ):
            if (i not in used_idx) and (left_label == right_label):
                right_features.append(right_feats)
                right_labels.append(right_label)
                used_idx.add(i)
                found = True
                break
            i += 1
        if not found:
            break
        counter += 1
        commons.print_counter(counter, 1000, 100, symbol='-')
    print(' end')
    if half:
        i = 0
        for right_feats, right_label in zip(features[larger_ds], labels[larger_ds]):
            if i not in used_idx:
                right_features.append(right_feats)
                right_labels.append(right_label)
            i += 1
    n = len(right_features)
    features[smaller_ds] = features[smaller_ds][:n]
    labels[smaller_ds] = labels[smaller_ds][:n]
    features[larger_ds] = np.array(right_features, dtype=int)
    labels[larger_ds] = np.array(right_labels, dtype=int)


def describe(features, labels):
    left_ds = commons.left_dataset
    right_ds = commons.right_dataset
    left_n = len(labels[left_ds])
    right_n = len(labels[right_ds])
    print(f'Elements in left dataset: {left_n}')
    print(f'Elements in right dataset: {right_n}')
    minimum = left_n if left_n < right_n else right_n
    matching = 0
    left_counts = np.zeros((commons.n_labels), dtype=int)
    right_counts = np.zeros((commons.n_labels), dtype=int)
    for i in range(minimum):
        left_label = labels[left_ds][i]
        right_label = labels[right_ds][i]
        left_counts[left_label] += 1
        right_counts[right_label] += 1
        matching += left_label == right_label
    print(f'Matching labels: {matching}')
    print(f'Unmatching labels: {minimum - matching}')
    print(f'Left labels counts: {left_counts}')
    print(f'Right labels counts: {right_counts}')


def show_weights_stats(weights):
    w = {}
    conds = ['TP', 'FN', 'FP', 'TN']
    for c in conds:
        if len(weights[c]) == 0:
            w[c] = (0.0, 0.0)
        else:
            mean = np.mean(weights[c])
            stdv = np.std(weights[c])
            w[c] = (mean, stdv)
    print(f'Weights: {w}')


def freqs_to_values(freqs):
    xs = []
    for v, f in enumerate(freqs):
        for _ in range(f):
            xs.append(v)
    random.shuffle(xs)
    return xs


def normality_test(relation):
    ps = []
    for column in relation:
        xs = freqs_to_values(column)
        shapiro_test = scipy.stats.shapiro(xs)
        ps.append(shapiro_test.pvalue)
    return np.mean(ps), np.std(ps)


def statistics_per_fold(dataset, es, fold):
    filling_features, filling_labels, testing_features, testing_labels = (
        features_per_fold(dataset, es, fold)
    )
    print(f'Calculating statistics for fold {fold}')
    return stats_measures(
        filling_features, filling_labels, testing_features, testing_labels
    )


def distances_per_fold(dataset, es, fold):
    filling_features, filling_labels, testing_features, testing_labels = (
        features_per_fold(dataset, es, fold)
    )

    print(f'Calculating distances for fold {fold}')
    means, stdvs = distance_matrices(
        filling_features, filling_labels, testing_features, testing_labels
    )
    return means, stdvs


def load_dataset_feats_n_labels(dataset, fold, es):
    features_filename = commons.features_name(dataset, es) + commons.filling_suffix
    features_filename = commons.data_filename(features_filename, es, fold)
    labels_filename = commons.labels_name(dataset, es) + commons.filling_suffix
    labels_filename = commons.data_filename(labels_filename, es, fold)
    filling_features = np.load(features_filename)
    filling_labels = np.load(labels_filename)
    features_filename = commons.features_name(dataset, es) + commons.testing_suffix
    features_filename = commons.data_filename(features_filename, es, fold)
    labels_filename = commons.labels_name(dataset, es) + commons.testing_suffix
    labels_filename = commons.data_filename(labels_filename, es, fold)
    testing_features = np.load(features_filename)
    testing_labels = np.load(labels_filename)
    return filling_features, filling_labels, testing_features, testing_labels


def load_features_n_labels(fold, es):
    filling_features = {}
    filling_labels = {}
    testing_features = {}
    testing_labels = {}
    for dataset in commons.datasets:
        (
            filling_features[dataset],
            filling_labels[dataset],
            testing_features[dataset],
            testing_labels[dataset],
        ) = load_dataset_feats_n_labels(dataset, fold, es)
    return filling_features, filling_labels, testing_features, testing_labels


def recognize_by_memory(eam, tef_rounded, testing_features, msize, qd, dataset):
    recalled_indexes = []
    recalled_memories = []
    total = len(testing_features)

    for idx, features in enumerate(tef_rounded):
        memory, recognized, _ = eam.recall(features)
        if recognized:
            recalled_indexes.append(idx)
            recalled_memories.append(memory)

    recognized = len(recalled_indexes)
    metrics = {
        'recognized': recognized,
        'unknown': total - recognized,
        'recognition_rate': 0.0 if total == 0 else recognized / float(total),
        'mean_cosine': np.nan,
        'cosine_score': 0.0,
        'mean_cosine_roundtrip': np.nan,
        'cosine_roundtrip_score': 0.0,
        'mean_l2': np.nan,
    }
    if recognized > 0:
        recalled_memories = qd.dequantize(np.array(recalled_memories), msize)
        targets = testing_features[recalled_indexes]
        valid = np.all(np.isfinite(recalled_memories), axis=1)
        valid &= np.all(np.isfinite(targets), axis=1)
        if np.any(valid):
            recalled_memories = recalled_memories[valid]
            targets = targets[valid]
            cosines = row_cosine(recalled_memories, targets)
            l2_distances = np.linalg.norm(recalled_memories - targets, axis=1)
            metrics['mean_cosine'] = float(np.mean(cosines))
            metrics['cosine_score'] = float(cosine_to_score(metrics['mean_cosine']))
            metrics['mean_l2'] = float(np.mean(l2_distances))
            add_roundtrip_metrics(metrics, recalled_memories, targets, dataset)

    print(f'Recognized elements: {recognized}/{total}')
    print(f'Unknown elements: {metrics["unknown"]}')
    if np.isnan(metrics['mean_cosine']):
        print('Mean cosine similarity: undefined')
        print('Mean round-trip cosine similarity: undefined')
        print('Mean L2 distance: undefined')
    else:
        print(f'Mean cosine similarity: {metrics["mean_cosine"]:.6f}')
        print(
            'Mean round-trip cosine similarity: '
            + f'{metrics["mean_cosine_roundtrip"]:.6f}'
        )
        print(f'Mean L2 distance: {metrics["mean_l2"]:.6f}')
    return metrics


def recognize_by_hetero_memory(
    hetero_eam: HeteroAssociativeMemory,
    left_eam: AssociativeMemory,
    right_eam: AssociativeMemory,
    tefs,
):
    confrix = np.zeros((2, 2), dtype=int)
    weights = {'TP': [], 'FN': [], 'FP': [], 'TN': []}
    print('Recognizing by hetero memory')
    counter = 0
    left_features = tefs[commons.left_dataset]
    right_features = tefs[commons.right_dataset]
    total = min(len(left_features), len(right_features))
    negative_right = None if total < 2 else np.roll(right_features[:total], 1, axis=0)

    for left_feat, right_feat in zip(
        left_features[:total],
        right_features[:total],
    ):
        _, left_weights = left_eam.recog_weights(left_feat)
        _, right_weights = right_eam.recog_weights(right_feat)
        recognized, weight = hetero_eam.recognize(
            left_feat, right_feat, left_weights, right_weights
        )
        if recognized:
            confrix[TP] += 1
            weights['TP'].append(weight)
        else:
            confrix[FN] += 1
            weights['FN'].append(weight)
        counter += 1
        commons.print_counter(counter, 10000, 1000, symbol='*')

    if negative_right is not None:
        for left_feat, right_feat in zip(left_features[:total], negative_right):
            _, left_weights = left_eam.recog_weights(left_feat)
            _, right_weights = right_eam.recog_weights(right_feat)
            recognized, weight = hetero_eam.recognize(
                left_feat, right_feat, left_weights, right_weights
            )
            if recognized:
                confrix[FP] += 1
                weights['FP'].append(weight)
            else:
                confrix[TN] += 1
                weights['TN'].append(weight)
            counter += 1
            commons.print_counter(counter, 10000, 1000, symbol='*')
    print(' end')
    show_weights_stats(weights)
    print(f'Confusion matrix:\n{confrix}')
    return confrix


def recall_by_hetero_memory_embeddings(
    remembered_dataset,
    recall,
    a_features,
    b_features,
    raw_b_features,
    b_labels,
    msize,
    recall_method,
    mfill,
    qd,
):
    gc.collect()
    indexes = []
    memories = []
    pair_ids = []
    stats = []
    print('Remembering ', end='')
    counter = 0
    counter_name = commons.set_counter()
    for idx, (a_feats, b_feats, pair_id) in enumerate(zip(a_features, b_features, b_labels)):
        memory, recognized, _, relation, s = (
            recall(a_feats, recall_method)
            if recall_method == commons.recall_with_sampling_n_search
            else recall(a_feats, recall_method, euc=b_feats, weights=None, label=pair_id)
        )
        if recognized:
            indexes.append(idx)
            memories.append(memory)
            pair_ids.append(pair_id)
            stats.append(s)
            if random.randrange(100) == 0:
                prefix = (
                    'projection-'
                    + remembered_dataset
                    + '-fill_'
                    + str(int(mfill)).zfill(3)
                    + '-idx_'
                    + str(idx).zfill(5)
                )
                plot_projection_relation(relation, prefix)
        counter += 1
        commons.print_counter(
            counter,
            1000,
            100,
            symbol='+',
            prefix=f'(Recognized : {len(memories)})',
            name=counter_name,
        )
    print(' done')

    recognized = len(indexes)
    total = len(b_labels)
    metrics = {
        'recognized': recognized,
        'unknown': total - recognized,
        'recognition_rate': 0.0 if total == 0 else recognized / float(total),
        'mean_cosine': np.nan,
        'cosine_score': 0.0,
        'mean_cosine_roundtrip': np.nan,
        'cosine_roundtrip_score': 0.0,
        'mean_l2': np.nan,
    }
    recalled = np.zeros((0, raw_b_features.shape[1]), dtype=np.float32)
    if recognized > 0:
        recalled = qd.dequantize(np.array(memories), msize).astype(np.float32)
        targets = raw_b_features[indexes]
        valid = np.all(np.isfinite(recalled), axis=1)
        valid &= np.all(np.isfinite(targets), axis=1)
        if np.any(valid):
            recalled = recalled[valid]
            targets = targets[valid]
            cosines = row_cosine(recalled, targets)
            l2_distances = np.linalg.norm(recalled - targets, axis=1)
            metrics['mean_cosine'] = float(np.mean(cosines))
            metrics['cosine_score'] = float(cosine_to_score(metrics['mean_cosine']))
            metrics['mean_l2'] = float(np.mean(l2_distances))
            add_roundtrip_metrics(metrics, recalled, targets, remembered_dataset)

    print(f'Recognized elements: {recognized}/{total}')
    if np.isnan(metrics['mean_cosine']):
        print('Mean cosine similarity: undefined')
        print('Mean round-trip cosine similarity: undefined')
        print('Mean L2 distance: undefined')
    else:
        print(f'Mean cosine similarity: {metrics["mean_cosine"]:.6f}')
        print(
            'Mean round-trip cosine similarity: '
            + f'{metrics["mean_cosine_roundtrip"]:.6f}'
        )
        print(f'Mean L2 distance: {metrics["mean_l2"]:.6f}')

    metadata = np.array([indexes, pair_ids], dtype=int).transpose() if recognized > 0 else np.zeros((0, 2), dtype=int)
    return metrics, recalled, metadata


def recall_by_hetero_memory(
    remembered_dataset,
    recall,
    eam_origin: AssociativeMemory,
    eam_destination: AssociativeMemory,
    classifier,
    a_features,
    b_features,
    b_labels,
    msize,
    recall_method,
    mfill,
    qd,
    mean_weight,
):
    gc.collect()
    # Each row is a correct label and each column is the prediction, including
    # no recognition.
    confrix = np.zeros((commons.n_labels, commons.n_labels + 1), dtype='int')
    behaviour = np.zeros(commons.n_behaviours, dtype=int)
    indexes = []
    memories = []
    correct = []
    mem_weights = []
    unknown = 0
    unknown_weights = []
    stats = []
    print('Remembering ', end='')
    counter = 0
    counter_name = commons.set_counter()
    for idx, (a_feats, b_feats, label) in enumerate(
        zip(a_features, b_features, b_labels)
    ):
        memory, recognized, weight, relation, s = (
            recall(a_feats, recall_method)
            if recall_method == commons.recall_with_sampling_n_search
            else recall(a_feats, recall_method, euc=b_feats, weights=None, label=label)
        )
        if recognized:
            stats.append(s)
            indexes.append(idx)
            memories.append(memory)
            correct.append(label)
            mem_weights.append(weight)
            if random.randrange(100) == 0:
                prefix = (
                    'projection-'
                    + remembered_dataset
                    + '-fill_'
                    + str(int(mfill)).zfill(3)
                    + '-lbl_'
                    + str(label).zfill(3)
                )
                plot_projection_relation(relation, prefix)
        else:
            unknown += 1
            confrix[label, commons.n_labels] += 1
            unknown_weights.append(weight)
        counter += 1
        commons.print_counter(
            counter,
            1000,
            100,
            symbol='+',
            prefix=f'(Recognized : {len(memories)})',
            name=counter_name,
        )
    print(' done')
    correct_weights = []
    incorrect_weights = []
    correct_stats = []
    incorrect_stats = []
    print('Validating ', end='')
    predictions = []
    if len(memories) > 0:
        memories = qd.dequantize(np.array(memories), msize)
        predictions = np.argmax(classifier.predict(memories), axis=1)
        for label, prediction, weight, s in zip(
            correct, predictions, mem_weights, stats
        ):
            # For calculation of per memory precision and recall
            confrix[label, prediction] += 1
            if label == prediction:
                correct_weights.append(weight)
                correct_stats.append(s)
            else:
                incorrect_weights.append(weight)
                incorrect_stats.append(s)
    print(' done')
    print(' end')
    behaviour[commons.no_response_idx] = unknown
    behaviour[commons.correct_response_idx] = np.sum(
        [confrix[i, i] for i in range(commons.n_labels)]
    )
    behaviour[commons.no_correct_response_idx] = (
        len(b_labels) - unknown - behaviour[commons.correct_response_idx]
    )
    print(f'Confusion matrix:\n{confrix}')
    print(
        f'Behaviour: nr = {behaviour[commons.no_response_idx]}, '
        + f'ir = {behaviour[commons.no_correct_response_idx]}, '
        + f'cr = {behaviour[commons.correct_response_idx]}'
    )
    unknown_weights_mean = (
        0.0 if len(unknown_weights) == 0 else np.mean(unknown_weights / mean_weight)
    )
    unknown_weights_stdv = (
        0.0 if len(unknown_weights) == 0 else np.std(unknown_weights / mean_weight)
    )
    incorrect_weights_mean = (
        0.0 if len(incorrect_weights) == 0 else np.mean(incorrect_weights / mean_weight)
    )
    incorrect_weights_stdv = (
        0.0 if len(incorrect_weights) == 0 else np.std(incorrect_weights / mean_weight)
    )
    correct_weights_mean = (
        0.0 if len(correct_weights) == 0 else np.mean(correct_weights / mean_weight)
    )
    correct_weights_stdv = (
        0.0 if len(correct_weights) == 0 else np.std(correct_weights / mean_weight)
    )
    print(f'Mean weight: {mean_weight}')
    print(
        f'Weights: correct = ({correct_weights_mean}, {correct_weights_stdv}), '
        + f'incorrect = ({incorrect_weights_mean}, {incorrect_weights_stdv}), '
        + f'unknown = ({unknown_weights_mean}, {unknown_weights_stdv})'
    )

    if len(stats) > 0:
        stats = np.array(stats)
        stats_mean = np.mean(stats, axis=0)
        stats_stdv = np.std(stats, axis=0)
        stats_skew = scipy.stats.skew(stats)
        stats_kurt = scipy.stats.kurtosis(stats)
        print(
            f'Stats: mean = {stats_mean}, stdv = {stats_stdv}, '
            + f'skew = {stats_skew}, kurt = {stats_kurt}.'
        )
    else:
        print('Stats not available')
    if len(correct_stats) > 0:
        correct_stats = np.array(correct_stats)
        correct_stats_mean = np.mean(correct_stats, axis=0)
        correct_stats_stdv = np.std(correct_stats, axis=0)
        correct_stats_skew = scipy.stats.skew(correct_stats)
        correct_stats_kurt = scipy.stats.kurtosis(correct_stats)
        print(
            f'Stats of correct: mean = {correct_stats_mean}, stdev = {correct_stats_stdv}, '
            + f'skew = {correct_stats_skew}, kurt = {correct_stats_kurt}.'
        )
    else:
        print('Stats of correct not available')
    if len(incorrect_stats) > 0:
        incorrect_stats = np.array(incorrect_stats)
        incorrect_stats_mean = np.mean(incorrect_stats, axis=0)
        incorrect_stats_stdv = np.std(incorrect_stats, axis=0)
        incorrect_stats_skew = scipy.stats.skew(incorrect_stats)
        incorrect_stats_kurt = scipy.stats.kurtosis(incorrect_stats)
        print(
            f'Stats of incorrect: mean = {incorrect_stats_mean}, stdev = {incorrect_stats_stdv}, '
            + f'skew = {incorrect_stats_skew}, kurt = {incorrect_stats_kurt}.'
        )
    else:
        print('Stats of incorrect not available')
    return confrix, behaviour, memories, indexes, correct, predictions


def remember_by_hetero_memory(
    eam: HeteroAssociativeMemory,
    left_eam: AssociativeMemory,
    right_eam: AssociativeMemory,
    left_classifier,
    right_classifier,
    testing_features,
    testing_labels,
    qudeqs,
    recall_method,
    proto_kind_suffix,
    percent,
    es,
    fold,
):
    left_ds = commons.left_dataset
    right_ds = commons.right_dataset
    rows = commons.codomains()
    confrixes = []
    behaviours = []
    mean_weight = eam.mean
    print('Remembering from left by hetero memory')
    qd = qudeqs[right_ds]
    confrix, behaviour, memories, indexes, correct, predictions = (
        recall_by_hetero_memory(
            right_ds,
            eam.recall_from_left,
            left_eam,
            right_eam,
            right_classifier,
            testing_features[left_ds],
            testing_features[right_ds],
            testing_labels[right_ds],
            rows[right_ds],
            recall_method,
            percent,
            qd,
            mean_weight,
        )
    )
    confrixes.append(confrix)
    behaviours.append(behaviour)
    name = (
        commons.memories_name(left_ds, es)
        + commons.recall_suffix(recall_method, proto_kind_suffix)
        + commons.int_suffix(percent, 'fll')
    )
    filename = commons.data_filename(name, es, fold)
    np.save(filename, memories)
    labels = np.array([indexes, correct, predictions], dtype=int).transpose()
    name = (
        commons.recall_labels_name(left_ds, es)
        + commons.recall_suffix(recall_method, proto_kind_suffix)
        + commons.int_suffix(percent, 'fll')
    )
    filename = commons.data_filename(name, es, fold)
    np.save(filename, labels)
    decode_memories(
        memories,
        indexes,
        correct,
        predictions,
        left_ds,
        percent,
        recall_method,
        proto_kind_suffix,
        es,
        fold,
    )

    print('Remembering from right by hetero memory')
    qd = qudeqs[left_ds]
    confrix, behaviour, memories, indexes, correct, predictions = (
        recall_by_hetero_memory(
            left_ds,
            eam.recall_from_right,
            right_eam,
            left_eam,
            left_classifier,
            testing_features[right_ds],
            testing_features[left_ds],
            testing_labels[left_ds],
            rows[left_ds],
            recall_method,
            percent,
            qd,
            mean_weight,
        )
    )
    confrixes.append(confrix)
    behaviours.append(behaviour)
    name = (
        commons.memories_name(right_ds, es)
        + commons.recall_suffix(recall_method, proto_kind_suffix)
        + commons.int_suffix(percent, 'fll')
    )
    filename = commons.data_filename(name, es, fold)
    np.save(filename, memories)
    labels = np.array([indexes, correct, predictions], dtype=int).transpose()
    name = (
        commons.recall_labels_name(right_ds, es)
        + commons.recall_suffix(recall_method, proto_kind_suffix)
        + commons.int_suffix(percent, 'fll')
    )
    filename = commons.data_filename(name, es, fold)
    np.save(filename, labels)
    decode_memories(
        memories,
        indexes,
        correct,
        predictions,
        right_ds,
        percent,
        recall_method,
        proto_kind_suffix,
        es,
        fold,
    )

    # confrixes has three dimensions: datasets, correct label, prediction.
    confrixes = np.array(confrixes, dtype=int)
    # behaviours has two dimensions: datasets, behaviours.
    behaviours = np.array(behaviours, dtype=int)
    return confrixes, behaviours


def optimum_indexes(precisions, recalls):
    f1s = []
    i = 0
    for p, r in zip(precisions, recalls):
        f1 = 0 if (r + p) == 0 else 2 * (r * p) / (r + p)
        f1s.append((f1, i))
        i += 1
    f1s.sort(reverse=True, key=lambda tuple: tuple[0])
    return [t[1] for t in f1s[: commons.n_best_memory_sizes]]


def get_ams_results(
    midx,
    msize,
    dataset,
    domain,
    filling_features,
    testing_features,
):
    # Round the values
    qd = qudeq.QuDeq(filling_features, percentiles=commons.use_percentiles)
    trf_rounded = qd.quantize(filling_features, msize)
    tef_rounded = qd.quantize(testing_features, msize)
    behaviour = np.zeros(commons.n_behaviours, dtype=np.float64)

    # Create the memory using default parameters.
    params = commons.ExperimentSettings()
    eam = AssociativeMemory(domain, msize, params)

    # Registrate filling data.
    for features in trf_rounded:
        eam.register(features)

    metrics = recognize_by_memory(eam, tef_rounded, testing_features, msize, qd, dataset)
    return midx, eam.entropy, metrics


def statistics(dataset, es):
    list_results = []
    for fold in range(commons.n_folds):
        results = statistics_per_fold(dataset, es, fold)
        print(f'Results: {results}')
        list_results.append(results)
    means = []
    stdvs = []
    for mean, stdv in list_results:
        means.append(mean)
        stdvs.append(stdv)
    means = np.concatenate(means, axis=1)
    stdvs = np.concatenate(stdvs, axis=1)
    data = [means, stdvs]
    suffixes = ['-means', '-stdvs']
    for d, suffix in zip(data, suffixes):
        print(f'Shape{suffix[0]},{suffix[1]}: {d.shape}')
        filename = commons.fstats_name(dataset, es)
        filename += suffix
        filename = commons.csv_filename(filename, es)
        np.savetxt(filename, d, delimiter=',')


def distances(dataset, es):
    distance_means = []
    distance_stdvs = []
    for fold in range(commons.n_folds):
        mean, stdv = distances_per_fold(dataset, es, fold)
        distance_means.append(mean)
        distance_stdvs.append(stdv)
        plot_distances(mean, f'distances_{dataset}', es, fold)
    distance_means = np.concatenate(distance_means, axis=1)
    distance_stdvs = np.concatenate(distance_stdvs, axis=1)
    data = [distance_means, distance_stdvs]
    suffixes = ['-means', '-stdvs']
    for d, suffix in zip(data, suffixes):
        print(f'Shape{suffix[0]},{suffix[1]}: {d.shape}')
        filename = commons.distance_name(dataset, es)
        filename += suffix
        filename = commons.csv_filename(filename, es)
        np.savetxt(filename, d, delimiter=',')


def test_memory_sizes(dataset, es):
    domain = commons.domain(dataset)
    all_entropies = []
    recognition = []
    cosine = []
    cosine_score = []
    cosine_roundtrip = []
    cosine_roundtrip_score = []
    l2_distances = []

    print(f'Testing the memory of {dataset}')
    for fold in range(commons.n_folds):
        gc.collect()
        print(f'Fold: {fold}')
        suffix = commons.filling_suffix
        filling_features_filename = commons.features_name(dataset, es) + suffix
        filling_features_filename = commons.data_filename(
            filling_features_filename, es, fold
        )
        filling_labels_filename = commons.labels_name(dataset, es) + suffix
        filling_labels_filename = commons.data_filename(
            filling_labels_filename, es, fold
        )

        suffix = commons.testing_suffix
        testing_features_filename = commons.features_name(dataset, es) + suffix
        testing_features_filename = commons.data_filename(
            testing_features_filename, es, fold
        )
        testing_labels_filename = commons.labels_name(dataset, es) + suffix
        testing_labels_filename = commons.data_filename(
            testing_labels_filename, es, fold
        )

        filling_features = np.load(filling_features_filename)
        testing_features = np.load(testing_features_filename)
        measures = []
        entropies = []
        fold_recognition = []
        fold_cosine = []
        fold_cosine_score = []
        fold_cosine_roundtrip = []
        fold_cosine_roundtrip_score = []
        fold_l2 = []
        for midx, msize in enumerate(commons.memory_sizes):
            print(f'Memory size: {msize}')
            results = get_ams_results(
                midx,
                msize,
                dataset,
                domain,
                filling_features,
                testing_features,
            )
            measures.append(results)
        for memory_index, entropy, metrics in measures:
            entropies.append(entropy)
            fold_recognition.append(metrics['recognition_rate'] * 100.0)
            fold_cosine.append(metrics['mean_cosine'])
            fold_cosine_score.append(metrics['cosine_score'] * 100.0)
            fold_cosine_roundtrip.append(metrics['mean_cosine_roundtrip'])
            fold_cosine_roundtrip_score.append(
                metrics['cosine_roundtrip_score'] * 100.0
            )
            fold_l2.append(metrics['mean_l2'])

        all_entropies.append(entropies)
        recognition.append(fold_recognition)
        cosine.append(fold_cosine)
        cosine_score.append(fold_cosine_score)
        cosine_roundtrip.append(fold_cosine_roundtrip)
        cosine_roundtrip_score.append(fold_cosine_roundtrip_score)
        l2_distances.append(fold_l2)

    # Every row is training fold, and every column is a memory size.
    all_entropies = np.array(all_entropies)
    recognition = np.array(recognition)
    cosine = np.array(cosine)
    cosine_score = np.array(cosine_score)
    cosine_roundtrip = np.array(cosine_roundtrip)
    cosine_roundtrip_score = np.array(cosine_roundtrip_score)
    l2_distances = np.array(l2_distances)

    average_entropy = np.mean(all_entropies, axis=0)
    average_recognition = np.mean(recognition, axis=0)
    stdev_recognition = np.std(recognition, axis=0)
    average_cosine = np.nanmean(cosine, axis=0)
    stdev_cosine = np.nanstd(cosine, axis=0)
    average_cosine_score = np.mean(cosine_score, axis=0)
    stdev_cosine_score = np.std(cosine_score, axis=0)
    average_cosine_roundtrip = np.nanmean(cosine_roundtrip, axis=0)
    stdev_cosine_roundtrip = np.nanstd(cosine_roundtrip, axis=0)
    average_cosine_roundtrip_score = np.mean(cosine_roundtrip_score, axis=0)
    stdev_cosine_roundtrip_score = np.std(cosine_roundtrip_score, axis=0)
    average_l2 = np.nanmean(l2_distances, axis=0)
    stdev_l2 = np.nanstd(l2_distances, axis=0)
    best_memory_idx = optimum_indexes(average_recognition / 100.0, average_cosine_score / 100.0)
    best_memory_sizes = [commons.memory_sizes[i] for i in best_memory_idx]

    np.savetxt(
        commons.csv_filename('memory_recognition-' + dataset, es),
        recognition,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('memory_cosine-' + dataset, es), cosine, delimiter=','
    )
    np.savetxt(
        commons.csv_filename('memory_cosine_score-' + dataset, es),
        cosine_score,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('memory_cosine_roundtrip-' + dataset, es),
        cosine_roundtrip,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('memory_cosine_roundtrip_score-' + dataset, es),
        cosine_roundtrip_score,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('memory_l2-' + dataset, es),
        l2_distances,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('memory_entropy-' + dataset, es),
        all_entropies,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('memory_mean_recognition-' + dataset, es),
        average_recognition,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('memory_mean_cosine-' + dataset, es),
        average_cosine,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('memory_mean_cosine_score-' + dataset, es),
        average_cosine_score,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('memory_mean_cosine_roundtrip-' + dataset, es),
        average_cosine_roundtrip,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('memory_mean_cosine_roundtrip_score-' + dataset, es),
        average_cosine_roundtrip_score,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('memory_mean_l2-' + dataset, es),
        average_l2,
        delimiter=',',
    )
    plot_metric_pair_graph(
        average_recognition,
        average_cosine_score,
        average_entropy,
        stdev_recognition,
        stdev_cosine_score,
        dataset,
        es,
        _('Recognition'),
        _('Cosine score'),
        prefix='homo_msizes-',
    )
    plot_metric_pair_graph(
        average_recognition,
        average_cosine_roundtrip_score,
        average_entropy,
        stdev_recognition,
        stdev_cosine_roundtrip_score,
        dataset,
        es,
        _('Recognition'),
        _('Round-trip cosine score'),
        prefix='homo_msizes-roundtrip-',
    )
    print(f'Mean cosine similarity by memory size: {average_cosine}')
    print(f'Mean round-trip cosine similarity by memory size: {average_cosine_roundtrip}')
    print(f'Mean L2 distance by memory size: {average_l2}')
    print('Memory size evaluation completed!')
    return best_memory_sizes


def test_filling_percent(eam, msize, qd, trf, tef, raw_tef, percent, dataset):
    # Registrate filling data.
    for features in trf:
        eam.register(features)
    print(f'Filling of memories done at {percent}%')
    metrics = recognize_by_memory(eam, tef, raw_tef, msize, qd, dataset)
    return metrics, eam.entropy


def test_hetero_filling_percent(
    hetero_eam: HeteroAssociativeMemory,
    left_eam: AssociativeMemory,
    right_eam: AssociativeMemory,
    trfs,
    tefs,
    percent,
):
    # Register filling data.
    print('Filling hetero memory')
    counter = 0
    for left_feat, right_feat in zip(
        trfs[commons.left_dataset], trfs[commons.right_dataset]
    ):
        hetero_eam.register(left_feat, right_feat)
        counter += 1
        commons.print_counter(counter, 1000, 100)
    print(' end')
    print(f'Filling of memories done at {percent}%')
    print(f'Memory full at {100*hetero_eam.fullness}%')
    confrix = recognize_by_hetero_memory(hetero_eam, left_eam, right_eam, tefs)
    return confrix, hetero_eam.entropy


def hetero_recall_by_cue_percent(
    eam: HeteroAssociativeMemory,
    filling_features,
    testing_features,
    raw_testing_features,
    testing_labels,
    qudeqs,
    percent,
    es,
    fold,
):
    left_ds = commons.left_dataset
    right_ds = commons.right_dataset
    rows = commons.codomains()
    print('Filling hetero memory')
    counter = 0
    for left_feat, right_feat in zip(
        filling_features[left_ds], filling_features[right_ds]
    ):
        eam.register(left_feat, right_feat)
        counter += 1
        commons.print_counter(counter, 1000, 100)
    print(' end')
    print(f'Filling of memories done at {percent}%')
    print(f'Memory full at {100*eam.fullness}%')

    qd = qudeqs[right_ds]
    left_metrics, left_memories, left_metadata = recall_by_hetero_memory_embeddings(
        right_ds,
        eam.recall_from_left,
        testing_features[left_ds],
        testing_features[right_ds],
        raw_testing_features[right_ds],
        testing_labels[right_ds],
        rows[right_ds],
        commons.recall_with_cue,
        percent,
        qd,
    )
    name = (
        commons.memories_name(left_ds, es)
        + commons.recall_suffix(commons.recall_with_cue, None)
        + commons.int_suffix(percent, 'fll')
    )
    np.save(commons.data_filename(name, es, fold), left_memories)
    name = (
        commons.recall_labels_name(left_ds, es)
        + commons.recall_suffix(commons.recall_with_cue, None)
        + commons.int_suffix(percent, 'fll')
    )
    np.save(commons.data_filename(name, es, fold), left_metadata)

    qd = qudeqs[left_ds]
    right_metrics, right_memories, right_metadata = recall_by_hetero_memory_embeddings(
        left_ds,
        eam.recall_from_right,
        testing_features[right_ds],
        testing_features[left_ds],
        raw_testing_features[left_ds],
        testing_labels[left_ds],
        rows[left_ds],
        commons.recall_with_cue,
        percent,
        qd,
    )
    name = (
        commons.memories_name(right_ds, es)
        + commons.recall_suffix(commons.recall_with_cue, None)
        + commons.int_suffix(percent, 'fll')
    )
    np.save(commons.data_filename(name, es, fold), right_memories)
    name = (
        commons.recall_labels_name(right_ds, es)
        + commons.recall_suffix(commons.recall_with_cue, None)
        + commons.int_suffix(percent, 'fll')
    )
    np.save(commons.data_filename(name, es, fold), right_metadata)

    return [left_metrics, right_metrics], eam.entropy


def hetero_remember_percent(
    eam: HeteroAssociativeMemory,
    left_eam: AssociativeMemory,
    right_eam: AssociativeMemory,
    left_classifier,
    right_classifier,
    filling_features,
    testing_features,
    testing_labels,
    qudeqs,
    recall_method,
    proto_kind_suffix,
    percent,
    es,
    fold,
):
    # Register filling data.
    print('Filling hetero memory')
    counter = 0
    for left_feat, right_feat in zip(
        filling_features[commons.left_dataset], filling_features[commons.right_dataset]
    ):
        eam.register(left_feat, right_feat)
        counter += 1
        commons.print_counter(counter, 1000, 100)
    print(' end')
    print(f'Filling of memories done at {percent}%')
    print(f'Memory full at {100*eam.fullness}%')
    confrixes, behaviours = remember_by_hetero_memory(
        eam,
        left_eam,
        right_eam,
        left_classifier,
        right_classifier,
        testing_features,
        testing_labels,
        qudeqs,
        recall_method,
        proto_kind_suffix,
        percent,
        es,
        fold,
    )
    return confrixes, behaviours, eam.entropy


def test_filling_per_fold(mem_size, domain, dataset, es, fold):
    # Create the required associative memories using default parameters.
    params = commons.ExperimentSettings()
    eam = AssociativeMemory(domain, mem_size, params)

    suffix = commons.filling_suffix
    filling_features_filename = commons.features_name(dataset, es) + suffix
    filling_features_filename = commons.data_filename(
        filling_features_filename, es, fold
    )
    filling_labels_filename = commons.labels_name(dataset, es) + suffix
    filling_labels_filename = commons.data_filename(filling_labels_filename, es, fold)

    suffix = commons.testing_suffix
    testing_features_filename = commons.features_name(dataset, es) + suffix
    testing_features_filename = commons.data_filename(
        testing_features_filename, es, fold
    )
    testing_labels_filename = commons.labels_name(dataset, es) + suffix
    testing_labels_filename = commons.data_filename(testing_labels_filename, es, fold)

    raw_filling_features = np.load(filling_features_filename)
    raw_testing_features = np.load(testing_features_filename)

    qd = qudeq.QuDeq(raw_filling_features, percentiles=commons.use_percentiles)
    filling_features = qd.quantize(raw_filling_features, mem_size)
    testing_features = qd.quantize(raw_testing_features, mem_size)

    total = len(raw_filling_features)
    percents = np.array(commons.memory_fills)
    steps = np.round(total * percents / 100.0).astype(int)

    fold_entropies = []
    fold_recognition = []
    fold_cosine = []
    fold_cosine_score = []
    fold_cosine_roundtrip = []
    fold_cosine_roundtrip_score = []
    fold_l2 = []

    start = 0
    for percent, end in zip(percents, steps):
        features = filling_features[start:end]
        print(f'Filling from {start} to {end}.')
        metrics, entropy = test_filling_percent(
            eam,
            mem_size,
            qd,
            features,
            testing_features,
            raw_testing_features,
            percent,
            dataset,
        )
        fold_entropies.append(entropy)
        fold_recognition.append(metrics['recognition_rate'])
        fold_cosine.append(metrics['mean_cosine'])
        fold_cosine_score.append(metrics['cosine_score'])
        fold_cosine_roundtrip.append(metrics['mean_cosine_roundtrip'])
        fold_cosine_roundtrip_score.append(metrics['cosine_roundtrip_score'])
        fold_l2.append(metrics['mean_l2'])
        start = end
    fold_entropies = np.array(fold_entropies)
    fold_recognition = np.array(fold_recognition)
    fold_cosine = np.array(fold_cosine)
    fold_cosine_score = np.array(fold_cosine_score)
    fold_cosine_roundtrip = np.array(fold_cosine_roundtrip)
    fold_cosine_roundtrip_score = np.array(fold_cosine_roundtrip_score)
    fold_l2 = np.array(fold_l2)
    print(f'Filling test completed for fold {fold}')
    return (
        fold,
        fold_entropies,
        fold_recognition,
        fold_cosine,
        fold_cosine_score,
        fold_cosine_roundtrip,
        fold_cosine_roundtrip_score,
        fold_l2,
    )


def test_hetero_filling_per_fold(es, fold):
    # Create the required associative memories.
    domains = commons.domains()
    rows = commons.codomains()
    left_ds = commons.left_dataset
    right_ds = commons.right_dataset
    params = commons.ExperimentSettings()
    left_eam = AssociativeMemory(domains[left_ds], rows[left_ds], params)
    right_eam = AssociativeMemory(domains[right_ds], rows[right_ds], params)
    hetero_eam = HeteroAssociativeMemory(
        domains[left_ds], domains[right_ds], rows[left_ds], rows[right_ds], es, fold
    )
    filling_features = {}
    testing_features = {}
    for dataset in commons.datasets:
        suffix = commons.filling_suffix
        filling_features_filename = commons.features_name(dataset, es) + suffix
        filling_features_filename = commons.data_filename(
            filling_features_filename, es, fold
        )
        suffix = commons.testing_suffix
        testing_features_filename = commons.features_name(dataset, es) + suffix
        testing_features_filename = commons.data_filename(
            testing_features_filename, es, fold
        )
        f_features = np.load(filling_features_filename)
        t_features = np.load(testing_features_filename)
        qd = qudeq.QuDeq(f_features, percentiles=commons.use_percentiles)
        filling_features[dataset] = qd.quantize(f_features, rows[dataset])
        testing_features[dataset] = qd.quantize(t_features, rows[dataset])
    for f in filling_features[left_ds]:
        left_eam.register(f)
    for f in filling_features[right_ds]:
        right_eam.register(f)
    total = len(filling_features[left_ds])
    print(f'Filling hetero-associative memory with a total of {total} pairs.')
    percents = np.array(commons.memory_fills)
    steps = np.round(total * percents / 100.0).astype(int)

    fold_entropies = []
    fold_precision = []
    fold_recall = []
    fold_accuracy = []

    start = 0
    for percent, end in zip(percents, steps):
        features = {}
        features[left_ds] = filling_features[left_ds][start:end]
        features[right_ds] = filling_features[right_ds][start:end]
        print(f'Filling from {start} to {end}.')
        confrix, entropy = test_hetero_filling_percent(
            hetero_eam,
            left_eam,
            right_eam,
            features,
            testing_features,
            percent,
        )
        # An array with average entropy per step.
        fold_entropies.append(entropy)
        # Arrays with precision, and recall.
        positives = confrix[TP] + confrix[FP]
        fold_precision.append(1.0 if positives == 0 else confrix[TP] / positives)
        fold_recall.append(confrix[TP] / (confrix[TP] + confrix[FN]))
        fold_accuracy.append((confrix[TP] + confrix[TN]) / np.sum(confrix))
        start = end
    fold_entropies = np.array(fold_entropies)
    fold_precision = np.array(fold_precision)
    fold_recall = np.array(fold_recall)
    fold_accuracy = np.array(fold_accuracy)
    print(f'Filling test of hetero-associative memory completed for fold {fold}')
    return fold, fold_entropies, fold_precision, fold_recall, fold_accuracy


def hetero_recall_by_cue_per_fold(es, fold):
    domains = commons.domains()
    rows = commons.codomains()
    left_ds = commons.left_dataset
    right_ds = commons.right_dataset
    params = commons.ExperimentSettings()
    eam = HeteroAssociativeMemory(
        domains[left_ds],
        domains[right_ds],
        rows[left_ds],
        rows[right_ds],
        es,
        fold,
    )
    filling_features = {}
    testing_features = {}
    raw_testing_features = {}
    testing_labels = {}
    qudeqs = {}
    for dataset in commons.datasets:
        suffix = commons.filling_suffix
        filling_features_filename = commons.features_name(dataset, es) + suffix
        filling_features_filename = commons.data_filename(
            filling_features_filename, es, fold
        )
        suffix = commons.testing_suffix
        testing_features_filename = commons.features_name(dataset, es) + suffix
        testing_features_filename = commons.data_filename(
            testing_features_filename, es, fold
        )
        testing_labels_filename = commons.labels_name(dataset, es) + suffix
        testing_labels_filename = commons.data_filename(
            testing_labels_filename, es, fold
        )

        f_features = np.load(filling_features_filename)
        t_features = np.load(testing_features_filename)
        testing_labels[dataset] = np.load(testing_labels_filename)
        qd = qudeq.QuDeq(f_features, percentiles=commons.use_percentiles)
        filling_features[dataset] = qd.quantize(f_features, rows[dataset])
        testing_features[dataset] = qd.quantize(t_features, rows[dataset])
        raw_testing_features[dataset] = t_features
        qudeqs[dataset] = qd

    total = min(len(filling_features[left_ds]), len(filling_features[right_ds]))
    total_test = min(len(testing_features[left_ds]), len(testing_features[right_ds]))
    top = int(commons.exploration_ratio * total_test)
    for dataset in commons.datasets:
        testing_features[dataset] = testing_features[dataset][:top]
        raw_testing_features[dataset] = raw_testing_features[dataset][:top]
        testing_labels[dataset] = testing_labels[dataset][:top]
    print(f'Filling hetero-associative memory with a total of {total} pairs.')
    percents = np.array(commons.memory_fills)
    steps = np.round(total * percents / 100.0).astype(int)

    fold_entropies = []
    fold_recognition = []
    fold_cosine = []
    fold_cosine_score = []
    fold_cosine_roundtrip = []
    fold_cosine_roundtrip_score = []
    fold_l2 = []
    start = 0
    for percent, end in zip(percents, steps):
        features = {
            left_ds: filling_features[left_ds][start:end],
            right_ds: filling_features[right_ds][start:end],
        }
        print(f'Filling from {start} to {end}.')
        metrics_by_ds, entropy = hetero_recall_by_cue_percent(
            eam,
            features,
            testing_features,
            raw_testing_features,
            testing_labels,
            qudeqs,
            percent,
            es,
            fold,
        )
        fold_entropies.append(entropy)
        fold_recognition.append([m['recognition_rate'] for m in metrics_by_ds])
        fold_cosine.append([m['mean_cosine'] for m in metrics_by_ds])
        fold_cosine_score.append([m['cosine_score'] for m in metrics_by_ds])
        fold_cosine_roundtrip.append([m['mean_cosine_roundtrip'] for m in metrics_by_ds])
        fold_cosine_roundtrip_score.append(
            [m['cosine_roundtrip_score'] for m in metrics_by_ds]
        )
        fold_l2.append([m['mean_l2'] for m in metrics_by_ds])
        start = end

    return (
        fold,
        np.array(fold_entropies),
        np.transpose(np.array(fold_recognition, dtype=float)),
        np.transpose(np.array(fold_cosine, dtype=float)),
        np.transpose(np.array(fold_cosine_score, dtype=float)),
        np.transpose(np.array(fold_cosine_roundtrip, dtype=float)),
        np.transpose(np.array(fold_cosine_roundtrip_score, dtype=float)),
        np.transpose(np.array(fold_l2, dtype=float)),
    )


def hetero_remember_per_fold(recall_method, proto_kind_suffix, es, fold):
    # Create the required associative memories.
    domains = commons.domains()
    rows = commons.codomains()
    left_ds = commons.left_dataset
    right_ds = commons.right_dataset
    params = commons.ExperimentSettings()
    left_eam = AssociativeMemory(domains[left_ds], rows[left_ds], params)
    right_eam = AssociativeMemory(domains[right_ds], rows[right_ds], params)

    # Retrieve the classifiers.
    model_prefix = commons.model_name(left_ds, es)
    filename = commons.classifier_filename(model_prefix, es, fold)
    left_classifier = tf.keras.models.load_model(filename)
    model_prefix = commons.model_name(right_ds, es)
    filename = commons.classifier_filename(model_prefix, es, fold)
    right_classifier = tf.keras.models.load_model(filename)
    classifiers = {left_ds: left_classifier, right_ds: right_classifier}
    filling_features = {}
    filling_labels = {}
    testing_features = {}
    testing_labels = {}
    testing_data = {}
    filling_prototypes = {}
    qudeqs = {}
    for dataset in commons.datasets:
        suffix = commons.filling_suffix
        filling_features_filename = commons.features_name(dataset, es) + suffix
        filling_features_filename = commons.data_filename(
            filling_features_filename, es, fold
        )
        # Original tagging of filling data.
        filling_labels_filename = commons.labels_name(dataset, es) + suffix
        filling_labels_filename = commons.data_filename(
            filling_labels_filename, es, fold
        )
        filling_labels[dataset] = np.load(filling_labels_filename)
        f_features = np.load(filling_features_filename)
        # Tagging of filling data produced by the classifier
        # labels = np.argmax(classifiers[dataset].predict(f_features), axis=1)
        # filling_labels[dataset] = labels
        proto_filename = (
            commons.features_name(dataset, es)
            + commons.proto_suffix
            + proto_kind_suffix
            + commons.means_suffix
        )
        proto_filename = commons.data_filename(proto_filename, es, fold)
        prototypes = np.load(proto_filename)

        suffix = commons.testing_suffix
        testing_features_filename = commons.features_name(dataset, es) + suffix
        testing_features_filename = commons.data_filename(
            testing_features_filename, es, fold
        )
        # Original tagging of testing data.
        testing_labels_filename = commons.labels_name(dataset, es) + suffix
        testing_labels_filename = commons.data_filename(
            testing_labels_filename, es, fold
        )
        testing_labels[dataset] = np.load(testing_labels_filename)
        testing_data_filename = commons.data_name(dataset, es) + suffix
        testing_data_filename = commons.data_filename(testing_data_filename, es, fold)
        testing_data[dataset] = np.load(testing_data_filename)
        t_features = np.load(testing_features_filename)
        # Tagging of testing data produced by the classifier
        # labels = np.argmax(classifiers[dataset].predict(t_features), axis=1)
        # testing_labels[dataset] = labels
        validating_network_data(
            f_features,
            filling_labels[dataset],
            classifiers[dataset],
            dataset,
            'filling data',
        )
        validating_network_data(
            t_features,
            testing_labels[dataset],
            classifiers[dataset],
            dataset,
            'testing data',
        )
        qd = qudeq.QuDeq(f_features, percentiles=commons.use_percentiles)
        filling_features[dataset] = qd.quantize(f_features, rows[dataset])
        filling_prototypes[dataset] = qd.quantize(prototypes, rows[dataset])
        testing_features[dataset] = qd.quantize(t_features, rows[dataset])
        qudeqs[dataset] = qd

    eam = HeteroAssociativeMemory(
        domains[left_ds],
        domains[right_ds],
        rows[left_ds],
        rows[right_ds],
        es,
        fold,
        qudeqs[left_ds],
        qudeqs[right_ds],
        [filling_prototypes[left_ds], filling_prototypes[right_ds]],
    )

    for f in filling_features[left_ds]:
        left_eam.register(f)
    for f in filling_features[right_ds]:
        right_eam.register(f)
    match_labels(filling_features, filling_labels)
    describe(filling_features, filling_labels)
    match_labels(testing_features, testing_labels)
    total = len(filling_labels[left_ds])
    total_test = len(testing_labels[left_ds])
    top = int(commons.exploration_ratio * total_test)
    testing_labels[left_ds] = testing_labels[left_ds][:top]
    testing_features[left_ds] = testing_features[left_ds][:top]
    testing_labels[right_ds] = testing_labels[right_ds][:top]
    testing_features[right_ds] = testing_features[right_ds][:top]
    describe(testing_features, testing_labels)
    decode_test_features(testing_data, testing_features, testing_labels, fold, es)
    print(f'Filling hetero-associative memory with a total of {total} pairs.')
    percents = np.array(commons.memory_fills)
    steps = np.round(total * percents / 100.0).astype(int)

    fold_entropies = []
    fold_precision = []
    fold_recall = []
    fold_confrixes = []
    fold_behaviours = []
    start = 0
    for percent, end in zip(percents, steps):
        features = {}
        features[left_ds] = filling_features[left_ds][start:end]
        features[right_ds] = filling_features[right_ds][start:end]
        print(f'Filling from {start} to {end}.')
        confrixes, behaviours, entropy = hetero_remember_percent(
            eam,
            left_eam,
            right_eam,
            left_classifier,
            right_classifier,
            features,
            testing_features,
            testing_labels,
            qudeqs,
            recall_method,
            proto_kind_suffix,
            percent,
            es,
            fold,
        )
        fold_entropies.append(entropy)
        fold_behaviours.append(behaviours)
        fold_confrixes.append(confrixes)
        # Arrays with precision, and recall.
        no_response = behaviours[:, commons.no_response_idx]
        correct = behaviours[:, commons.correct_response_idx]
        incorrect = behaviours[:, commons.no_correct_response_idx]
        with np.errstate(divide='ignore'):
            fold_precision.append(
                np.where(
                    (correct + incorrect) == 0, 1.0, correct / (correct + incorrect)
                )
            )
        fold_recall.append(
            behaviours[:, commons.correct_response_idx]
            / (no_response + correct + incorrect)
        )
        start = end
    fold_entropies = np.array(fold_entropies)
    fold_precision = np.transpose(np.array(fold_precision))
    fold_recall = np.transpose(np.array(fold_recall))
    fold_behaviours = np.transpose(np.array(fold_behaviours, dtype=int), axes=(1, 0, 2))
    fold_confrixes = np.transpose(
        np.array(fold_confrixes, dtype=int), axes=(1, 0, 2, 3)
    )
    print(f'Filling test of hetero-associative memory completed for fold {fold}')
    return (
        fold,
        fold_entropies,
        fold_precision,
        fold_recall,
        fold_confrixes,
        fold_behaviours,
    )


def test_memory_fills(mem_sizes, dataset, es):
    memory_fills = commons.memory_fills
    testing_folds = commons.n_folds
    best_filling_percents = []
    for mem_size in mem_sizes:
        domain = commons.domain(dataset)
        total_entropies = np.zeros((testing_folds, len(memory_fills)))
        total_recognitions = np.zeros((testing_folds, len(memory_fills)))
        total_cosines = np.full((testing_folds, len(memory_fills)), np.nan)
        total_cosine_scores = np.zeros((testing_folds, len(memory_fills)))
        total_cosine_roundtrips = np.full((testing_folds, len(memory_fills)), np.nan)
        total_cosine_roundtrip_scores = np.zeros((testing_folds, len(memory_fills)))
        total_l2 = np.full((testing_folds, len(memory_fills)), np.nan)
        list_results = []

        for fold in range(testing_folds):
            results = test_filling_per_fold(mem_size, domain, dataset, es, fold)
            list_results.append(results)
        for (
            fold,
            entropies,
            recognitions,
            cosines,
            cosine_scores,
            cosine_roundtrips,
            cosine_roundtrip_scores,
            l2_distances,
        ) in list_results:
            total_recognitions[fold] = recognitions
            total_cosines[fold] = cosines
            total_cosine_scores[fold] = cosine_scores
            total_cosine_roundtrips[fold] = cosine_roundtrips
            total_cosine_roundtrip_scores[fold] = cosine_roundtrip_scores
            total_l2[fold] = l2_distances
            total_entropies[fold] = entropies

        main_avrge_entropies = np.mean(total_entropies, axis=0)
        main_stdev_entropies = np.std(total_entropies, axis=0)
        main_avrge_recognitions = np.mean(total_recognitions, axis=0)
        main_stdev_recognitions = np.std(total_recognitions, axis=0)
        main_avrge_cosines = np.nanmean(total_cosines, axis=0)
        main_stdev_cosines = np.nanstd(total_cosines, axis=0)
        main_avrge_cosine_scores = np.mean(total_cosine_scores, axis=0)
        main_stdev_cosine_scores = np.std(total_cosine_scores, axis=0)
        main_avrge_cosine_roundtrips = np.nanmean(total_cosine_roundtrips, axis=0)
        main_stdev_cosine_roundtrips = np.nanstd(total_cosine_roundtrips, axis=0)
        main_avrge_cosine_roundtrip_scores = np.mean(
            total_cosine_roundtrip_scores, axis=0
        )
        main_stdev_cosine_roundtrip_scores = np.std(
            total_cosine_roundtrip_scores, axis=0
        )
        main_avrge_l2 = np.nanmean(total_l2, axis=0)
        main_stdev_l2 = np.nanstd(total_l2, axis=0)

        np.savetxt(
            commons.csv_filename(
                'main_average_recognition-'
                + dataset
                + commons.numeric_suffix('sze', mem_size),
                es,
            ),
            main_avrge_recognitions,
            delimiter=',',
        )
        np.savetxt(
            commons.csv_filename(
                'main_average_cosine-'
                + dataset
                + commons.numeric_suffix('sze', mem_size),
                es,
            ),
            main_avrge_cosines,
            delimiter=',',
        )
        np.savetxt(
            commons.csv_filename(
                'main_average_cosine_score-'
                + dataset
                + commons.numeric_suffix('sze', mem_size),
                es,
            ),
            main_avrge_cosine_scores,
            delimiter=',',
        )
        np.savetxt(
            commons.csv_filename(
                'main_average_cosine_roundtrip-'
                + dataset
                + commons.numeric_suffix('sze', mem_size),
                es,
            ),
            main_avrge_cosine_roundtrips,
            delimiter=',',
        )
        np.savetxt(
            commons.csv_filename(
                'main_average_cosine_roundtrip_score-'
                + dataset
                + commons.numeric_suffix('sze', mem_size),
                es,
            ),
            main_avrge_cosine_roundtrip_scores,
            delimiter=',',
        )
        np.savetxt(
            commons.csv_filename(
                'main_average_l2-'
                + dataset
                + commons.numeric_suffix('sze', mem_size),
                es,
            ),
            main_avrge_l2,
            delimiter=',',
        )
        np.savetxt(
            commons.csv_filename(
                'main_average_entropy-'
                + dataset
                + commons.numeric_suffix('sze', mem_size),
                es,
            ),
            main_avrge_entropies,
            delimiter=',',
        )
        np.savetxt(
            commons.csv_filename(
                'main_stdev_recognition-'
                + dataset
                + commons.numeric_suffix('sze', mem_size),
                es,
            ),
            main_stdev_recognitions,
            delimiter=',',
        )
        np.savetxt(
            commons.csv_filename(
                'main_stdev_cosine-'
                + dataset
                + commons.numeric_suffix('sze', mem_size),
                es,
            ),
            main_stdev_cosines,
            delimiter=',',
        )
        np.savetxt(
            commons.csv_filename(
                'main_stdev_cosine_score-'
                + dataset
                + commons.numeric_suffix('sze', mem_size),
                es,
            ),
            main_stdev_cosine_scores,
            delimiter=',',
        )
        np.savetxt(
            commons.csv_filename(
                'main_stdev_cosine_roundtrip-'
                + dataset
                + commons.numeric_suffix('sze', mem_size),
                es,
            ),
            main_stdev_cosine_roundtrips,
            delimiter=',',
        )
        np.savetxt(
            commons.csv_filename(
                'main_stdev_cosine_roundtrip_score-'
                + dataset
                + commons.numeric_suffix('sze', mem_size),
                es,
            ),
            main_stdev_cosine_roundtrip_scores,
            delimiter=',',
        )
        np.savetxt(
            commons.csv_filename(
                'main_stdev_l2-'
                + dataset
                + commons.numeric_suffix('sze', mem_size),
                es,
            ),
            main_stdev_l2,
            delimiter=',',
        )
        np.savetxt(
            commons.csv_filename(
                'main_stdev_entropy-'
                + dataset
                + commons.numeric_suffix('sze', mem_size),
                es,
            ),
            main_stdev_entropies,
            delimiter=',',
        )

        plot_metric_pair_graph(
            main_avrge_recognitions * 100,
            main_avrge_cosine_scores * 100,
            main_avrge_entropies,
            main_stdev_recognitions * 100,
            main_stdev_cosine_scores * 100,
            dataset,
            es,
            _('Recognition'),
            _('Cosine score'),
            prefix='homo_fills' + commons.numeric_suffix('sze', mem_size) + '-',
            xlabels=commons.memory_fills,
            xtitle=_('Percentage of memory corpus'),
        )
        plot_metric_pair_graph(
            main_avrge_recognitions * 100,
            main_avrge_cosine_roundtrip_scores * 100,
            main_avrge_entropies,
            main_stdev_recognitions * 100,
            main_stdev_cosine_roundtrip_scores * 100,
            dataset,
            es,
            _('Recognition'),
            _('Round-trip cosine score'),
            prefix='homo_fills-roundtrip' + commons.numeric_suffix('sze', mem_size) + '-',
            xlabels=commons.memory_fills,
            xtitle=_('Percentage of memory corpus'),
        )

        bf_idx = optimum_indexes(main_avrge_recognitions, main_avrge_cosine_scores)
        best_filling_percents.append(commons.memory_fills[bf_idx[0]])
        print(f'Mean cosine similarity by fill for size {mem_size}: {main_avrge_cosines}')
        print(
            'Mean round-trip cosine similarity by fill for size '
            + f'{mem_size}: {main_avrge_cosine_roundtrips}'
        )
        print(f'Mean L2 distance by fill for size {mem_size}: {main_avrge_l2}')
        print(f'Testing fillings for memory size {mem_size} done.')
    return best_filling_percents


def test_hetero_fills(es):
    memory_fills = commons.memory_fills
    testing_folds = commons.n_folds
    # All entropies, precision, and recall, per size, fold, and fill.
    total_entropies = np.zeros((testing_folds, len(memory_fills)))
    total_precisions = np.zeros((testing_folds, len(memory_fills)))
    total_recalls = np.zeros((testing_folds, len(memory_fills)))
    total_accuracies = np.zeros((testing_folds, len(memory_fills)))
    list_results = []

    for fold in range(testing_folds):
        results = test_hetero_filling_per_fold(es, fold)
        list_results.append(results)
    for fold, entropies, precisions, recalls, accuracies in list_results:
        total_precisions[fold] = precisions
        total_recalls[fold] = recalls
        total_entropies[fold] = entropies
        total_accuracies[fold] = accuracies
    main_avrge_entropies = np.mean(total_entropies, axis=0)
    main_stdev_entropies = np.std(total_entropies, axis=0)
    main_avrge_precisions = np.mean(total_precisions, axis=0)
    main_stdev_precisions = np.std(total_precisions, axis=0)
    main_avrge_recalls = np.mean(total_recalls, axis=0)
    main_stdev_recalls = np.std(total_recalls, axis=0)
    main_avrge_accuracies = np.mean(total_accuracies, axis=0)
    main_stdev_accuracies = np.std(total_accuracies, axis=0)

    np.savetxt(
        commons.csv_filename('hetero_average_precision', es),
        main_avrge_precisions,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('hetero_average_recall', es),
        main_avrge_recalls,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('hetero_average_accuracy', es),
        main_avrge_accuracies,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('hetero_average_entropy', es),
        main_avrge_entropies,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('hetero_stdev_precision', es),
        main_stdev_precisions,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('hetero_stdev_recall', es),
        main_stdev_recalls,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('hetero_stdev_accuracy', es),
        main_stdev_accuracies,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('hetero_stdev_entropy', es),
        main_stdev_entropies,
        delimiter=',',
    )

    prefix = 'hetero_recognize-'
    plot_prerec_graph(
        100 * main_avrge_precisions,
        100 * main_avrge_recalls,
        main_avrge_entropies,
        100 * main_stdev_precisions,
        100 * main_stdev_recalls,
        'hetero',
        es,
        acc_mean=100 * main_avrge_accuracies,
        acc_std=100 * main_stdev_accuracies,
        prefix=prefix,
        xlabels=commons.memory_fills,
        xtitle=_('Percentage of memory corpus'),
    )
    print('Testing fillings for hetero-associative done.')


def validating_network_data(features, labels, classifier, dataset, description):
    predictions = np.argmax(classifier.predict(features), axis=1)
    total = labels.size
    agreements = np.count_nonzero(predictions == labels)
    print(
        f'Validating coherence between data and network for {description} of {dataset}'
    )
    print(f'Agreement percentage: {100*agreements/total}')


def save_history(history, prefix, es):
    """Saves the stats of neural networks.

    Neural networks stats may come either as a History object, that includes
    a History.history dictionary with stats, or directly as a dictionary.
    """
    stats = {}
    stats['history'] = []
    for h in history:
        while not isinstance(h, (dict, list)):
            h = h.history
        stats['history'].append(h)
    with open(commons.json_filename(prefix, es), 'w') as outfile:
        json.dump(stats, outfile)


def save_legacy_conf_matrix(matrix, dataset, prefix, es, vmax=None):
    plot_legacy_confusion_matrix(
        matrix, range(commons.n_labels), dataset, es, prefix, vmax=vmax
    )
    fname = prefix + commons.matrix_suffix + '-' + dataset
    filename = commons.data_filename(fname)
    np.save(filename, matrix)


def save_conf_matrix(matrix, dataset, prefix, es, vmax=None):
    """Backward-compatible alias for legacy confusion-matrix persistence."""
    save_legacy_conf_matrix(matrix, dataset, prefix, es, vmax=vmax)


def save_learned_params(mem_sizes, fill_percents, dataset, es):
    name = commons.learn_params_name(dataset, es)
    filename = commons.data_filename(name, es)
    np.save(filename, np.array([mem_sizes, fill_percents], dtype=int))


def remember(recall_method, proto_kind_suffix, es):
    if recall_method == commons.recall_with_cue:
        return remember_with_cue(es)
    return remember_with_legacy_metrics(recall_method, proto_kind_suffix, es)


def remember_with_legacy_metrics(recall_method, proto_kind_suffix, es):
    memory_fills = commons.memory_fills
    testing_folds = commons.n_folds
    total_entropies = np.zeros((testing_folds, len(memory_fills)))
    # We are capturing left and right measures.
    n_sets = 2
    total_precisions = np.zeros((testing_folds, n_sets, len(memory_fills)))
    total_recalls = np.zeros((testing_folds, n_sets, len(memory_fills)))
    total_confrixes = []
    total_behaviours = []

    for fold in range(testing_folds):
        fold, entropies, precisions, recalls, confrixes, behaviours = (
            hetero_remember_per_fold(recall_method, proto_kind_suffix, es, fold)
        )
        total_precisions[fold] = precisions
        total_recalls[fold] = recalls
        total_entropies[fold] = entropies
        total_confrixes.append(confrixes)
        total_behaviours.append(behaviours)
    total_confrixes = np.array(total_confrixes, dtype=int)
    total_behaviours = np.array(total_behaviours, dtype=int)

    main_avrge_entropies = np.mean(total_entropies, axis=0)
    main_stdev_entropies = np.std(total_entropies, axis=0)
    main_avrge_precisions = np.mean(total_precisions, axis=0)
    main_stdev_precisions = np.std(total_precisions, axis=0)
    main_avrge_recalls = np.mean(total_recalls, axis=0)
    main_stdev_recalls = np.std(total_recalls, axis=0)
    main_avrge_confrixes = np.mean(total_confrixes, axis=0)
    main_stdev_confrixes = np.std(total_confrixes, axis=0)
    main_avrge_behaviours = np.mean(total_behaviours, axis=0)
    main_stdev_behaviours = np.std(total_behaviours, axis=0)
    suffix = commons.recall_suffix(recall_method, proto_kind_suffix)
    np.savetxt(
        commons.csv_filename('remember_average_precision' + suffix, es),
        main_avrge_precisions,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('remember_average_recall' + suffix, es),
        main_avrge_recalls,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('remember_average_entropy' + suffix, es),
        main_avrge_entropies,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('remember_stdev_precision' + suffix, es),
        main_stdev_precisions,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('remember_stdev_recall' + suffix, es),
        main_stdev_recalls,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('remember_stdev_entropy' + suffix, es),
        main_stdev_entropies,
        delimiter=',',
    )
    np.save(
        commons.data_filename('remember_mean_behaviours' + suffix, es),
        main_avrge_behaviours,
    )
    np.save(
        commons.data_filename('remember_stdv_behaviours' + suffix, es),
        main_stdev_behaviours,
    )
    np.save(
        commons.data_filename('remember_mean_confrixes' + suffix, es),
        main_avrge_confrixes,
    )
    np.save(
        commons.data_filename('remember_stdv_confrixes' + suffix, es),
        main_stdev_confrixes,
    )

    for i in range(len(commons.datasets)):
        dataset = commons.datasets[i]
        plot_prerec_graph(
            100 * main_avrge_precisions[i],
            100 * main_avrge_recalls[i],
            main_avrge_entropies,
            100 * main_stdev_precisions[i],
            100 * main_stdev_recalls[i],
            dataset,
            es,
            prefix='hetero_remember' + suffix + '-',
            xlabels=commons.memory_fills,
            xtitle=_('Percentage of memory corpus'),
        )
        mean_no_response = main_avrge_behaviours[i, :, commons.no_response_idx]
        mean_no_correct_response = main_avrge_behaviours[
            i, :, commons.no_correct_response_idx
        ]
        mean_correct_response = main_avrge_behaviours[
            i, :, commons.correct_response_idx
        ]
        plot_behs_graph(
            mean_no_response,
            mean_no_correct_response,
            mean_correct_response,
            dataset,
            es,
            xtags=commons.memory_fills,
            prefix='hetero_remember' + suffix + '-',
        )
        for j, f in enumerate(commons.memory_fills):
            save_legacy_conf_matrix(
                main_avrge_confrixes[i, j],
                dataset,
                f'hetero_remember{suffix}-fll_{str(f).zfill(3)}',
                es,
            )
    print('Remembering done!')


def remember_with_cue(es):
    memory_fills = commons.memory_fills
    testing_folds = commons.n_folds
    n_sets = len(commons.datasets)
    total_entropies = np.zeros((testing_folds, len(memory_fills)))
    total_recognitions = np.zeros((testing_folds, n_sets, len(memory_fills)))
    total_cosines = np.full((testing_folds, n_sets, len(memory_fills)), np.nan)
    total_cosine_scores = np.zeros((testing_folds, n_sets, len(memory_fills)))
    total_cosine_roundtrips = np.full((testing_folds, n_sets, len(memory_fills)), np.nan)
    total_cosine_roundtrip_scores = np.zeros((testing_folds, n_sets, len(memory_fills)))
    total_l2 = np.full((testing_folds, n_sets, len(memory_fills)), np.nan)

    for fold in range(testing_folds):
        (
            fold,
            entropies,
            recognitions,
            cosines,
            cosine_scores,
            cosine_roundtrips,
            cosine_roundtrip_scores,
            l2_distances,
        ) = hetero_recall_by_cue_per_fold(es, fold)
        total_entropies[fold] = entropies
        total_recognitions[fold] = recognitions
        total_cosines[fold] = cosines
        total_cosine_scores[fold] = cosine_scores
        total_cosine_roundtrips[fold] = cosine_roundtrips
        total_cosine_roundtrip_scores[fold] = cosine_roundtrip_scores
        total_l2[fold] = l2_distances

    main_avrge_entropies = np.mean(total_entropies, axis=0)
    main_stdev_entropies = np.std(total_entropies, axis=0)
    main_avrge_recognitions = np.mean(total_recognitions, axis=0)
    main_stdev_recognitions = np.std(total_recognitions, axis=0)
    main_avrge_cosines = np.nanmean(total_cosines, axis=0)
    main_stdev_cosines = np.nanstd(total_cosines, axis=0)
    main_avrge_cosine_scores = np.mean(total_cosine_scores, axis=0)
    main_stdev_cosine_scores = np.std(total_cosine_scores, axis=0)
    main_avrge_cosine_roundtrips = np.nanmean(total_cosine_roundtrips, axis=0)
    main_stdev_cosine_roundtrips = np.nanstd(total_cosine_roundtrips, axis=0)
    main_avrge_cosine_roundtrip_scores = np.mean(total_cosine_roundtrip_scores, axis=0)
    main_stdev_cosine_roundtrip_scores = np.std(total_cosine_roundtrip_scores, axis=0)
    main_avrge_l2 = np.nanmean(total_l2, axis=0)
    main_stdev_l2 = np.nanstd(total_l2, axis=0)
    suffix = commons.recall_suffix(commons.recall_with_cue, None)

    np.savetxt(
        commons.csv_filename('remember_average_recognition' + suffix, es),
        main_avrge_recognitions,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('remember_average_cosine' + suffix, es),
        main_avrge_cosines,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('remember_average_cosine_score' + suffix, es),
        main_avrge_cosine_scores,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('remember_average_cosine_roundtrip' + suffix, es),
        main_avrge_cosine_roundtrips,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('remember_average_cosine_roundtrip_score' + suffix, es),
        main_avrge_cosine_roundtrip_scores,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('remember_average_l2' + suffix, es),
        main_avrge_l2,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('remember_average_entropy' + suffix, es),
        main_avrge_entropies,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('remember_stdev_recognition' + suffix, es),
        main_stdev_recognitions,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('remember_stdev_cosine' + suffix, es),
        main_stdev_cosines,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('remember_stdev_cosine_score' + suffix, es),
        main_stdev_cosine_scores,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('remember_stdev_cosine_roundtrip' + suffix, es),
        main_stdev_cosine_roundtrips,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('remember_stdev_cosine_roundtrip_score' + suffix, es),
        main_stdev_cosine_roundtrip_scores,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('remember_stdev_l2' + suffix, es),
        main_stdev_l2,
        delimiter=',',
    )
    np.savetxt(
        commons.csv_filename('remember_stdev_entropy' + suffix, es),
        main_stdev_entropies,
        delimiter=',',
    )

    for i, dataset in enumerate(commons.datasets):
        plot_metric_pair_graph(
            100 * main_avrge_recognitions[i],
            100 * main_avrge_cosine_scores[i],
            main_avrge_entropies,
            100 * main_stdev_recognitions[i],
            100 * main_stdev_cosine_scores[i],
            dataset,
            es,
            _('Recognition'),
            _('Cosine score'),
            prefix='hetero_remember' + suffix + '-',
            xlabels=commons.memory_fills,
            xtitle=_('Percentage of memory corpus'),
        )
        plot_metric_pair_graph(
            100 * main_avrge_recognitions[i],
            100 * main_avrge_cosine_roundtrip_scores[i],
            main_avrge_entropies,
            100 * main_stdev_recognitions[i],
            100 * main_stdev_cosine_roundtrip_scores[i],
            dataset,
            es,
            _('Recognition'),
            _('Round-trip cosine score'),
            prefix='hetero_remember-roundtrip' + suffix + '-',
            xlabels=commons.memory_fills,
            xtitle=_('Percentage of memory corpus'),
        )
        print(f'Mean cosine similarity for {dataset}: {main_avrge_cosines[i]}')
        print(
            f'Mean round-trip cosine similarity for {dataset}: '
            + f'{main_avrge_cosine_roundtrips[i]}'
        )
        print(f'Mean L2 distance for {dataset}: {main_avrge_l2[i]}')
    print('Remembering with cue done!')


def store_memory(image, directory, name, idx, correct, prediction, es, fold):
    filename = commons.memory_image_filename(
        directory, name, idx, correct, prediction, es, fold
    )
    full_directory = commons.dirname(filename)
    commons.create_directory(full_directory)
    store_image(filename, image)


def store_dream(image, initial_label, depth, label, path):
    filename = commons.dream_image_filename(path, initial_label, depth, label)
    full_directory = commons.dirname(filename)
    commons.create_directory(full_directory)
    store_image(filename, image)


def store_image(filename, image):
    pixels = image.round().astype(np.uint8)
    png.from_array(pixels, 'L;8').save(filename)


def store_test(original, produced, test_dir, idx, label, dataset, es, fold):
    directory = os.path.join(test_dir, dataset)
    orig_test_filename = commons.testing_image_filename(directory, idx, label, es, fold)
    prod_test_filename = commons.prod_testing_image_filename(
        directory, idx, label, es, fold
    )
    store_image(orig_test_filename, original)
    store_image(prod_test_filename, produced)


def decode_memories(
    memories,
    indexes,
    corrects,
    predictions,
    dataset,
    percent,
    recall_method,
    proto_kind_suffix,
    es,
    fold,
):
    if len(indexes) == 0:
        return
    model_prefix = commons.model_name(commons.alt(dataset), es)
    model_filename = commons.decoder_filename(model_prefix, es, fold)
    # Loads the decoder.
    model = tf.keras.models.load_model(model_filename)
    images = model.predict(memories)
    name = (
        commons.memories_name(dataset, es)
        + commons.recall_suffix(recall_method, proto_kind_suffix)
        + commons.int_suffix(percent, 'fll')
    )
    memories_path = commons.memories_path
    for idx, image, correct, prediction in zip(indexes, images, corrects, predictions):
        store_memory(image, memories_path, name, idx, correct, prediction, es, fold)


def decode_test_features(data, features, labels, fold, es):
    """Creates images directly from test features.

    Uses the decoder part of the neural networks to (re)create
    images from features generated by the encoder.
    """
    for dataset in commons.datasets:
        model_prefix = commons.model_name(dataset, es)
        # Load test features and labels
        model_filename = commons.decoder_filename(model_prefix, es, fold)
        model = tf.keras.models.load_model(model_filename)
        model.summary()
        # Generate images.
        prod_test_images = model.predict(features[dataset])
        n = len(labels[dataset])
        # Save images.
        for i, original, produced, label in zip(
            range(n), data[dataset], prod_test_images, labels[dataset]
        ):
            # if testing_label != label:
            #    raise ValueError(
            #        f'La etiqueta en los datos ({testing_label}) es diferente a ({label})')
            store_test(
                original, produced, commons.testing_path, i, label, dataset, es, fold
            )


def features_parameters(dataset, es):
    cols = commons.datasets_to_domains[dataset]
    rows = commons.datasets_to_codomains[dataset]
    means = np.zeros((commons.n_folds, 4, commons.n_labels, cols), dtype=float)
    stdvs = np.zeros((commons.n_folds, 4, commons.n_labels, cols), dtype=float)
    hists = np.zeros((commons.n_folds, 4, commons.n_labels), dtype=int)
    model_prefix = commons.model_name(dataset, es)
    for fold in range(commons.n_folds):
        proto_kinds = []
        features_filename = commons.features_name(dataset, es) + commons.filling_suffix
        features_filename = commons.data_filename(features_filename, es, fold)
        labels_filename = commons.labels_name(dataset, es) + commons.filling_suffix
        labels_filename = commons.data_filename(labels_filename, es, fold)
        filling_features = np.load(features_filename)
        filling_labels = np.load(labels_filename)
        features_filename = commons.features_name(dataset, es) + commons.testing_suffix
        features_filename = commons.data_filename(features_filename, es, fold)
        labels_filename = commons.labels_name(dataset, es) + commons.testing_suffix
        labels_filename = commons.data_filename(labels_filename, es, fold)
        testing_features = np.load(features_filename)
        const_means, const_stdvs = construct_prototypes(
            filling_features, filling_labels, cols
        )
        means[fold, 0] = const_means
        stdvs[fold, 0] = const_stdvs
        hists[fold, 0] = np.zeros(commons.n_labels, dtype=int)
        proto_kinds.append(commons.proto_kind_constructed)
        filename = commons.classifier_filename(model_prefix, es, fold)
        classifier = tf.keras.models.load_model(filename)
        extrt_means, extrt_stdvs, extrt_hist = extract_prototypes(
            filling_features, classifier, cols, rows
        )
        means[fold, 1] = extrt_means
        stdvs[fold, 1] = extrt_stdvs
        hists[fold, 1] = extrt_hist
        proto_kinds.append(commons.proto_kind_extracted)
        recll_means, recll_stdvs, recll_hist = recall_prototypes(
            filling_features, filling_features, classifier, cols, rows
        )
        means[fold, 2] = recll_means
        stdvs[fold, 2] = recll_stdvs
        hists[fold, 2] = recll_hist
        proto_kinds.append(commons.proto_kind_fill_recalled)
        rectd_means, rectd_stdvs, rectd_hist = recall_prototypes(
            filling_features, testing_features, classifier, cols, rows
        )
        means[fold, 3] = rectd_means
        stdvs[fold, 3] = rectd_stdvs
        hists[fold, 3] = rectd_hist
        proto_kinds.append(commons.proto_kind_test_recalled)
    return means, stdvs, hists, proto_kinds


def construct_prototypes(features, labels, cols):
    means = np.zeros((commons.n_labels, cols), dtype=float)
    stdvs = np.zeros((commons.n_labels, cols), dtype=float)
    for label in commons.all_labels:
        feats = np.array(
            [f for f, l in zip(features, labels) if l == label]  # noqa: E741
        )
        mean = np.mean(feats, axis=0)
        stdv = np.std(feats, axis=0)
        means[label] = mean
        stdvs[label] = stdv
    return means, stdvs


def extract_prototypes(features, classifier, cols, rows):
    qd = qudeq.QuDeq(features, percentiles=commons.use_percentiles)
    features = qd.quantize(features, rows)
    ams = AssociativeMemory(cols, rows)
    for f in features:
        ams.register(f)
    samples = []
    n = int(math.pow(2, round(math.log(commons.n_labels * 100, 2) + 0.5)))
    print(f'Samples: {n}')
    for _ in range(n):
        recall, _, _ = ams.recall()
        samples.append(recall)
    samples = qd.dequantize(np.array(samples), rows)
    predictions = np.argmax(classifier.predict(samples), axis=1)
    clusters = {i: [] for i in commons.all_labels}
    for p, s in zip(predictions, samples):
        clusters[p].append(s)
    frequencies = [len(clusters[i]) for i in commons.all_labels]
    print(f'Samples per label: {frequencies}')
    means = np.zeros((commons.n_labels, cols), dtype=float)
    stdvs = np.zeros((commons.n_labels, cols), dtype=float)
    for label in commons.all_labels:
        mean = np.mean(clusters[label], axis=0)
        stdv = np.std(clusters[label], axis=0)
        means[label] = mean
        stdvs[label] = stdv
    return means, stdvs, frequencies


def recall_prototypes(filling_features, testing_features, classifier, cols, rows):
    qd = qudeq.QuDeq(filling_features, percentiles=commons.use_percentiles)
    filling_features = qd.quantize(filling_features, rows)
    testing_features = qd.quantize(testing_features, rows)
    ams = AssociativeMemory(cols, rows)
    for f in filling_features:
        ams.register(f)
    samples = []
    for t in testing_features:
        recall, _, _ = ams.recall(t)
        samples.append(recall)
    samples = qd.dequantize(np.array(samples), rows)
    predictions = np.argmax(classifier.predict(samples), axis=1)
    clusters = {i: [] for i in commons.all_labels}
    for p, s in zip(predictions, samples):
        clusters[p].append(s)
    frequencies = [len(clusters[i]) for i in commons.all_labels]
    print(f'Samples per label: {frequencies}')
    means = np.zeros((commons.n_labels, cols), dtype=float)
    stdvs = np.zeros((commons.n_labels, cols), dtype=float)
    for label in commons.all_labels:
        mean = np.mean(clusters[label], axis=0)
        stdv = np.std(clusters[label], axis=0)
        means[label] = mean
        stdvs[label] = stdv
    return means, stdvs, frequencies


def save_prototypes(means, stdvs, proto_kinds, dataset, es):
    proto_suffix = commons.proto_suffix
    for fold in range(commons.n_folds):
        model_prefix = commons.model_name(dataset, es)
        model_filename = commons.decoder_filename(model_prefix, es, fold)
        # Loads the decoder.
        model = tf.keras.models.load_model(model_filename)
        model.summary()
        for i, kind in zip(range(len(proto_kinds)), proto_kinds):
            suffix = commons.proto_kind_suffix(kind)
            proto_filename = commons.features_name(dataset, es) + proto_suffix + suffix
            proto_means_filename = commons.data_filename(
                proto_filename + commons.means_suffix, es, fold
            )
            proto_stdvs_filename = commons.data_filename(
                proto_filename + commons.stdvs_suffix, es, fold
            )
            np.save(proto_means_filename, means[fold, i])
            np.save(proto_stdvs_filename, stdvs[fold, i])
            proto_images = model.predict(means[fold, i])
            prototypes_path = (
                commons.prototypes_path + suffix + commons.dataset_suffix(dataset)
            )
            name = commons.prototypes_name(dataset, es)
            for memory, label in zip(proto_images, commons.all_labels):
                store_memory(memory, prototypes_path, name, 0, label, label, es, fold)


def save_features_graphs(means, stdvs, hists, proto_kinds, dataset, es):
    means = np.mean(means, axis=0)
    stdvs = np.mean(stdvs, axis=0)
    hists = np.mean(hists, axis=0)
    labels = [commons.proto_labels[kind] for kind in proto_kinds]
    plot_features_graph(
        commons.datasets_to_domains[dataset], means, stdvs, labels, dataset, es
    )
    for hist, label, kind in zip(hists, labels, proto_kinds):
        fname = commons.prototypes_prefix + dataset + commons.proto_kind_suffix(kind)
        plot_histo_bar(
            hist,
            dataset,
            es,
            xtags=commons.all_labels,
            xlabel='Label',
            label=label,
            name=fname,
        )


def sample_features_for_sequencing(features, labels):
    feats = {}
    labls = {}
    for dataset in commons.datasets:
        unique, counts = np.unique(labels[dataset], return_counts=True)
        chosen_feats = []
        chosen_labls = []
        for u, c in zip(unique, counts):
            target = int(random.randrange(c))
            i = 0
            print(f'Choosing feautures number {target} with label {u} from {dataset}')
            for fs, ll in zip(features[dataset], labels[dataset]):
                if (ll == u) and (i == target):
                    chosen_feats.append(fs)
                    chosen_labls.append(ll)
                    break
                elif ll == u:
                    i += 1
        feats[dataset] = np.array(chosen_feats, dtype=float)
        labls[dataset] = np.array(chosen_labls, dtype=int)
    return feats, labls


def produce_testing_sequences(
    hetero: HeteroAssociativeMemory, features, labels, qds, recall_method
):
    """
    Produces sequences of memories for an array of features.

    :param hetero:          A filled hetero-associative memory.
    :param features:        A dictionary with lists of features to be used as starting points.
    :param labels:          A dictionary with lists of Labels for features.
    :param qds:             A dictionary of QuDeq (quantizer-dequantizer).
    :param recall_method:   Identifier of the method used by the hetero memory for recalling.
    :return:                A dictionary, an entry for each dataset, with a list of sequences.
    """
    rows = commons.datasets_to_codomains
    sequences = {}
    for orig_ds in commons.datasets:
        sequences[orig_ds] = []
        print(f'Generating sequences starting at {orig_ds}')
        if len(features[orig_ds]) == 0:
            continue
        progress_columns = [
            TextColumn('[progress.description]{task.description}'),
            BarColumn(),
            TextColumn('{task.completed}/{task.total}'),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]
        with Progress(*progress_columns) as progress:
            task = progress.add_task(f'Seq {orig_ds}', total=len(features[orig_ds]))
            for feats, pair_id in zip(features[orig_ds], labels[orig_ds]):
                current_dataset = orig_ds
                current_features = feats
                step_datasets = [current_dataset]
                sequence = [
                    qds[current_dataset].dequantize(current_features, rows[current_dataset])
                ]
                while len(sequence) < commons.sequence_length:
                    if current_dataset == commons.left_dataset:
                        current_features, recognized, _, _, _ = hetero.recall_from_left(
                            current_features, method=recall_method, label=pair_id
                        )
                    else:
                        current_features, recognized, _, _, _ = hetero.recall_from_right(
                            current_features, method=recall_method, label=pair_id
                        )
                    if not recognized:
                        break
                    current_dataset = commons.alt(current_dataset)
                    step_datasets.append(current_dataset)
                    sequence.append(
                        qds[current_dataset].dequantize(
                            current_features, rows[current_dataset]
                        )
                    )
                sequences[orig_ds].append(
                    {
                        'pair_id': int(pair_id),
                        'datasets': step_datasets,
                        'embeddings': np.array(sequence, dtype=np.float32),
                    }
                )
                progress.advance(task)
            current_dataset = orig_ds
            current_features = feats
            step_datasets = [current_dataset]
            sequence = [
                qds[current_dataset].dequantize(current_features, rows[current_dataset])
            ]
            while len(sequence) < commons.sequence_length:
                if current_dataset == commons.left_dataset:
                    current_features, recognized, _, _, _ = hetero.recall_from_left(
                        current_features, method=recall_method, label=pair_id
                    )
                else:
                    current_features, recognized, _, _, _ = hetero.recall_from_right(
                        current_features, method=recall_method, label=pair_id
                    )
                if not recognized:
                    break
                current_dataset = commons.alt(current_dataset)
                step_datasets.append(current_dataset)
                sequence.append(
                    qds[current_dataset].dequantize(
                        current_features, rows[current_dataset]
                    )
                )
            sequences[orig_ds].append(
                {
                    'pair_id': int(pair_id),
                    'datasets': step_datasets,
                    'embeddings': np.array(sequence, dtype=np.float32),
                }
            )
            commons.print_counter(counter, 10, 1, '+', name=counter_name)
            counter += 1
    return sequences


def sequences_of_memories(recall_method, filling_percent, es):
    cols = commons.datasets_to_domains
    rows = commons.datasets_to_codomains
    sequences = []
    labels = []
    targets = []
    for fold in range(commons.n_folds):
        filling_features, filling_labels, testing_features, testing_labels = (
            load_features_n_labels(fold, es)
        )
        testing_features, testing_labels = sample_features_for_sequencing(
            testing_features, testing_labels
        )
        raw_testing_features = {
            dataset: np.array(testing_features[dataset], dtype=np.float32, copy=True)
            for dataset in commons.datasets
        }
        match_labels(filling_features, filling_labels)
        total = len(filling_labels[commons.left_dataset])
        top = int(total * filling_percent / 100.0)
        qds = {}
        for dataset in commons.datasets:
            qd = qudeq.QuDeq(
                filling_features[dataset], percentiles=commons.use_percentiles
            )
            filling_features[dataset] = qd.quantize(
                filling_features[dataset][:top], rows[dataset]
            )
            testing_features[dataset] = qd.quantize(
                testing_features[dataset], rows[dataset]
            )
            qds[dataset] = qd
        hetero = HeteroAssociativeMemory(
            cols[commons.left_dataset],
            cols[commons.right_dataset],
            rows[commons.left_dataset],
            rows[commons.right_dataset],
            es,
            fold,
            qds[commons.left_dataset],
            qds[commons.right_dataset],
            None,
        )
        for left_feat, right_feat in zip(
            filling_features[commons.left_dataset],
            filling_features[commons.right_dataset],
        ):
            hetero.register(left_feat, right_feat)
        seqs = produce_testing_sequences(
            hetero, testing_features, testing_labels, qds, recall_method
        )
        sequences.append(seqs)
        labels.append(testing_labels)
        targets.append(
            {
                dataset: {
                    int(pair_id): raw_feature
                    for pair_id, raw_feature in zip(
                        testing_labels[dataset], raw_testing_features[dataset]
                    )
                }
                for dataset in commons.datasets
            }
        )
    return sequences, labels, targets


def decode_sequence_texts(records):
    decoded_texts = {}
    pending_embeddings = {dataset: [] for dataset in commons.datasets}
    pending_positions = {dataset: [] for dataset in commons.datasets}

    for record_idx, record in enumerate(records):
        decoded_texts[record_idx] = [None] * len(record['datasets'])
        for step_idx, (dataset, embedding) in enumerate(
            zip(record['datasets'], record['embeddings'])
        ):
            pending_embeddings[dataset].append(embedding)
            pending_positions[dataset].append((record_idx, step_idx))

    for dataset in commons.datasets:
        if not pending_embeddings[dataset]:
            continue
        embeddings = np.array(pending_embeddings[dataset], dtype=np.float32)
        texts = neural_net.decode_embeddings(embeddings, dataset)
        for (record_idx, step_idx), text in zip(pending_positions[dataset], texts):
            decoded_texts[record_idx][step_idx] = text

    return decoded_texts


def evaluate_sequence_records(records, targets_by_dataset):
    total_steps = 0
    recognized_steps = 0
    recalled_by_dataset = {dataset: [] for dataset in commons.datasets}
    targets_by_recall_dataset = {dataset: [] for dataset in commons.datasets}

    for record in records:
        datasets = record['datasets']
        embeddings = record['embeddings']
        pair_id = record['pair_id']
        total_steps += max(0, commons.sequence_length - 1)
        recognized_steps += max(0, len(datasets) - 1)
        for dataset, embedding in zip(datasets[1:], embeddings[1:]):
            recalled_by_dataset[dataset].append(embedding)
            targets_by_recall_dataset[dataset].append(targets_by_dataset[dataset][pair_id])

    metrics = {
        'recognized': recognized_steps,
        'unknown': total_steps - recognized_steps,
        'recognition_rate': 0.0 if total_steps == 0 else recognized_steps / float(total_steps),
        'mean_cosine': np.nan,
        'cosine_score': 0.0,
        'mean_cosine_roundtrip': np.nan,
        'cosine_roundtrip_score': 0.0,
        'mean_l2': np.nan,
    }

    cosines = []
    roundtrip_cosines = []
    l2_distances = []
    for dataset in commons.datasets:
        if not recalled_by_dataset[dataset]:
            continue
        recalled = np.array(recalled_by_dataset[dataset], dtype=np.float32)
        targets = np.array(targets_by_recall_dataset[dataset], dtype=np.float32)
        valid = np.all(np.isfinite(recalled), axis=1)
        valid &= np.all(np.isfinite(targets), axis=1)
        if not np.any(valid):
            continue
        recalled = recalled[valid]
        targets = targets[valid]
        cosines.append(row_cosine(recalled, targets))
        l2_distances.append(np.linalg.norm(recalled - targets, axis=1))

        roundtrip = neural_net.roundtrip_embeddings(recalled, dataset)
        roundtrip_valid = np.all(np.isfinite(roundtrip), axis=1)
        roundtrip_valid &= np.all(np.isfinite(targets), axis=1)
        if np.any(roundtrip_valid):
            roundtrip_cosines.append(
                row_cosine(roundtrip[roundtrip_valid], targets[roundtrip_valid])
            )

    if cosines:
        cosine_values = np.concatenate(cosines)
        l2_values = np.concatenate(l2_distances)
        metrics['mean_cosine'] = float(np.mean(cosine_values))
        metrics['cosine_score'] = float(cosine_to_score(metrics['mean_cosine']))
        metrics['mean_l2'] = float(np.mean(l2_values))
    if roundtrip_cosines:
        roundtrip_values = np.concatenate(roundtrip_cosines)
        metrics['mean_cosine_roundtrip'] = float(np.mean(roundtrip_values))
        metrics['cosine_roundtrip_score'] = float(
            cosine_to_score(metrics['mean_cosine_roundtrip'])
        )

    return metrics


def save_sequences(sequences, labels, targets, es):
    for fold, (seqs, _, fold_targets) in enumerate(zip(sequences, labels, targets)):
        for orig_ds in commons.datasets:
            records = seqs[orig_ds]
            embeddings = np.array(
                [record['embeddings'] for record in records], dtype=object
            )
            texts_by_record = decode_sequence_texts(records)
            metrics = evaluate_sequence_records(records, fold_targets)
            serialized_records = []
            for record_idx, record in enumerate(records):
                serialized_records.append(
                    {
                        'pair_id': record['pair_id'],
                        'steps': [
                            {
                                'depth': step_idx + 1,
                                'dataset': dataset,
                                'text': texts_by_record[record_idx][step_idx],
                            }
                            for step_idx, dataset in enumerate(record['datasets'])
                        ],
                    }
                )

            embeddings_prefix = os.path.join(
                commons.dreams_path,
                'sequence-embeddings' + commons.dataset_suffix(orig_ds),
            )
            embeddings_filename = commons.data_filename(embeddings_prefix, es, fold)
            commons.create_directory(commons.dirname(embeddings_filename))
            np.save(embeddings_filename, embeddings)

            texts_prefix = os.path.join(
                commons.dreams_path,
                'sequence-texts' + commons.dataset_suffix(orig_ds),
            )
            texts_filename = commons.filename(texts_prefix, es, fold, extension='.json')
            commons.create_directory(commons.dirname(texts_filename))
            with open(texts_filename, 'w', encoding='utf-8') as output_file:
                json.dump(serialized_records, output_file, ensure_ascii=False, indent=2)

            metrics_prefix = os.path.join(
                commons.dreams_path,
                'sequence-metrics' + commons.dataset_suffix(orig_ds),
            )
            metrics_filename = commons.filename(
                metrics_prefix, es, fold, extension='.json'
            )
            commons.create_directory(commons.dirname(metrics_filename))
            with open(metrics_filename, 'w', encoding='utf-8') as output_file:
                json.dump(metrics, output_file, ensure_ascii=False, indent=2)

            print(
                f'Saved {len(records)} SONAR sequences for {orig_ds} in fold {fold}.'
            )
            print(
                'Sequence metrics for '
                + f'{orig_ds}: recognition={metrics["recognition_rate"]:.6f}, '
                + f'cosine={metrics["mean_cosine"]:.6f}, '
                + f'roundtrip={metrics["mean_cosine_roundtrip"]:.6f}, '
                + f'l2={metrics["mean_l2"]:.6f}'
            )


##############################################################################
# Main section


def produce_features_from_data(dataset, es):
    model_prefix = commons.model_name(dataset, es)
    features_prefix = commons.features_name(dataset, es)
    labels_prefix = commons.labels_name(dataset, es)
    data_prefix = commons.data_name(dataset, es)
    neural_net.obtain_features(
        dataset, model_prefix, features_prefix, labels_prefix, data_prefix, es
    )


def characterize_features(dataset, es):
    """Produces a graph of features averages and standard deviations."""
    means, stdvs, hists, proto_kinds = features_parameters(dataset, es)
    save_prototypes(means, stdvs, proto_kinds, dataset, es)
    save_features_graphs(means, stdvs, hists, proto_kinds, dataset, es)


def describe_dataset(dataset, es):
    statistics(dataset, es)
    distances(dataset, es)


def run_separate_evaluation(dataset, es):
    best_memory_sizes = test_memory_sizes(dataset, es)
    print(f'Best memory sizes: {best_memory_sizes}')
    best_filling_percents = test_memory_fills(commons.memory_sizes, dataset, es)
    save_learned_params(commons.memory_sizes, best_filling_percents, dataset, es)


def run_evaluation(es):
    test_hetero_fills(es)


def generate_memories(recall_method, proto_kind_suffix, es):
    remember(recall_method, proto_kind_suffix, es)


def generate_sequences(recall_method, filling_percent, es):
    sequences, labels, targets = sequences_of_memories(recall_method, filling_percent, es)
    save_sequences(sequences, labels, targets, es)


def requested_datasets(dataset_name):
    if dataset_name is None or dataset_name == 'all':
        return commons.datasets
    if dataset_name in commons.datasets:
        return [dataset_name]
    raise ValueError(f'Dataset {dataset_name} is not supported.')


if __name__ == '__main__':
    args = docopt(__doc__)
    if args['--relsmean']:
        n = int(args['--relsmean'])
        if n <= 0:
            print('The mean of relations must be a positive integer.')
            exit(1)
        else:
            commons.mean_matches = n
    if args['--relsstdv']:
        s = float(args['--relsstdv'])
        if s <= 0:
            print('The standard deviation of relations must be a positive number.')
            exit(1)
        else:
            commons.stdv_matches = n
    if args['--runpath']:
        commons.run_path = args['--runpath']
    if args['es']:
        es_lang = gettext.translation('eam', localedir='locale', languages=['es'])
        es_lang.install()

    # Reading memories parameters
    _prefix = commons.memory_parameters_prefix
    _filename = commons.csv_filename(_prefix)
    if os.path.exists(_filename):
        parameters = np.genfromtxt(_filename, dtype=float, delimiter=',', skip_header=1)
        exp_settings = commons.ExperimentSettings(params=parameters)
    else:
        print(f'Memory parameters file not found: {_filename}. Using default parameters.')
        exp_settings = commons.ExperimentSettings()
    print(f'Working directory: {commons.run_path}')
    print(f'Experimental settings: {exp_settings}')
    np.set_printoptions(linewidth=1024)
    start_of_experiment = time.time()
    # PROCESSING OF MAIN OPTIONS.
    random.seed(0)
    #if args['-n']:
    #    _dataset = args['<dataset>']
    #    if _dataset in commons.datasets:
    #        create_and_train_network(_dataset, exp_settings)
    #    else:
    #        print(f'Dataset {_dataset} is not supported.')
    if args['-f']:
        for dataset in requested_datasets(args['<dataset>']):
            produce_features_from_data(dataset, exp_settings)
    elif args['-c']:
        for dataset in requested_datasets(args['<dataset>']):
            print(f'Characterizing features for {dataset}...')
            characterize_features(dataset, exp_settings)
    elif args['-d']:
        for dataset in requested_datasets(args['<dataset>']):
            describe_dataset(dataset, exp_settings)
    elif args['-s']:
        for dataset in requested_datasets(args['<dataset>']):
            run_separate_evaluation(dataset, exp_settings)
    elif args['-e']:
        run_evaluation(exp_settings)
    elif args['-r']:
        generate_memories(
            commons.recall_with_sampling_n_search,
            commons.constructed_suffix,
            exp_settings,
        )
    elif args['-q']:
        generate_memories(
            commons.recall_with_cue, commons.constructed_suffix, exp_settings
        )
    elif args['-p']:
        _kind = args['<kind>']
        if _kind not in commons.proto_kinds:
            raise ValueError(f'"{_kind}" is not a prototype kind')
        generate_memories(
            commons.recall_with_protos, commons.proto_kind_suffix(_kind), exp_settings
        )
    elif args['-P']:
        _kind = args['<kind>']
        if _kind not in commons.proto_kinds:
            raise ValueError(f'"{_kind}" is not a prototype kind')
        generate_memories(
            commons.recall_with_correct_proto,
            commons.proto_kind_suffix(_kind),
            exp_settings,
        )
    elif args['-u']:
        generate_sequences(
            commons.sequence_recall_method, commons.sequence_recall_fill, exp_settings
        )
    end_of_experiment = time.time()
    duration_of_experiment = end_of_experiment - start_of_experiment
    print(f'DURATION: {duration_of_experiment:.1f} seconds')
