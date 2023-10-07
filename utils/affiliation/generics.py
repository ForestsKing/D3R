import glob
import gzip
import math
import os
from itertools import groupby
from operator import itemgetter


def convert_vector_to_events(vector=None):
    if vector is None:
        vector = [0, 1, 1, 0, 0, 1, 0]
    positive_indexes = [idx for idx, val in enumerate(vector) if val > 0]
    events = []
    for k, g in groupby(enumerate(positive_indexes), lambda ix: ix[0] - ix[1]):
        cur_cut = list(map(itemgetter(1), g))
        events.append((cur_cut[0], cur_cut[-1]))

    events = [(x, y + 1) for (x, y) in events]

    return events


def infer_Trange(events_pred, events_gt):
    if len(events_gt) == 0:
        raise ValueError('The gt events should contain at least one event')
    if len(events_pred) == 0:
        return infer_Trange(events_gt, events_gt)

    min_pred = min([x[0] for x in events_pred])
    min_gt = min([x[0] for x in events_gt])
    max_pred = max([x[1] for x in events_pred])
    max_gt = max([x[1] for x in events_gt])
    Trange = (min(min_pred, min_gt), max(max_pred, max_gt))
    return Trange


def has_point_anomalies(events):
    if len(events) == 0:
        return False
    return min([x[1] - x[0] for x in events]) == 0


def _sum_wo_nan(vec):
    vec_wo_nan = [e for e in vec if not math.isnan(e)]
    return sum(vec_wo_nan)


def _len_wo_nan(vec):
    vec_wo_nan = [e for e in vec if not math.isnan(e)]
    return len(vec_wo_nan)


def read_gz_data(filename='data/machinetemp_groundtruth.gz'):
    with gzip.open(filename, 'rb') as f:
        content = f.read().splitlines()
    content = [int(x) for x in content]
    return content


def read_all_as_events():
    filepaths = glob.glob('data/*.gz')
    datasets = dict()
    Tranges = dict()
    for filepath in filepaths:
        vector = read_gz_data(filepath)
        events = convert_vector_to_events(vector)
        cut_filepath = (os.path.split(filepath)[1]).split('_')
        data_name = cut_filepath[0]
        algo_name = (cut_filepath[1]).split('.')[0]
        if not data_name in datasets:
            datasets[data_name] = dict()
            Tranges[data_name] = (0, len(vector))
        datasets[data_name][algo_name] = events
    return datasets, Tranges


def f1_func(p, r):
    return 2 * p * r / (p + r)
