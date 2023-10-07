import math

from utils.affiliation.affiliation_zone import get_all_E_gt_func, affiliation_partition
from utils.affiliation.integral_interval import integral_interval_distance, sum_interval_lengths, \
    integral_interval_probaCDF_precision, interval_length, integral_interval_probaCDF_recall


def affiliation_precision_distance(Is=None, J=(2, 5.5)):
    if Is is None:
        Is = [(1, 2), (3, 4), (5, 6)]
    if all([I is None for I in Is]):
        return math.nan
    return sum([integral_interval_distance(I, J) for I in Is]) / sum_interval_lengths(Is)


def affiliation_precision_proba(Is=None, J=(2, 5.5), E=(0, 8)):
    if Is is None:
        Is = [(1, 2), (3, 4), (5, 6)]
    if all([I is None for I in Is]):
        return math.nan
    return sum([integral_interval_probaCDF_precision(I, J, E) for I in Is]) / sum_interval_lengths(Is)


def affiliation_recall_distance(Is=None, J=(2, 5.5)):
    if Is is None:
        Is = [(1, 2), (3, 4), (5, 6)]
    Is = [I for I in Is if I is not None]
    if len(Is) == 0:
        return math.inf
    E_gt_recall = get_all_E_gt_func(Is, (-math.inf, math.inf))
    Js = affiliation_partition([J], E_gt_recall)
    return sum([integral_interval_distance(J[0], I) for I, J in zip(Is, Js)]) / interval_length(J)


def affiliation_recall_proba(Is=None, J=(2, 5.5), E=(0, 8)):
    if Is is None:
        Is = [(1, 2), (3, 4), (5, 6)]
    Is = [I for I in Is if I is not None]
    if len(Is) == 0:
        return 0
    E_gt_recall = get_all_E_gt_func(Is, E)
    Js = affiliation_partition([J], E_gt_recall)
    return sum([integral_interval_probaCDF_recall(I, J[0], E) for I, J in zip(Is, Js)]) / interval_length(J)
