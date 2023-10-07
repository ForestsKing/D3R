from utils.affiliation.integral_interval import interval_intersection


def t_start(j, Js=None, Trange=(1, 10)):
    if Js is None:
        Js = [(1, 2), (3, 4), (5, 6)]
    b = max(Trange)
    n = len(Js)
    if j == n:
        return 2 * b - t_stop(n - 1, Js, Trange)
    else:
        return Js[j][0]


def t_stop(j, Js=None, Trange=(1, 10)):
    if Js is None:
        Js = [(1, 2), (3, 4), (5, 6)]
    if j == -1:
        a = min(Trange)
        return 2 * a - t_start(0, Js, Trange)
    else:
        return Js[j][1]


def E_gt_func(j, Js, Trange):
    range_left = (t_stop(j - 1, Js, Trange) + t_start(j, Js, Trange)) / 2
    range_right = (t_stop(j, Js, Trange) + t_start(j + 1, Js, Trange)) / 2
    return range_left, range_right


def get_all_E_gt_func(Js, Trange):
    E_gt = [E_gt_func(j, Js, Trange) for j in range(len(Js))]
    return E_gt


def affiliation_partition(Is=None, E_gt=None):
    if E_gt is None:
        E_gt = [(1, 2.5), (2.5, 4.5), (4.5, 10)]
    if Is is None:
        Is = [(1, 1.5), (2, 5), (5, 6), (8, 9)]
    out = [None] * len(E_gt)
    for j in range(len(E_gt)):
        E_gt_j = E_gt[j]
        discarded_idx_before = [I[1] < E_gt_j[0] for I in Is]
        discarded_idx_after = [I[0] > E_gt_j[1] for I in Is]
        kept_index = [not (a or b) for a, b in zip(discarded_idx_before, discarded_idx_after)]
        Is_j = [x for x, y in zip(Is, kept_index)]
        out[j] = [interval_intersection(I, E_gt[j]) for I in Is_j]
    return out
