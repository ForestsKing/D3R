import math

from utils.affiliation.generics import _sum_wo_nan


def interval_length(J=(1, 2)):
    if J is None:
        return 0
    return J[1] - J[0]


def sum_interval_lengths(Is=None):
    if Is is None:
        Is = [(1, 2), (3, 4), (5, 6)]
    return sum([interval_length(I) for I in Is])


def interval_intersection(I=(1, 3), J=(2, 4)):
    if I is None:
        return None
    if J is None:
        return None

    I_inter_J = (max(I[0], J[0]), min(I[1], J[1]))
    if I_inter_J[0] >= I_inter_J[1]:
        return None
    else:
        return I_inter_J


def interval_subset(I=(1, 3), J=(0, 6)):
    if (I[0] >= J[0]) and (I[1] <= J[1]):
        return True
    else:
        return False


def cut_into_three_func(I, J):
    if I is None:
        return None, None, None

    I_inter_J = interval_intersection(I, J)
    if I == I_inter_J:
        I_before = None
        I_after = None
    elif I[1] <= J[0]:
        I_before = I
        I_after = None
    elif I[0] >= J[1]:
        I_before = None
        I_after = I
    elif (I[0] <= J[0]) and (I[1] >= J[1]):
        I_before = (I[0], I_inter_J[0])
        I_after = (I_inter_J[1], I[1])
    elif I[0] <= J[0]:
        I_before = (I[0], I_inter_J[0])
        I_after = None
    elif I[1] >= J[1]:
        I_before = None
        I_after = (I_inter_J[1], I[1])
    else:
        raise ValueError('unexpected unconsidered case')
    return I_before, I_inter_J, I_after


def get_pivot_j(I, J):
    if interval_intersection(I, J) is not None:
        raise ValueError('I and J should have a void intersection')

    j_pivot = None
    if max(I) <= min(J):
        j_pivot = min(J)
    elif min(I) >= max(J):
        j_pivot = max(J)
    else:
        raise ValueError('I should be outside J')
    return j_pivot


def integral_mini_interval(I, J):
    if I is None:
        return 0

    j_pivot = get_pivot_j(I, J)
    a = min(I)
    b = max(I)
    return (b - a) * abs((j_pivot - (a + b) / 2))


def integral_interval_distance(I, J):
    def f(I_cut):
        return integral_mini_interval(I_cut, J)

    def f0(I_middle):
        return (0)

    cut_into_three = cut_into_three_func(I, J)
    d_left = f(cut_into_three[0])
    d_middle = f0(cut_into_three[1])
    d_right = f(cut_into_three[2])
    return d_left + d_middle + d_right


def integral_mini_interval_P_CDFmethod__min_piece(I, J, E):
    if interval_intersection(I, J) is not None:
        raise ValueError('I and J should have a void intersection')
    if not interval_subset(J, E):
        raise ValueError('J should be included in E')
    if not interval_subset(I, E):
        raise ValueError('I should be included in E')

    e_min = min(E)
    j_min = min(J)
    j_max = max(J)
    e_max = max(E)
    i_min = min(I)
    i_max = max(I)

    d_min = max(i_min - j_max, j_min - i_max)
    d_max = max(i_max - j_max, j_min - i_min)
    m = min(j_min - e_min, e_max - j_max)
    A = min(d_max, m) ** 2 - min(d_min, m) ** 2
    B = max(d_max, m) - max(d_min, m)
    C = (1 / 2) * A + m * B
    return (C)


def integral_mini_interval_Pprecision_CDFmethod(I, J, E):
    integral_min_piece = integral_mini_interval_P_CDFmethod__min_piece(I, J, E)

    e_min = min(E)
    j_min = min(J)
    j_max = max(J)
    e_max = max(E)
    i_min = min(I)
    i_max = max(I)
    d_min = max(i_min - j_max, j_min - i_max)
    d_max = max(i_max - j_max, j_min - i_min)
    integral_linear_piece = (1 / 2) * (d_max ** 2 - d_min ** 2)
    integral_remaining_piece = (j_max - j_min) * (i_max - i_min)

    DeltaI = i_max - i_min
    DeltaE = e_max - e_min

    output = DeltaI - (1 / DeltaE) * (integral_min_piece + integral_linear_piece + integral_remaining_piece)
    return output


def integral_interval_probaCDF_precision(I, J, E):
    def f(I_cut):
        if I_cut is None:
            return 0
        else:
            return integral_mini_interval_Pprecision_CDFmethod(I_cut, J, E)

    def f0(I_middle):
        if I_middle is None:
            return 0
        else:
            return max(I_middle) - min(I_middle)

    cut_into_three = cut_into_three_func(I, J)
    d_left = f(cut_into_three[0])
    d_middle = f0(cut_into_three[1])
    d_right = f(cut_into_three[2])
    return d_left + d_middle + d_right


def cut_J_based_on_mean_func(J, e_mean):
    if J is None:
        J_before = None
        J_after = None
    elif e_mean >= max(J):
        J_before = J
        J_after = None
    elif e_mean <= min(J):
        J_before = None
        J_after = J
    else:
        J_before = (min(J), e_mean)
        J_after = (e_mean, max(J))

    return J_before, J_after


def integral_mini_interval_Precall_CDFmethod(I, J, E):
    i_pivot = get_pivot_j(J, I)
    e_min = min(E)
    e_max = max(E)
    e_mean = (e_min + e_max) / 2

    if i_pivot <= min(E):
        return 0
    elif i_pivot >= max(E):
        return 0

    cut_J_based_on_e_mean = cut_J_based_on_mean_func(J, e_mean)
    J_before = cut_J_based_on_e_mean[0]
    J_after = cut_J_based_on_e_mean[1]

    iemin_mean = (e_min + i_pivot) / 2
    cut_Jbefore_based_on_iemin_mean = cut_J_based_on_mean_func(J_before, iemin_mean)
    J_before_closeE = cut_Jbefore_based_on_iemin_mean[
        0]
    J_before_closeI = cut_Jbefore_based_on_iemin_mean[
        1]

    iemax_mean = (e_max + i_pivot) / 2
    cut_Jafter_based_on_iemax_mean = cut_J_based_on_mean_func(J_after, iemax_mean)
    J_after_closeI = cut_Jafter_based_on_iemax_mean[0]
    J_after_closeE = cut_Jafter_based_on_iemax_mean[1]

    if J_before_closeE is not None:
        j_before_before_min = min(J_before_closeE)
        j_before_before_max = max(J_before_closeE)
    else:
        j_before_before_min = math.nan
        j_before_before_max = math.nan

    if J_before_closeI is not None:
        j_before_after_min = min(J_before_closeI)
        j_before_after_max = max(J_before_closeI)
    else:
        j_before_after_min = math.nan
        j_before_after_max = math.nan

    if J_after_closeI is not None:
        j_after_before_min = min(J_after_closeI)
        j_after_before_max = max(J_after_closeI)
    else:
        j_after_before_min = math.nan
        j_after_before_max = math.nan

    if J_after_closeE is not None:
        j_after_after_min = min(J_after_closeE)
        j_after_after_max = max(J_after_closeE)
    else:
        j_after_after_min = math.nan
        j_after_after_max = math.nan

    if i_pivot >= max(J):
        part1_before_closeE = (i_pivot - e_min) * (
                j_before_before_max - j_before_before_min)
        part2_before_closeI = 2 * i_pivot * (j_before_after_max - j_before_after_min) - (
                j_before_after_max ** 2 - j_before_after_min ** 2)
        part3_after_closeI = 2 * i_pivot * (j_after_before_max - j_after_before_min) - (
                j_after_before_max ** 2 - j_after_before_min ** 2)
        part4_after_closeE = (e_max + i_pivot) * (j_after_after_max - j_after_after_min) - (
                j_after_after_max ** 2 - j_after_after_min ** 2)
        out_parts = [part1_before_closeE, part2_before_closeI, part3_after_closeI, part4_after_closeE]
    elif i_pivot <= min(J):
        part1_before_closeE = (j_before_before_max ** 2 - j_before_before_min ** 2) - (e_min + i_pivot) * (
                j_before_before_max - j_before_before_min)
        part2_before_closeI = (j_before_after_max ** 2 - j_before_after_min ** 2) - 2 * i_pivot * (
                j_before_after_max - j_before_after_min)
        part3_after_closeI = (j_after_before_max ** 2 - j_after_before_min ** 2) - 2 * i_pivot * (
                j_after_before_max - j_after_before_min)
        part4_after_closeE = (e_max - i_pivot) * (
                j_after_after_max - j_after_after_min)
        out_parts = [part1_before_closeE, part2_before_closeI, part3_after_closeI, part4_after_closeE]
    else:
        raise ValueError('The i_pivot should be outside J')

    out_integral_min_dm_plus_d = _sum_wo_nan(out_parts)

    DeltaJ = max(J) - min(J)
    DeltaE = max(E) - min(E)
    C = DeltaJ - (1 / DeltaE) * out_integral_min_dm_plus_d

    return C


def integral_interval_probaCDF_recall(I, J, E):
    def f(J_cut):
        if J_cut is None:
            return 0
        else:
            return integral_mini_interval_Precall_CDFmethod(I, J_cut, E)

    def f0(J_middle):
        if J_middle is None:
            return 0
        else:
            return max(J_middle) - min(J_middle)

    cut_into_three = cut_into_three_func(J, I)
    d_left = f(cut_into_three[0])
    d_middle = f0(cut_into_three[1])
    d_right = f(cut_into_three[2])
    return d_left + d_middle + d_right
