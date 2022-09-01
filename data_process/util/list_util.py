import string

from util import *


def longest_common_sublist(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]  
    mmax = 0 
    p = 0  
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], p - mmax, p - 1


def find_sublist(x, y):
    l1, l2 = len(x), len(y)
    for i in range(l1):
        if x[i:min(i + l2, l1)] == y:
            return i
    return -1


@dispatch(list)
def interval_nesting(s):
    s_temp = copy.deepcopy(s)
    s_temp.sort(key=lambda si: si[0])
    for i in range(1, len(s_temp)):
        sc = s_temp[i]
        sp = s_temp[i - 1]
        if sc[0] <= sp[1] or sc[0] == sp[0]:
            return True
    return False


@dispatch(list, list)
def interval_nesting(s1, s2):

    s1_temp = copy.deepcopy(s1)
    s2_temp = copy.deepcopy(s2)
    s1_temp.sort(key=lambda si: si[0])
    s2_temp.sort(key=lambda si: si[0])
    loop_1 = 0
    loop_2 = 0
    while loop_1 < len(s1_temp) and loop_2 < len(s2_temp):
        if s1_temp[loop_1][0] < s2_temp[loop_2][0]:
            if s1_temp[loop_1][1] >= s2_temp[loop_2][0]:
                return True
            loop_1 += 1
        elif s1_temp[loop_1][0] > s2_temp[loop_2][0]:
            if s1_temp[loop_1][0] <= s2_temp[loop_2][1]:
                return True
            loop_2 += 1
        else:
            return True
    return False


def interval_merge(s):
    if len(s) == 0:
        return []
    res = list()
    s_temp = copy.deepcopy(s)
    s_temp.sort(key=lambda si: si[0])
    res.append(s_temp[0])
    for i in range(1, len(s_temp)):
        sc = s_temp[i]
        sp = res[-1]
        if sc[0] <= sp[1]:
            res[-1][1] = max(res[-1][1], sc[1])
        else:
            res.append(sc)
    return res


def interval_deconflict(s):
    if len(s) == 0:
        return []
    res = []
    s_temp = copy.deepcopy(s)
    s_temp.sort(key=lambda si: si[1])
    res.append(s_temp[0])
    for i in range(1, len(s_temp)):
        if s_temp[i][0] > res[-1][1]:
            res.append(s_temp[i])
        elif (s_temp[i][1] - s_temp[i][0]) > (res[-1][1] - res[-1][0]):
            res[-1] = s_temp[i]
    return res


def interval_len(s):
    res = 0
    s_temp = interval_merge(s)
    for i in range(len(s_temp)):
        assert i == 0 or (i > 0 and s_temp[i - 1][1] < s_temp[i][0])
        res += (s_temp[i][1] - s_temp[i][0] + 1)
    return res


def deduplication_keep_order(input_list):
    return sorted(set(input_list), key=input_list.index)


def only_punctuation(input_list):
    punc = string.punctuation
    for i in input_list:
        if i not in punc:
            return False
    return True


def only_subword(input_list):
    for i in input_list:
        if "##" not in i:
            return False
    return True


def rindex(lst, val, start=None):
    if start is None:
        start = len(lst)-1
    for i in range(start, -1, -1):
        if lst[i] == val:
            return i


if __name__ == '__main__':
    # print(longest_common_sublist("apple pie available", "apple pies"))
    print(longest_common_sublist(["a", "b", "c", "a", "b"], ["a", "b"]))
