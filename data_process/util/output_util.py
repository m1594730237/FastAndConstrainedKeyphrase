
def split_percent_output(sum_list, percent_list, output_func, all_num=None):
    percent_i = 0
    if all_num is None:
        all_num = sum(sum_list)
    for idx, s in enumerate(sum_list):
        if percent_i >= len(percent_list):
            return
        if s/(all_num + 1e-6) > percent_list[percent_i]:
            output_func(s, percent_list[percent_i], all_num, idx)
            percent_i += 1


def len_dict_output(len_dict, split_percent, total, max_len, output_func, average_output_func, split_output_func):
    len_sum = []
    for i in range(max_len + 1):
        num = 0 if i not in len_dict else len_dict[i]
        len_sum.append(num if i == 0 else len_sum[i - 1] + num)
        if num > 0:
            output_func(i, num, total)
    average_output_func(str(sum([0 if i not in len_dict else len_dict[i] * i for i in range(max_len + 1)])
                            / (total + 1e-6)))
    split_percent_output(len_sum, split_percent, split_output_func, total)




