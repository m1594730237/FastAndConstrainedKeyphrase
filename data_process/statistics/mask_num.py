from statistics import *
from nltk import PorterStemmer


def mask_num(items, num_mask, split_percent, cached_path, auxiliary_sentence):
    stemmer = PorterStemmer()

    total_present = 0
    total_absent = 0
    total_left = 0
    total_right = 0
    total_center = 0
    total_present_also_center = 0
    total_absolute_absent = 0

    total_absolute_absent_doc = 0
    total_semi_absent_doc = 0
    total_only_present_doc = 0
    total_only_absent_doc = 0
    total_present_nest_doc = 0
    total_center_nest_doc = 0
    total_pc_nest_doc = 0

    absent_center_len_dict = dict()
    present_len_dict = dict()
    absolute_absent_len_dict = dict()
    mask_prediction_len_dict = dict()
    content_len_dict = dict()
    absolute_absent_num_dict = dict()
    hit_miss_dict = {
        "left_hit": [0] * num_mask,
        "right_hit": [0] * num_mask,
        "left_miss": [0] * num_mask,
        "right_miss": [0] * num_mask,
    }

    tokenization = Tokenization()
    all_content, all_keywords = item2token(items, cached_path=cached_path)
    all_content_word = [" ".join(c).replace(" ##", "").split(" ") for c in all_content]
    all_content_stem = [[stemmer.stem(cc) for cc in c] for c in all_content_word]
    all_keywords_word = [[" ".join(k).replace(" ##", "").split(" ") for k in ks] for ks in all_keywords]
    all_keywords_stem = [[[stemmer.stem(kk) for kk in k]for k in ks] for ks in all_keywords_word]
    split_percent.sort()
    assert len(all_content) == len(all_keywords)
    auxiliary_len = len(tokenization.tokenize(auxiliary_sentence))
    for i in tqdm(range(len(all_content))):
        content = all_content_stem[i]
        has_absent = False
        has_present = False
        has_absolute_absent = False
        has_semi_absent = False
        all_absent_center = list()
        all_present = list()
        absolute_absent_num_in_doc = 0
        prediction_len = 0
        for keyword in all_keywords_stem[i]:
            center, _, _ = longest_common_sublist(content, keyword)
            center_idx = find_sublist(keyword, center)
            left_len = center_idx
            right_len = len(keyword) - len(center) - left_len
            is_absent = left_len > 0 or right_len > 0
            is_absolute_absent = len(center) == 0 and is_absent
            is_semi_absent = is_absent and not is_absolute_absent

            has_absent = has_absent or is_absent
            has_present = has_present or not is_absent
            total_present += 1 if len(center) > 0 and not is_absent else 0
            total_absent += 1 if is_absent else 0
            total_center += 1 if len(center) > 0 else 0
            total_absolute_absent += 1 if is_absolute_absent else 0
            has_absolute_absent = True if is_absolute_absent else has_absolute_absent
            has_semi_absent = True if is_semi_absent else has_semi_absent

            if is_absolute_absent:
                absolute_absent_len_dict[len(keyword)] = \
                    absolute_absent_len_dict[len(keyword)] + 1 if len(keyword) in absolute_absent_len_dict else 1
                absolute_absent_num_in_doc += 1

            if is_absent:
                hit_miss_dict["left_hit"] = [s + 1 if num_mask - left_len <= idx else s
                                             for idx, s in enumerate(hit_miss_dict["left_hit"])]
                hit_miss_dict["right_hit"] = [s + 1 if right_len > idx else s
                                              for idx, s in enumerate(hit_miss_dict["right_hit"])]
                hit_miss_dict["left_miss"] = [s + 1 if num_mask - left_len > idx else s
                                              for idx, s in enumerate(hit_miss_dict["left_miss"])]
                hit_miss_dict["right_miss"] = [s + 1 if right_len <= idx else s
                                               for idx, s in enumerate(hit_miss_dict["right_miss"])]

            if is_absent:
                absent_center_len_dict[len(center)] = \
                    1 if len(center) not in absent_center_len_dict else absent_center_len_dict[len(center)] + 1
            else:
                present_len_dict[len(center)] = \
                    1 if len(center) not in present_len_dict else present_len_dict[len(center)] + 1

            if not is_absent:
                all_present.append(center)
            else:
                all_absent_center.append(center)

            if is_absent:
                prediction_len += len(center) * 2 + auxiliary_len + num_mask * 2 + 1  # +1 for <S> or [SEP]

            total_left += left_len
            total_right += right_len

        assert absolute_absent_num_in_doc <= len(all_keywords[i])
        content_len_dict[len(content)] = content_len_dict[len(content)] + 1 if len(content) in content_len_dict else 1
        absolute_absent_num_dict[absolute_absent_num_in_doc] = \
            absolute_absent_num_dict[absolute_absent_num_in_doc] + 1 \
            if absolute_absent_num_in_doc in absolute_absent_num_dict else 1
        if prediction_len > 0:
            mask_prediction_len_dict[prediction_len] = \
                1 if prediction_len not in mask_prediction_len_dict else mask_prediction_len_dict[prediction_len] + 1
        total_only_present_doc += 1 if has_present and not has_absent else 0
        total_only_absent_doc += 1 if has_absent and not has_present else 0
        total_present_also_center += 1 if np.any([p in all_absent_center for p in all_present]) else 0

        total_absolute_absent_doc += 1 if has_absolute_absent else 0
        total_semi_absent_doc += 1 if has_semi_absent else 0
        nest = nest_check(all_present, all_absent_center)
        total_present_nest_doc += 1 if nest["present_present"] else 0
        total_center_nest_doc += 1 if nest["center_center"] else 0
        total_pc_nest_doc += 1 if nest["present_center"] else 0
        # print(f"keyword: {keyword}")
        # print(f"center: {center}")
        # print(f"content: {content}")
        # print(f"left_len: {left_len}")
        # print(f"right_len: {right_len}")
    total_phrase = total_present + total_absent
    total_doc = len(all_content)

    # len_dict_output(content_len_dict, split_percent, total_doc, 10000,
    #                 lambda ii, nn, total: print(f"Content Len = {ii} Num = {nn}({nn / total * 100}%)"),
    #                 lambda s: print(f"Average Content Len = {s}"),
    #                 lambda s, percent, all_num, idx:
    #                 print(f"Content {percent * 100}% Percent Split Len is {idx} ({s / all_num * 100}%)")
    #                 )
    print(f"Average Left [MASK] Num = {total_left / total_absent}")
    print(f"Average Right [MASK] Num = {total_right / total_absent}")
    print("Length Distribution: ")
    # ll = [s if idx == 0 else s - hit_miss_dict['left_hit'][idx - 1]
    #       for idx, s in enumerate(hit_miss_dict['left_hit'])]
    # rr = [s if idx == num_mask - 1 else s - hit_miss_dict['right_hit'][idx + 1]
    #       for idx, s in enumerate(hit_miss_dict['right_hit'])]
    # left_sum_list = []
    # right_sum_list = []
    # left_sum_list.append(total_absent - sum(ll))
    # print(f"Left Tokens Len = 0 Num = {total_absent - sum(ll)}({(total_absent - sum(ll)) / total_absent * 100}%)")
    # for i in range(num_mask - 1, -1, -1):
    #     if i == 0:
    #         pre = ">"
    #     else:
    #         pre = "="
    #     num = ll[i]
    #     left_sum_list.append(num if num_mask - 1 - i == 0 else left_sum_list[num_mask - 1 - i - 1] + num)
    #     print(f"Left Tokens Len {pre} {num_mask - i} Num = {num}({num / total_absent * 100}%)")
    # split_percent_output(left_sum_list, split_percent,
    #                      lambda s, percent, all_num, idx:
    #                      print(f"Left Tokens {percent * 100}% Percent Split Len is {idx} ({s / all_num * 100}%)"),
    #                      total_absent)
    # right_sum_list.append(total_absent - sum(rr))
    # print(f"Right Tokens Len = 0 Num = {total_absent - sum(rr)}({(total_absent - sum(rr)) / total_absent * 100}%)")
    # for i in range(num_mask):
    #     if i == num_mask - 1:
    #         pre = ">"
    #     else:
    #         pre = "="
    #     num = rr[i]
    #     right_sum_list.append(num if i == 0 else right_sum_list[i - 1] + num)
    #     print(f"Right Tokens Len {pre} {i + 1} Num = {num}({num / total_absent * 100}%)")
    # split_percent_output(left_sum_list, split_percent,
    #                      lambda s, percent, all_num, idx:
    #                      print(f"Right Tokens {percent * 100}% Percent Split Len is {idx} ({s / all_num * 100}%)"),
    #                      total_absent)
    # len_dict_output(mask_prediction_len_dict, split_percent, total_doc-total_only_present_doc, 3000,
    #                 lambda ii, nn, total: print(f"MASK Prediction Len = {ii} Num = {nn}({nn / total * 100}%)"),
    #                 lambda s: print(f"Average MASK Prediction Len = {s}"),
    #                 lambda s, percent, all_num, idx:
    #                 print(f"MASK Prediction {percent * 100}% Percent Split Len is {idx} ({s / all_num * 100}%)")
    #                 )
    print(f"Total Present Phrase Num = {total_present}")
    print(f"Total Absent Phrase Num = {total_absent}")
    print(f"If Mask Num = {num_mask}, "
          f"Token Num = {hit_miss_dict['left_hit'] + hit_miss_dict['right_hit']}, "
          f"<T> Num = {hit_miss_dict['left_miss'] + hit_miss_dict['right_miss']}")
    print("<T> Weight = " +
          str([h / m for h, m in zip(hit_miss_dict['left_hit'] + hit_miss_dict['right_hit'],
                                     hit_miss_dict['left_miss'] + hit_miss_dict['right_miss'])]))
    print("Average <T> Weight = " +
          str(sum(hit_miss_dict['left_hit'] + hit_miss_dict['right_hit']) /
              sum(hit_miss_dict['left_miss'] + hit_miss_dict['right_miss'])))

    print(f"Absolutely Absent Phrase Num = {total_absolute_absent}({total_absolute_absent / total_absent * 100}% "
          f"in Absent Phrase)")
    print(f"Semi-Absent Phrase Num = {total_absent - total_absolute_absent}"
          f"({(total_absent - total_absolute_absent) / total_absent * 100}% "
          f"in Absent Phrase)")
    print(f"Semi-Absent Doc Num = {total_semi_absent_doc}({total_semi_absent_doc/total_doc*100}% in Total Doc)")
    print(f"Absolutely Absent Phrase Doc Num = {total_absolute_absent_doc}"
          f"({total_absolute_absent_doc / total_doc * 100}% "
          f"in All Doc)")
    # len_dict_output(absolute_absent_len_dict, split_percent, total_absolute_absent, 2000,
    #                 lambda ii, nn, total: print(f"Absolute Absent Len = {ii} Num = {nn}({nn / total * 100}%)"),
    #                 lambda s: print(f"Average Absolutely Absent Len = {s}"),
    #                 lambda s, percent, all_num, idx:
    #                 print(f"Absolute Absent {percent * 100}% Percent Split Len is {idx} ({s / all_num * 100}%)")
    #                 )
    # len_dict_output(absolute_absent_num_dict, split_percent, total_doc, 200,
    #                 lambda ii, nn, total: print(f"Absolute Absent Num = {ii} Doc Num = {nn}({nn / total * 100}%)"),
    #                 lambda s: print(f"Average Absolutely Absent Num in Doc = {s}"),
    #                 lambda s, percent, all_num, idx:
    #                 print(f"Absolute Absent Num {percent * 100}% Percent Split Len is {idx} ({s / all_num * 100}%)")
    #                 )

    print(f"Present is also Center: {total_present_also_center}({total_present_also_center / total_phrase * 100}%)")
    print(f"Total Absent Center Num(Total Absent Phrase Num) = {total_absent}({total_absent / total_phrase * 100}%)")
    # len_dict_output(present_len_dict, split_percent, total_present, 20,
    #                 lambda ii, nn, total: print(f"Present Len = {ii} Num = {nn}({nn / total * 100}%)"),
    #                 lambda s: print(f"Average Present Len = {s}"),
    #                 lambda s, percent, all_num, idx:
    #                 print(f"Present {percent * 100}% Percent Split Len is {idx} ({s / all_num * 100}%)")
    #                 )
    #
    # len_dict_output(absent_center_len_dict, split_percent, total_absent, 2000,
    #                 lambda ii, nn, total: print(f"Absent Center Len = {ii} Num = {nn}({nn / total * 100}%)"),
    #                 lambda s: print(f"Average Absent Center Len = {s}"),
    #                 lambda s, percent, all_num, idx:
    #                 print(f"Absent Center {percent * 100}% Percent Split Len is {idx} ({s / all_num * 100}%)")
    #                 )

    print(
        f"Articles with Only Present Phrases Num = {total_only_present_doc}({total_only_present_doc / total_doc * 100}%)")
    print(
        f"Articles with Only Absent Phrases Num = {total_only_absent_doc}({total_only_absent_doc / total_doc * 100}%)")

    print(f"Present Phrase Nested Doc Num = {total_present_nest_doc}({total_present_nest_doc / total_doc * 100}%)")
    print(f"Center Phrase Nested Doc Num = {total_center_nest_doc}({total_center_nest_doc / total_doc * 100}%)")
    print(f"Present and Center Phrase Nested Doc Num = {total_pc_nest_doc}({total_pc_nest_doc / total_doc * 100}%)")
    print(f"AVG Present Phrases Num = {total_present/total_doc}")
    print(f"AVG Absent Phrases Num = {total_absent/total_doc}")



