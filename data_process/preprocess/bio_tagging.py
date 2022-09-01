import itertools

from nltk import PorterStemmer

from preprocess import *
from util.list_util import find_sublist
from scipy.special import perm
import numpy as np


def bio_tagging(items, num_center_mask, num_absent_mask, max_mask_num, is_test, is_bix,
                split_percent, center_as, absent_as, cached_path, output_path):
    all_content, all_keywords = item2token(items, cached_path=cached_path)
    is_nkw = len(all_keywords) == 1
    if len(all_keywords) == 1:
        all_keywords = [[""]] * len(all_content)
    assert len(all_content) == len(all_keywords)
    center_nest = 0
    present_nest = 0
    cp_nest = 0
    center_len = 0
    present_len = 0
    cp_len = 0
    doc_len = 0
    mask_truncated_num = 0
    absent_truncated_num = 0
    disabled_num = 0
    total_center = 0
    filtered_center = 0
    all_repeat = 0
    len_filtered_center = 0
    absent_truncated_len_dict = dict()
    assert len(center_as.split(WILDCARD_AS)) <= 2
    tokenization = Tokenization()
    center_as_pre, center_as_post = center_as.split(WILDCARD_AS)
    center_as_pre = tokenization.tokenize(center_as_pre)
    center_as_post = tokenization.tokenize(center_as_post)
    absent_as = tokenization.tokenize(absent_as)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    filename_in = os.path.join(output_path, "seq.in")
    filename_out = os.path.join(output_path, "seq.out")
    filename_present = os.path.join(output_path, "seq.labelp")
    filename_center = os.path.join(output_path, "seq.labelc")
    filename_center_word = os.path.join(output_path, "seq.center")
    f_in = open(filename_in, "w+", encoding="utf-8")
    f_out = open(filename_out, "w+", encoding="utf-8")
    f_present = open(filename_present, "w+", encoding="utf-8")
    f_center = open(filename_center, "w+", encoding="utf-8")
    f_center_word = open(filename_center_word, "w+", encoding="utf-8")

    """
        present_se: list([start idx of present kp, end idx of present kp, kp])
        center_se is the same as present_se
    """
    for i in tqdm(range(len(all_content))):
        content = all_content[i]
        center_se = []
        present_se = []
        absolute_absent_kp = []
        for keyword in all_keywords[i]:
            temp_content = content
            cur_len = 0
            final_start = -1
            final_end = -1
            while True:
                center, center_start, center_end = longest_common_sublist(temp_content, keyword)
                assert len(center) == center_end - center_start + 1 and len(center) <= len(keyword)
                temp_content = temp_content[center_end + 1:]
                if len(center) > cur_len and 0 < len(center) < len(keyword):
                    final_start = center_start
                    final_end = center_end
                    cur_len = len(center)
                elif len(center) > cur_len and len(center) == len(keyword):
                    final_start = center_start
                    final_end = center_end
                    cur_len = len(center)
                elif len(center) == 0:
                    break
            if final_end >= 0 and final_start >= 0:
                assert 0 <= final_end - final_start + 1 <= len(keyword)
                c = content[final_start:final_end + 1]
                if 0 < final_end - final_start + 1 < len(keyword):
                    # center_se.append([final_start, final_end, keyword])
                    loop = 0
                    while True:
                        if find_sublist(content[loop:], c) == -1:
                            break
                        loop = loop + find_sublist(content[loop:], c)
                        center_se.append([loop, loop + len(c) - 1, keyword])
                        loop = loop + len(c)
                elif final_end - final_start + 1 == len(keyword):
                    # present_se.append([final_start, final_end, keyword])
                    loop = 0
                    while True:
                        if find_sublist(content[loop:], c) == -1:
                            break
                        loop = loop + find_sublist(content[loop:], c)
                        present_se.append([loop, loop + len(c) - 1, keyword])
                        loop = loop + len(c)
            else:
                assert final_start == -1 and final_end == -1
                absolute_absent_kp.append(keyword)

        '''
            statistic of length
        '''
        center_len += interval_len(center_se)
        present_len += interval_len(present_se)
        cp_len += interval_len(center_se + present_se)
        doc_len += len(content)

        '''
            get line_in、line_out、line_bio_center、line_bio_present and output to file by center_se and present_se
        '''
        line_in = []
        line_out = []
        line_bio_center = []
        line_bio_present = []
        center_se_id = interval_deconflict(center_se)
        present_se_id = interval_deconflict(present_se)

        # filter some keywords
        new_center_se_id = []
        for idx, se in enumerate(center_se_id):
            total_center += 1
            s = se[0]
            e = se[1]
            c = content[s:e + 1]
            if only_subword(c):
                continue
            c = " ".join(c).replace(" ##", "")  # subword to word
            c_len = len(c.replace(" ", ""))  # the length of keyword
            if only_punctuation(c) or c in set(stopwords("en")) or c_len <= 2:
                continue
            new_center_se_id.append(se)
        total_center += len(center_se_id)
        filtered_center += len(center_se_id) - len(new_center_se_id)
        assert len(new_center_se_id) <= len(center_se_id)
        center_se_id = new_center_se_id

        # filter too long absent kp
        new_center_se_id = []
        for idx, se in enumerate(center_se_id):
            s = se[0]
            e = se[1]
            k = se[2]
            c = content[s:e + 1]
            c_new, s_new, e_new = longest_common_sublist(k, c)
            left_len = s_new
            right_len = len(k) - e_new - 1
            if left_len > num_center_mask or right_len > num_center_mask:
                len_filtered_center += 1
                continue
            new_center_se_id.append(se)
        assert len(new_center_se_id) <= len(center_se_id)
        center_se_id = new_center_se_id

        # dedup the keywords
        center_se_id_de = []
        center_dict = dict()
        for idx, se in enumerate(center_se_id):
            s = se[0]
            e = se[1]
            c = content[s:e + 1]
            center_dict[tuple(c)] = se
        for c, se in center_dict.items():
            center_se_id_de.append(se)

        # get the keywords
        line_center_word = []
        for idx, se in enumerate(center_se_id_de):
            s = se[0]
            e = se[1]
            c = content[s:e + 1]
            line_center_word.append(c)

        max_loop_count = (max_mask_num - num_absent_mask) // (num_center_mask * 2)
        # *.in files
        loop_count = 0
        for idx, se in enumerate(center_se_id_de):
            if not is_test and loop_count >= max_loop_count:
                mask_truncated_num += 1
                break
            s = se[0]
            e = se[1]
            c = content[s:e + 1]
            line_in += center_as_pre + c + center_as_post
            line_in += ["[MASK]"] * num_center_mask + c + ["[MASK]"] * num_center_mask
            # if idx != len(center_se_id) - 1:
            line_in += ["<S>"]
            loop_count += 1
        if not is_nkw:
            line_in += absent_as + ["[MASK]"] * num_absent_mask
        line_in += ["[SEP]"]
        before_content_len = len(line_in)
        line_in += content

        # *.out files
        random.shuffle(absolute_absent_kp)
        loop_count = 0
        for idx, se in enumerate(center_se_id_de):
            if not is_test and loop_count >= max_loop_count:
                break
            s = se[0]
            e = se[1]
            k = se[2]
            c = content[s:e + 1]
            c_new, s_new, e_new = longest_common_sublist(k, c)
            assert c_new == c
            line_out += center_as_pre + c + center_as_post
            line_out += ["<T>"] * max(0, num_center_mask - s_new)
            line_out += k[max(0, s_new - num_center_mask):min(len(k), e_new + 1 + num_center_mask)]
            line_out += ["<T>"] * max(0, num_center_mask - (len(k) - e_new - 1))
            # if idx != len(center_se_id) - 1:
            line_out += ["<S>"]
            loop_count += 1
        # if len(line_out) > 0:
        #     line_out += ["<S>"]
        if not is_nkw:
            line_out += absent_as
            init_out_len = len(line_out)
            is_truncated = False
            for idx, kp in enumerate(absolute_absent_kp):
                remained_len = init_out_len + num_absent_mask - len(line_out)
                assert remained_len >= 0
                if remained_len == 0 or (idx == 0 and remained_len < len(kp)) or (idx > 0 and remained_len < len(kp) + 1):
                    is_truncated = True
                    break
                if idx >= 1:
                    line_out += ";"
                line_out += kp
            truncated_len = len(line_out) - init_out_len
            assert num_absent_mask >= truncated_len >= 0
            absent_truncated_len_dict[truncated_len] = \
                absent_truncated_len_dict[truncated_len] + 1 if truncated_len in absent_truncated_len_dict else 1
            absent_truncated_num += 1 if is_truncated else 0
            line_out += ["<T>"] * (num_absent_mask - (len(line_out) - init_out_len))
        line_out += ["[SEP]"]
        assert len(line_out) == before_content_len
        line_out += content

        # *.labelc files
        line_bio_center = ["N"] * before_content_len + ["O"] * (len(line_in) - before_content_len)
        for idx, se in enumerate(center_se_id):
            s = se[0] + before_content_len
            e = se[1] + before_content_len
            assert len(line_bio_center) == len(line_bio_center[:s] + ["B"] + ["I"] * (e - s) + line_bio_center[e + 1:])
            line_bio_center = line_bio_center[:s] + ["B"] + ["I"] * (e - s) + line_bio_center[e + 1:]

        # *.labelp files
        line_bio_present = ["N"] * before_content_len + ["O"] * (len(line_in) - before_content_len)
        for idx, se in enumerate(present_se_id):
            s = se[0] + before_content_len
            e = se[1] + before_content_len
            assert len(line_bio_present) == len(
                line_bio_present[:s] + ["B"] + ["I"] * (e - s) + line_bio_present[e + 1:])
            line_bio_present = line_bio_present[:s] + ["B"] + ["I"] * (e - s) + line_bio_present[e + 1:]

        if is_bix:
            for idx2, (ii, cc, pp) in enumerate(zip(line_in, line_bio_center, line_bio_present)):
                if "##" in ii:
                    line_bio_center[idx2] = "X" if cc == "I" else cc
                    line_bio_present[idx2] = "X" if pp == "I" else pp

        assert len(line_in) == len(line_out) and \
               len(line_in) == len(line_bio_present) and \
               len(line_in) == len(line_bio_center)
        if "" in line_in or "" in line_out:
            disabled_num += 1
            continue
        f_in.write(" ".join(line_in) + "\n")
        f_out.write(" ".join(line_out) + "\n")
        f_center.write(" ".join(line_bio_center) + "\n")
        f_present.write(" ".join(line_bio_present) + "\n")
        if is_test:
            f_center_word.write("\t".join([" ".join(c).replace(" ##", "") for c in line_center_word]) + "\n")
        if interval_nesting(center_se):
            center_nest += 1
        if interval_nesting(present_se):
            present_nest += 1
        if interval_nesting(center_se, present_se):
            cp_nest += 1
    total_doc = len(all_content)
    print(f"Center Nesting Num = {center_nest}({center_nest / total_doc * 100}%)")
    print(f"Present Nesting Num = {present_nest}({present_nest / total_doc * 100}%)")
    print(f"Center & Present Nesting Num = {cp_nest}({cp_nest / total_doc * 100}%)")
    print(f"BIO Weight For Center = {0 if center_len == 0 else (doc_len - center_len) / center_len}")
    print(f"BIO Weight For Present = {0 if present_len == 0 else (doc_len - present_len) / present_len}")
    print(f"BIO Weight For Center & Present = {0 if cp_len == 0 else (doc_len - cp_len) / cp_len}")
    print(f"Average Center Len = {center_len / total_doc}")
    print(f"Average Present Len = {present_len / total_doc}")
    print(f"Average Doc Len = {doc_len / total_doc}")
    print(
        f"Truncated Absolutely Absent Phrase Doc Num = {absent_truncated_num}({absent_truncated_num / total_doc * 100}%)")
    len_dict_output(absent_truncated_len_dict, split_percent,
                    sum([v for k, v in absent_truncated_len_dict.items()]), 100,
                    lambda ii, nn, total: print(f"Content Len = {ii} Num = {nn}({nn / total * 100}%)"),
                    lambda s: print(f"Average Content Len = {s}"),
                    lambda s, percent, all_num, idx:
                    print(f"Content {percent * 100}% Percent Split Len is {idx} ({s / all_num * 100}%)")
                    )
    print(f"Disabled Num = {disabled_num}")
    print(f"Total Center Word Num = {total_center}, "
          f"Total Filtered Center Num = {filtered_center}({filtered_center / (total_center + 1e-6) * 100}%)")
    print(f"Total Center Word Num = {total_center}, "
          f"Total LENGTH Filtered Center Num = {len_filtered_center}({len_filtered_center / (total_center + 1e-6) * 100}%)")
    print(f"MASK Truncated Doc Num = {mask_truncated_num}({mask_truncated_num / total_doc * 100}%)")


def bio_tagging_ps_cs(items, num_center_mask, num_absent_mask, max_mask_num, is_test, is_bix, is_stem, is_prompt, repeat_num,
                      split_percent, center_as, absent_as, cached_path, output_path):
    stemmer = PorterStemmer()
    all_content, all_keywords = item2token(items, cached_path=cached_path)
    all_content = [" ".join(c).replace(" ##", "").split(" ") for c in tqdm(all_content)]
    all_content_stem = [[stemmer.stem(cc) for cc in c] for c in tqdm(all_content)] if is_stem else []
    all_keywords = [[" ".join(k).replace(" ##", "").split(" ") for k in ks] for ks in tqdm(all_keywords)]
    all_keywords_stem = [[[stemmer.stem(kk) for kk in k]for k in ks] for ks in tqdm(all_keywords)] if is_stem else []
    is_nkw = len(all_keywords) == 1
    if len(all_keywords) == 1:
        all_keywords = [[""]] * len(all_content)
    assert len(all_content) == len(all_keywords)
    center_nest = 0
    present_nest = 0
    cp_nest = 0
    center_len = 0
    present_len = 0
    cp_len = 0
    doc_len = 0
    mask_truncated_num = 0
    absent_truncated_num = 0
    disabled_num = 0
    total_center = 0  # total number of keywords
    filtered_center = 0  # number of filtered keywords
    all_repeat = 0
    len_filtered_center = 0
    absent_truncated_len_dict = dict()
    assert len(center_as.split(WILDCARD_AS)) <= 2
    tokenization = Tokenization()
    center_as_pre, center_as_post = center_as.split(WILDCARD_AS)
    center_as_pre = tokenization.tokenize(center_as_pre)
    center_as_post = tokenization.tokenize(center_as_post)
    absent_as = tokenization.tokenize(absent_as)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    filename_in = os.path.join(output_path, "seq.in")
    filename_out = os.path.join(output_path, "seq.out")
    filename_present = os.path.join(output_path, "seq.labelp")
    filename_center = os.path.join(output_path, "seq.labelc")
    filename_center_word = os.path.join(output_path, "seq.center")
    f_in = open(filename_in, "w+", encoding="utf-8")
    f_out = open(filename_out, "w+", encoding="utf-8")
    f_present = open(filename_present, "w+", encoding="utf-8")
    f_center = open(filename_center, "w+", encoding="utf-8")
    f_center_word = open(filename_center_word, "w+", encoding="utf-8")

    """
        present_se：list([starting idx of the present, ending idx of the present, keywords(used by center_se)])
        center_se is the same as present_se
    """
    for i in tqdm(range(len(all_content))):
        content = all_content[i]
        content_stem = all_content_stem[i] if is_stem else []
        assert not is_stem or len(content) == len(content_stem)
        center_se = []
        present_se = []
        absolute_absent_kp = []
        for j in range(len(all_keywords[i])):
            keyword = all_keywords[i][j]
            keyword_stem = all_keywords_stem[i][j] if is_stem else []
            assert not is_stem or len(keyword) == len(keyword_stem)
            temp_content = content_stem if is_stem else content
            cur_len = 0
            final_start = -1
            final_end = -1
            while True:
                center, center_start, center_end = \
                    longest_common_sublist(temp_content, keyword_stem if is_stem else keyword)
                assert len(center) == center_end - center_start + 1 and len(center) <= len(keyword)
                temp_content = temp_content[center_end + 1:]
                if len(center) > cur_len and 0 < len(center) < len(keyword):
                    final_start = center_start
                    final_end = center_end
                    cur_len = len(center)
                elif len(center) > cur_len and len(center) == len(keyword):
                    final_start = center_start
                    final_end = center_end
                    cur_len = len(center)
                elif len(center) == 0:
                    break
            if final_end >= 0 and final_start >= 0:
                assert 0 <= final_end - final_start + 1 <= len(keyword)
                c = content[final_start:final_end + 1]
                if 0 < final_end - final_start + 1 < len(keyword):
                    loop = 0
                    while True:
                        if find_sublist(content[loop:], c) == -1:
                            break
                        loop = loop + find_sublist(content[loop:], c)
                        center_se.append([loop, loop + len(c) - 1, keyword])
                        loop = loop + len(c)
                elif final_end - final_start + 1 == len(keyword):
                    loop = 0
                    while True:
                        if find_sublist(content[loop:], c) == -1:
                            break
                        loop = loop + find_sublist(content[loop:], c)
                        present_se.append([loop, loop + len(c) - 1, keyword])
                        loop = loop + len(c)
            else:
                assert final_start == -1 and final_end == -1
                absolute_absent_kp.append(keyword)

        '''
            length statistics
        '''
        center_len += interval_len(center_se)
        present_len += interval_len(present_se)
        cp_len += interval_len(center_se + present_se)
        doc_len += len(content)

        '''
            get line_in、line_out、line_bio_center、line_bio_present by center_se & present_se
        '''

        center_se_id = interval_deconflict(center_se)
        present_se_id = interval_deconflict(present_se)

        # filter the incorrect keywords
        new_center_se_id = []
        for idx, se in enumerate(center_se_id):
            total_center += 1
            s = se[0]
            e = se[1]
            c = tokenization.tokenize(" ".join(content[s:e + 1]))
            if only_subword(c):
                continue
            c = " ".join(c).replace(" ##", "")  # subword to word (and list to string)
            c_len = len(c.replace(" ", ""))  # the length of keyword
            if only_punctuation(c) or c in set(stopwords("en")) or c_len <= 2:
                continue
            new_center_se_id.append(se)
        total_center += len(center_se_id)
        filtered_center += len(center_se_id) - len(new_center_se_id)
        assert len(new_center_se_id) <= len(center_se_id)
        center_se_id = new_center_se_id

        # filter absent keyphrases by length
        new_center_se_id = []
        for idx, se in enumerate(center_se_id):
            s = se[0]
            e = se[1]
            k = se[2]
            c = content[s:e + 1]
            k_stem = [stemmer.stem(t) for t in k] if is_stem else k
            c_stem = [stemmer.stem(t) for t in c] if is_stem else c
            c_new, s_new, e_new = longest_common_sublist(k_stem, c_stem)
            assert not is_stem or c_new == [stemmer.stem(cc) for cc in c]
            kc = tokenization.tokenize(" ".join(k[s_new:e_new + 1]))
            tk = tokenization.tokenize(" ".join(k))
            c_new, s_new, e_new = longest_common_sublist(tk, kc)
            left_len = s_new
            right_len = len(tk) - e_new - 1
            if left_len > num_center_mask or right_len > num_center_mask:
                len_filtered_center += 1
                continue
            new_center_se_id.append(se)
        assert len(new_center_se_id) <= len(center_se_id)
        center_se_id = new_center_se_id

        # dedup the keywords
        center_se_id_de = []
        center_dict = dict()
        for idx, se in enumerate(center_se_id):
            s = se[0]
            e = se[1]
            c = content[s:e + 1]
            center_dict[tuple(c)] = se
        for c, se in center_dict.items():
            center_se_id_de.append(se)

        # get the keywords
        line_center_word = []
        for idx, se in enumerate(center_se_id_de):
            s = se[0]
            e = se[1]
            c = content[s:e + 1]
            line_center_word.append(c)

        max_loop_count = (max_mask_num - num_absent_mask) // (num_center_mask * 2)

        # Permutation
        if not is_test:
            random.shuffle(center_se_id_de)
            center_se_id_de = center_se_id_de[0:min(len(center_se_id_de), 8)]
            center_se_id_de_list = list(itertools.permutations(center_se_id_de, len(center_se_id_de)))
            random.shuffle(center_se_id_de_list)
            center_se_id_de_list = center_se_id_de_list[0:min(repeat_num, len(center_se_id_de_list))]
        else:
            center_se_id_de_list = [center_se_id_de]
        all_repeat += len(center_se_id_de_list)

        for center_se_id_de in center_se_id_de_list:
            line_in = []
            line_out = []
            line_bio_center = []
            line_bio_present = []

            # .in file
            loop_count = 0
            for idx, se in enumerate(center_se_id_de):
                if not is_test and loop_count >= max_loop_count:
                    mask_truncated_num += 1
                    break
                s = se[0]
                e = se[1]
                k = se[2]
                c = content[s:e + 1]
                k_stem = [stemmer.stem(t) for t in k] if is_stem else k
                c_stem = [stemmer.stem(t) for t in c] if is_stem else c
                c_new, s_new, e_new = longest_common_sublist(k_stem, c_stem)
                kc = tokenization.tokenize(" ".join(k[s_new:e_new + 1]))
                assert not is_stem or c_new == [stemmer.stem(cc) for cc in c]
                if is_prompt:
                    line_in += center_as_pre + kc + center_as_post
                line_in += ["[MASK]"] * num_center_mask + kc + ["[MASK]"] * num_center_mask
                # if idx != len(center_se_id) - 1:
                line_in += ["<S>"]
                loop_count += 1
            # if len(line_in) > 0:
            #     line_in += ["<S>"]
            if not is_nkw:
                if is_prompt:
                    line_in += absent_as
                line_in += ["[MASK]"] * num_absent_mask
            line_in += ["[SEP]"]
            before_content_len = len(line_in)
            line_in += tokenization.tokenize(" ".join(content))

            # 生成.out文件
            random.shuffle(absolute_absent_kp)
            loop_count = 0
            for idx, se in enumerate(center_se_id_de):
                if not is_test and loop_count >= max_loop_count:
                    break
                s = se[0]
                e = se[1]
                k = se[2]
                c = content[s:e + 1]
                k_stem = [stemmer.stem(t) for t in k] if is_stem else k
                c_stem = [stemmer.stem(t) for t in c] if is_stem else c
                c_new, s_new, e_new = longest_common_sublist(k_stem, c_stem)
                assert not is_stem or c_new == [stemmer.stem(cc) for cc in c]
                kc = tokenization.tokenize(" ".join(k[s_new:e_new + 1]))
                tk = tokenization.tokenize(" ".join(k))
                c_new, s_new, e_new = longest_common_sublist(tk, kc)
                if is_prompt:
                    line_out += center_as_pre + kc + center_as_post
                line_out += ["<T>"] * max(0, num_center_mask - s_new)
                line_out += tokenization.tokenize(" ".join(
                    k[max(0, s_new - num_center_mask):min(len(tk), e_new + 1 + num_center_mask)]))
                line_out += ["<T>"] * max(0, num_center_mask - (len(tk) - e_new - 1))
                # if idx != len(center_se_id) - 1:
                line_out += ["<S>"]
                loop_count += 1
            # if len(line_out) > 0:
            #     line_out += ["<S>"]
            if not is_nkw:
                if is_prompt:
                    line_out += absent_as
                init_out_len = len(line_out)
                is_truncated = False
                for idx, kp in enumerate(absolute_absent_kp):
                    kp = tokenization.tokenize(" ".join(kp))
                    remained_len = init_out_len + num_absent_mask - len(line_out)
                    assert remained_len >= 0
                    if remained_len == 0 or (idx == 0 and remained_len < len(kp)) or (idx > 0 and remained_len < len(kp) + 1):
                        is_truncated = True
                        break
                    if idx >= 1:
                        line_out += ";"
                    line_out += kp
                truncated_len = len(line_out) - init_out_len
                assert num_absent_mask >= truncated_len >= 0
                absent_truncated_len_dict[truncated_len] = \
                    absent_truncated_len_dict[truncated_len] + 1 if truncated_len in absent_truncated_len_dict else 1
                absent_truncated_num += 1 if is_truncated else 0
                line_out += ["<T>"] * (num_absent_mask - (len(line_out) - init_out_len))
            line_out += ["[SEP]"]
            assert len(line_out) == before_content_len
            line_out += tokenization.tokenize(" ".join(content))

            # *.labelc files
            line_bio_center = ["N"] * before_content_len + ["O"] * (len(line_in) - before_content_len)
            for idx, se in enumerate(center_se_id):
                s = len(tokenization.tokenize(" ".join(content[0:se[0]]))) + before_content_len
                e = len(tokenization.tokenize(" ".join(content[0:se[1] + 1]))) - 1 + before_content_len
                assert len(line_bio_center) == len(line_bio_center[:s] + ["B"] + ["I"] * (e - s) + line_bio_center[e + 1:])
                line_bio_center = line_bio_center[:s] + ["B"] + ["I"] * (e - s) + line_bio_center[e + 1:]

            # *.labelp files
            line_bio_present = ["N"] * before_content_len + ["O"] * (len(line_in) - before_content_len)
            for idx, se in enumerate(present_se_id):
                s = len(tokenization.tokenize(" ".join(content[0:se[0]]))) + before_content_len
                e = len(tokenization.tokenize(" ".join(content[0:se[1] + 1]))) - 1 + before_content_len
                assert len(line_bio_present) == len(
                    line_bio_present[:s] + ["B"] + ["I"] * (e - s) + line_bio_present[e + 1:])
                line_bio_present = line_bio_present[:s] + ["B"] + ["I"] * (e - s) + line_bio_present[e + 1:]

            if is_bix:
                for idx2, (ii, cc, pp) in enumerate(zip(line_in, line_bio_center, line_bio_present)):
                    if "##" in ii:
                        line_bio_center[idx2] = "X" if cc == "I" else cc
                        line_bio_present[idx2] = "X" if pp == "I" else pp

            assert len(line_in) == len(line_out) and \
                   len(line_in) == len(line_bio_present) and \
                   len(line_in) == len(line_bio_center)
            if "" in line_in or "" in line_out:
                disabled_num += 1
                continue
            f_in.write(" ".join(line_in) + "\n")
            f_out.write(" ".join(line_out) + "\n")
            f_center.write(" ".join(line_bio_center) + "\n")
            f_present.write(" ".join(line_bio_present) + "\n")
            if is_test:
                f_center_word.write("\t".join([" ".join(c).replace(" ##", "") for c in line_center_word]) + "\n")
            if interval_nesting(center_se):
                center_nest += 1
            if interval_nesting(present_se):
                present_nest += 1
            if interval_nesting(center_se, present_se):
                cp_nest += 1
    total_doc = len(all_content)
    print(f"Center Nesting Num = {center_nest}({center_nest / total_doc * 100}%)")
    print(f"Present Nesting Num = {present_nest}({present_nest / total_doc * 100}%)")
    print(f"Center & Present Nesting Num = {cp_nest}({cp_nest / total_doc * 100}%)")
    print(f"BIO Weight For Center = {0 if center_len == 0 else (doc_len - center_len) / center_len}")
    print(f"BIO Weight For Present = {0 if present_len == 0 else (doc_len - present_len) / present_len}")
    print(f"BIO Weight For Center & Present = {0 if cp_len == 0 else (doc_len - cp_len) / cp_len}")
    print(f"Average Center Len = {center_len / total_doc}")
    print(f"Average Present Len = {present_len / total_doc}")
    print(f"Average Doc Len = {doc_len / total_doc}")
    print(
        f"Truncated Absolutely Absent Phrase Doc Num = {absent_truncated_num}({absent_truncated_num / total_doc * 100}%)")
    len_dict_output(absent_truncated_len_dict, split_percent,
                    sum([v for k, v in absent_truncated_len_dict.items()]), 100,
                    lambda ii, nn, total: print(f"Content Len = {ii} Num = {nn}({nn / total * 100}%)"),
                    lambda s: print(f"Average Content Len = {s}"),
                    lambda s, percent, all_num, idx:
                    print(f"Content {percent * 100}% Percent Split Len is {idx} ({s / all_num * 100}%)")
                    )
    print(f"Disabled Num = {disabled_num}")
    print(f"Total Center Word Num = {total_center}, "
          f"Total Filtered Center Num = {filtered_center}({filtered_center / (total_center + 1e-6) * 100}%)")
    print(f"Total Center Word Num = {total_center}, "
          f"Total LENGTH Filtered Center Num = {len_filtered_center}({len_filtered_center / (total_center + 1e-6) * 100}%)")
    print(f"MASK Truncated Doc Num = {mask_truncated_num}({mask_truncated_num / total_doc * 100}%)")
    print(f"Total Repeat Num = {all_repeat}({all_repeat/total_doc})")


