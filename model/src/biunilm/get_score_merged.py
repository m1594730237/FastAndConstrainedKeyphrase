import json
import os
import pickle
import sys

from nltk.stem.porter import PorterStemmer
from tqdm import tqdm

from in2test import in2test
from tokenization import BertTokenizer, WhitespaceTokenizer


def read_lines(filename):
    with open(filename, encoding='utf-8') as fp:
        lines = fp.read().rstrip().split('\n')
        return lines


def write_line(fp, line, split_str="\n"):
    fp.write(line + split_str)


def write_lines(filename, lines, mode, split_str="\n"):
    with open(filename, mode=mode, encoding="utf-8") as fp:
        for line in lines:
            write_line(fp, line, split_str)


def item2token(items, tokenizer, cached_path=""):
    all_content = list()
    all_keywords = list()
    if cached_path != "" and not os.path.exists(cached_path):
        os.mkdir(cached_path)
    cached_content = os.path.join(cached_path, "content")
    cached_keyword = os.path.join(cached_path, "keyword")
    cached_finished = os.path.join(cached_path, "finished")
    if os.path.exists(cached_finished) and os.path.exists(cached_content) and os.path.exists(cached_keyword):
        all_content = read_lines(cached_content)
        all_keywords = read_lines(cached_keyword)
        # if len(all_keywords) <= 1:
        #     all_keywords = [""] * len(all_content)
        for idx, line in tqdm(enumerate(all_content)):
            all_content[idx] = line.split("\t\t")
        for idx, line in tqdm(enumerate(all_keywords)):
            all_keywords[idx] = [k.split("  ") for k in line.split("\t\t")]
        pass
    else:
        with open(cached_content, "w+", encoding="utf-8") as fpc, open(cached_keyword, "w+", encoding="utf-8") as fpk:
            for idx, item in tqdm(enumerate(items)):
                temp_split = " . " if item["title"] != "" else " "
                content = tokenizer.tokenize((item["title"] + temp_split + item["abstract"]))
                keywords = item["keywords"]
                for idx2, keyword in enumerate(keywords):
                    keywords[idx2] = tokenizer.tokenize(keyword)
                fpc.write("\t\t".join(content) + "\n")
                fpk.write("\t\t".join(["  ".join(k) for k in keywords]) + "\n")
                all_content.append(content)
                all_keywords.append(keywords)
        with open(cached_finished, "w+") as _:
            pass
    # if len(all_keywords) <= 1:
    #     all_keywords = []
    return all_content, all_keywords


def deduplication_keep_order(input_list):
    return sorted(set(input_list), key=input_list.index)


def find_sublist(x, y):
    l1, l2 = len(x), len(y)
    for i in range(l1):
        if x[i:min(i + l2, l1)] == y:
            return i
    return -1


def longest_common_sublist(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]  # 生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax = 0  # 最长匹配的长度
    p = 0  # 最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], p - mmax, p - 1


def bio2phrases(in_lines, tagging_lines, tagging_prob_lines, top_n=1000000, entity_threshold=-100000,
                stop_char=("。", "，", "、", "?", ".", "：", "[UNK]", ",", "！", "；")):
    def res_check(r, sc):
        # if len(one_res) < 2:
        #     return False
        for c in sc:
            if c in r:
                return False
        return True

    res = list()
    for idx, (in_line, tagging_line, tagging_prob_line) in \
            enumerate(zip(in_lines, tagging_lines, tagging_prob_lines)):

        res_line = list()
        one_res = list()
        total_prob = 0
        has_b = False
        for idx2, (i, t, p) in enumerate(zip(in_line, tagging_line, tagging_prob_line)):
            t_last = "S" if idx2 == 0 else tagging_line[idx2 - 1]
            if t == "B":
                if len(one_res) != 0 and (t_last == "B" or t_last == "I"):
                    prob = total_prob / len(one_res)
                    if prob > entity_threshold and res_check(one_res, stop_char):
                        res_line.append([one_res, prob])
                one_res = list()
                total_prob = 0
                one_res.append(i)
                total_prob += float(p)
                has_b = True
            elif t == "I":
                if not has_b:
                    continue
                one_res.append(i)
                total_prob += float(p)
            elif t == "O" or t == "N" or t == "[CLS]" or t == "[SEP]":
                has_b = False
                if len(one_res) == 0 or t_last == "O" or t_last == "N" or t == "[CLS]" or t == "[SEP]":
                    continue
                prob = total_prob / len(one_res)
                if prob > entity_threshold and res_check(one_res, stop_char):
                    res_line.append([one_res, prob])
            else:
                print(t)
                assert False
        res_line.sort(key=lambda item: item[1], reverse=True)
        # res_line = [item[0] for item in res_line]
        # res_line = sorted(set(res_line), key=res_line.index)  # 保持原顺序去重
        res_line = res_line[0:top_n]
        res.append(res_line)
    return res


def f1_score(p, r):
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def sub2word(sub_word_list):
    res = []
    for ws in sub_word_list:
        res.append(["\t".join(w).replace("\t##", "").split("\t") for w in ws])
    assert len(res) == len(sub_word_list)
    return res


def word2stem(word_list):
    res = []
    stemmer = PorterStemmer()
    for ws in word_list:
        res.append([[stemmer.stem(s) for s in w] for w in ws])
    assert len(res) == len(word_list)
    return res


def get_pr_kws(prediction, kws, top_n, complete):
    assert len(prediction) == len(kws)
    prediction = [[pp[0] for pp in sorted(p, key=lambda t: t[1], reverse=True)] for p in prediction]
    prediction = [deduplication_keep_order(p)[0:min(top_n, len(deduplication_keep_order(p)))] for p in prediction]
    kws = [deduplication_keep_order(k) for k in kws]
    prediction = word2stem(sub2word(prediction))
    kws = word2stem(sub2word(kws))

    p = 0
    total_p = 0
    r = 0
    total_r = 0
    for i in tqdm(range(len(prediction))):
        # kw = set([tuple(item) for item in kws[i]])
        kw = set(["".join(item) for item in kws[i]])
        # if len(kw) == 0:
        #     continue
        prediction_line = prediction[i]
        # prediction_line.sort(key=lambda item: item[1], reverse=True)
        # prediction_line = prediction_line[0:min(top_n, len(prediction_line))]
        # prediction_line = set([tuple(item) for item in prediction_line])
        prediction_line = set(["".join(item) for item in prediction_line])
        # print("***********************")
        # print(kw)
        # print(prediction_line)
        for pre in prediction_line:
            if pre in kw:
                p += 1
            total_p += 1
        total_p += max(top_n - len(prediction_line), 0) if complete else 0
        for k in kw:
            if k in prediction_line:
                r += 1
            total_r += 1
    if total_p == 0:
        assert p == 0
    if total_r == 0:
        assert r == 0
    return p / (total_p + 1e-6), r / (total_r + 1e-6)


def get_pr_doc(prediction, kws, top_n):
    assert len(prediction) == len(kws)
    prediction = [[pp[0] for pp in sorted(p, key=lambda t: t[1], reverse=True)] for p in prediction]
    prediction = [deduplication_keep_order(p)[0:min(top_n, len(deduplication_keep_order(p)))] for p in prediction]
    kws = [deduplication_keep_order(k) for k in kws]
    prediction = word2stem(sub2word(prediction))
    kws = word2stem(sub2word(kws))

    total_p = 0
    total_r = 0
    total_n = 0
    for i in tqdm(range(len(prediction))):
        kw = set([tuple(item) for item in kws[i]])
        if len(kw) == 0:
            continue
        prediction_line = prediction[i]
        # prediction_line.sort(key=lambda item: item[1], reverse=True)
        # prediction_line = prediction_line[0:min(top_n, len(prediction_line))]
        prediction_line = set([tuple(item) for item in prediction_line])
        p = 0
        r = 0
        for pre in prediction_line:
            if pre in kw:
                p += 1
        for k in kw:
            if k in prediction_line:
                r += 1
        p = p / len(prediction_line) if len(prediction_line) > 0 else 0
        r = r / len(kw)
        assert p <= 1 and r <= 1
        total_p += p
        total_r += r
        total_n += 1
    return total_p / total_n, total_r / total_n


def get_pr(prediction, kws, top_n, complete):
    return get_pr_kws(prediction, kws, top_n, complete)


def remove_mask(line, mask_num, total_mask_num):
    loop = 0
    line = line.copy()
    while loop < len(line):
        if line[loop] == "[MASK]" and find_sublist(line[:loop + 1], ["other", "phrases", "are"]) == -1:
            line = line[:loop + mask_num] + line[loop + total_mask_num:]
            loop += mask_num - 1
        loop += 1
    return line


def postprocess_test_result(line, mask_num, max_id):
    loop = 0
    max_id = min(max_id, len(line))
    while loop < max_id:
        cen_left_id = loop + mask_num - 1
        cen_right_id = mask_num
        if cen_left_id < max_id and loop + 2 * mask_num - 1 < max_id:
            if line[cen_left_id] == "[unused101]":
                for i in range(loop, cen_left_id):
                    line[i] = "[unused101]"
            else:
                for i in range(loop, cen_left_id + 1):
                    if i - 1 >= loop and line[i] != "[unused101]" and line[i - 1] == "[unused101]":
                        if "##" in line[i]:
                            line[i] = "[unused101]"
                            i = i + 1
        if cen_right_id < max_id and cen_right_id + mask_num - 1 < max_id:
            if line[cen_right_id] == "[unused101]":
                for i in range(cen_right_id + 1, cen_right_id + mask_num):
                    line[i] = "[unused101]"
        loop += 2 * mask_num
    return line


def three_type_kws(contents, kws):
    absent = []
    present = []
    absent_ab = []
    stemmer = PorterStemmer()
    assert len(contents) == len(kws)
    for i in tqdm(range(len(kws))):
        c = contents[i]
        kw = kws[i]
        absent_line = []
        absent_ab_line = []
        present_line = []
        c_word = "\t".join(c).replace("\t##", "").split("\t")
        c_stem = [stemmer.stem(s) for s in c_word]
        for k in kw:
            k_word = "\t".join(k).replace("\t##", "").split("\t")
            k_stem = [stemmer.stem(s) for s in k_word]
            if find_sublist(c_stem, k_stem) != -1:
                present_line.append(k)
            # elif len(longest_common_sublist(c_stem, k_stem)[0]) == 0:
            #    absent_ab_line.append(k)
            else:
                absent_line.append(k)
        absent.append(absent_line)
        absent_ab.append(absent_ab_line)
        present.append(present_line)
    return present, absent, absent_ab


def load_dataset(filename):
    lines = read_lines(filename)
    res = list()
    for line in tqdm(lines):
        j = json.loads(line)
        res.append(j)
    return res


def get_score(path, test_tagging_p_lines, test_tagging_p_prob_lines,
              test_result_lines, test_result_prob_lines, absent_type, mask_num=3, total_mask_num=3):
    """
        TAGGING
    """
    bert_model = "/root_path/PTModel/unilm_pytorch/bert-base-cased"
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
    items = load_dataset(os.path.join(path, "data.json"))
    contents, kws = item2token(items, tokenizer, cached_path=os.path.join(path, "cache"))
    seq_in_lines = read_lines(os.path.join(path, "seq.in"))
    test_tagging_p_lines = test_tagging_p_lines.copy()
    test_tagging_p_prob_lines = test_tagging_p_prob_lines.copy()
    assert len(seq_in_lines) == len(test_tagging_p_lines)
    assert len(seq_in_lines) == len(test_tagging_p_prob_lines)

    kws_present, kws_absent, kws_absent_ab = three_type_kws(contents, kws)
    kws_present = [[tuple(pp) for pp in p] for p in kws_present]
    kws_absent = [[tuple(pp) for pp in p] for p in kws_absent]
    kws_absent_ab = [[tuple(pp) for pp in p] for p in kws_absent_ab]
    kws_absent_all = [p1 + p2 for p1, p2 in zip(kws_absent, kws_absent_ab)]

    for idx, (tagging_line, tagging_prob_line) in enumerate(zip(test_tagging_p_lines, test_tagging_p_prob_lines)):
        tagging_line = tagging_line.replace("X", "I").split(" ")
        # tagging_prob_line = tagging_prob_line.split(" ")
        tagging_prob_line = [max(item) for item in tagging_prob_line]
        tagging_line = tagging_line[1:-1]
        tagging_prob_line = tagging_prob_line[1:-1]
        test_tagging_p_lines[idx] = tagging_line
        test_tagging_p_prob_lines[idx] = tagging_prob_line

    prediction_present = bio2phrases(contents, test_tagging_p_lines, test_tagging_p_prob_lines)
    prediction_present = [[[tuple(pp[0]), pp[1]] for pp in p] for p in prediction_present]

    """
        MLM
    """
    bert_model = "/root_path/PTModel/unilm_pytorch/bert-base-cased"
    # bert_model = os.path.join("..", "bert-base-cased")
    seq_in_lines = in2test(read_lines(os.path.join(path, "seq.in")), mask_num) \
        if absent_type == "GC" else read_lines(os.path.join(path, "mlm_input", "seq.in"))
    # print(read_lines(os.path.join(path, "seq.in"))[0])
    # print(seq_in_lines[0])
    test_result_lines = test_result_lines.copy()
    test_result_prob_lines = test_result_prob_lines.copy()
    assert len(seq_in_lines) == len(test_result_lines)
    assert len(seq_in_lines) == len(test_result_prob_lines)

    prediction_absent = []
    prediction_absent_ab = []
    for i in tqdm(range(len(seq_in_lines))):
        prediction_line = []
        prediction_line_ab = []
        phrase = []
        sum_prob = 0
        test_result_loop = 0
        test_prob_loop = 0
        seq_in_line = seq_in_lines[i].split(" ")
        seq_in_line = remove_mask(seq_in_line, mask_num, total_mask_num)
        max_id = seq_in_line.index("other")
        start_center = False
        has_words = False
        test_result_line = test_result_lines[i].split(" ")
        test_result_line = postprocess_test_result(test_result_line, mask_num, max_id)
        for idx, t in enumerate(seq_in_line):
            if test_result_loop >= len(test_result_line):
                break
            # test_result_prob_line = test_result_prob_lines[i].split(" ")
            test_result_prob_line = test_result_prob_lines[i]
            if t == "[MASK]":
                if test_result_line[test_result_loop] != "[unused101]":
                    phrase.append(test_result_line[test_result_loop])
                    has_words = True
                sum_prob += float(test_result_prob_line[test_prob_loop])
                test_result_loop += 1
                test_prob_loop += 1
            if idx >= 1 and seq_in_line[idx - 1] == "[MASK]" and \
                    seq_in_line[idx] != "[MASK]" and seq_in_line[idx] != "[SEP]" and seq_in_line[idx] != "<S>":
                start_center = True
            if idx >= 1 and start_center and seq_in_line[idx - 1] != "[MASK]" and seq_in_line[idx] == "[MASK]":
                start_center = False
            if t == "<S>":
                if ";" in phrase:
                    phrase.remove(";")
                if len(phrase) > 0 and has_words:
                    prediction_line.append([phrase.copy(), sum_prob / len(phrase)])
                phrase.clear()
                sum_prob = 0
                start_center = False
                has_words = False
            if t == ";" or t == "[SEP]":
                if ";" in phrase:
                    phrase.remove(";")
                if len(phrase) > 0 and has_words:
                    prediction_line_ab.append([phrase.copy(), sum_prob / len(phrase)])
                phrase.clear()
                sum_prob = 0
                start_center = False
                has_words = False
            if t == "[SEP]":
                break
            if start_center:
                phrase.append(t)
        prediction_line.sort(key=lambda item: item[1], reverse=True)
        prediction_line_ab.sort(key=lambda item: item[1], reverse=True)
        prediction_absent.append(prediction_line)
        prediction_absent_ab.append(prediction_line_ab)

    # print(prediction_absent[0])

    assert len(prediction_absent) == len(seq_in_lines)
    assert len(prediction_absent_ab) == len(seq_in_lines)
    prediction_absent = [[[tuple(pp[0]), pp[1]] for pp in p] for p in prediction_absent]
    prediction_absent_ab = [[[tuple(pp[0]), pp[1]] for pp in p] for p in prediction_absent_ab]
    prediction_absent_all = [p1 + p2 for p1, p2 in zip(prediction_absent, prediction_absent_ab)]

    """
        KEYWORDS
    """

    prediction_all = [p + a for p, a in zip(prediction_present, prediction_absent_all)]
    assert len(prediction_all) == len(prediction_present) and len(prediction_all) == len(prediction_absent_all)
    for idx2, items in enumerate(prediction_all):
        prediction_present[idx2] = []
        prediction_absent_all[idx2] = []
        for item in items:
            if find_sublist(contents[idx2], list(item[0])) != -1:
                prediction_present[idx2].append(item)
            else:
                prediction_absent_all[idx2].append(item)

    """
        OUTPUT
    """
    if not os.path.exists(os.path.join(path, "output")):
        os.makedirs(os.path.join(path, "output"))
    write_lines(os.path.join(path, "output", "present.out"),
                [str([[" ".join(pp[0]).replace(" ##", "").split(" "), pp[1]] for pp in p]) for p in prediction_present],
                "w+")
    write_lines(os.path.join(path, "output", "present.golden"), [str(ii) for ii in kws_present], "w+")

    write_lines(os.path.join(path, "output", "present.out.readable"),
                ["\t".join([" ".join(pp[0]).replace(" ##", "") for pp in p]) for p in prediction_present],
                "w+")
    write_lines(os.path.join(path, "output", "present.golden.readable"),
                ["\t".join([" ".join(pp).replace(" ##", "") for pp in p]) for p in kws_present], "w+")

    assert len(prediction_present) == len(seq_in_lines)
    prediction_present = [[[tuple(pp[0]), pp[1]] for pp in p] for p in prediction_present]
    kws_present = [[tuple(pp) for pp in p] for p in kws_present]

    if not os.path.exists(os.path.join(path, "output")):
        os.makedirs(os.path.join(path, "output"))
    # write_lines(os.path.join(path, "output", "absent_nab.out"),
    #             [str([[" ".join(pp[0]).replace(" ##", "").split(" "), pp[1]] for pp in p]) for p in prediction_absent],
    #             "w+")
    # write_lines(os.path.join(path, "output", "absent_ab.out"),
    #             [str([[" ".join(pp[0]).replace(" ##", "").split(" "), pp[1]] for pp in p]) for p in prediction_absent_ab],
    #             "w+")
    write_lines(os.path.join(path, "output", "absent_all.out"),
                [str([[" ".join(pp[0]).replace(" ##", "").split(" "), pp[1]] for pp in p]) for p in prediction_absent_all],
                "w+")
    # write_lines(os.path.join(path, "output", "absent_nab.out.readable"),
    #             ["\t".join([" ".join(pp[0]).replace(" ##", "") for pp in p]) for p in prediction_absent],
    #             "w+")
    # write_lines(os.path.join(path, "output", "absent_ab.out.readable"),
    #             ["\t".join([" ".join(pp[0]).replace(" ##", "") for pp in p]) for p in prediction_absent_ab],
    #             "w+")
    write_lines(os.path.join(path, "output", "absent_all.out.readable"),
                ["\t".join([" ".join(pp[0]).replace(" ##", "") for pp in p]) for p in prediction_absent_all],
                "w+")

    # write_lines(os.path.join(path, "output", "absent_nab.golden"), [str(ii) for ii in kws_absent], "w+")
    # write_lines(os.path.join(path, "output", "absent_ab.golden"), [str(ii) for ii in kws_absent_ab], "w+")
    write_lines(os.path.join(path, "output", "absent_all.golden"), [str(ii) for ii in kws_absent_all], "w+")
    # write_lines(os.path.join(path, "output", "absent_nab.golden.readable"),
    #             ["\t".join([" ".join(pp).replace(" ##", "") for pp in p]) for p in kws_absent], "w+")
    # write_lines(os.path.join(path, "output", "absent_ab.golden.readable"),
    #             ["\t".join([" ".join(pp).replace(" ##", "") for pp in p]) for p in kws_absent_ab], "w+")
    write_lines(os.path.join(path, "output", "absent_all.golden.readable"),
                ["\t".join([" ".join(pp).replace(" ##", "") for pp in p]) for p in kws_absent_all], "w+")

    """
        CALCULATING
    """

    p_5_p, r_5_p = get_pr(prediction_present, kws_present, top_n=5, complete=True)
    p_all_p, r_all_p = get_pr(prediction_present, kws_present, top_n=10000000, complete=False)
    print("********** present **********")
    print(f"{str(p_5_p)[:5]}  {str(r_5_p)[:5]}  {str(f1_score(p_5_p, r_5_p))[:5]} \t"
          f"{str(p_all_p)[:5]}  {str(r_all_p)[:5]}  {str(f1_score(p_all_p, r_all_p))[:5]}\t")

    # p_5_a, r_5_a = get_pr(prediction_absent, kws_absent, top_n=5, complete=True)
    # p_all_a, r_all_a = get_pr(prediction_absent, kws_absent, top_n=10000000, complete=False)
    # p_5_ab, r_5_ab = get_pr(prediction_absent_ab, kws_absent_ab, top_n=5, complete=True)
    # p_all_ab, r_all_ab = get_pr(prediction_absent_ab, kws_absent_ab, top_n=10000000, complete=False)
    p_5_a_all, r_5_a_all = get_pr(prediction_absent_all, kws_absent_all, top_n=5, complete=True)
    p_all_a_all, r_all_a_all = get_pr(prediction_absent_all, kws_absent_all, top_n=10000000, complete=False)
    print(
        "********** absent all **********\n"
        f"{str(p_5_a_all)[:5]}  {str(r_5_a_all)[:5]}  {str(f1_score(p_5_a_all, r_5_a_all))[:5]}\t"
        f"{str(p_all_a_all)[:5]}  {str(r_all_a_all)[:5]}  {str(f1_score(p_all_a_all, r_all_a_all))[:5]}\t")
        # print("********** absent absolutely **********")
        # f"{str(p_5_ab)[:5]}  {str(r_5_ab)[:5]}  {str(f1_score(p_5_ab, r_5_ab))[:5]}\t"
        # f"{str(p_all_ab)[:5]}  {str(r_all_ab)[:5]}  {str(f1_score(p_all_ab, r_all_ab))[:5]}\t"
        # # print("********** absent not absolutely **********")
        # f"{str(p_5_a)[:5]}  {str(r_5_a)[:5]}  {str(f1_score(p_5_a, r_5_a))[:5]}\t"
        # f"{str(p_all_a)[:5]}  {str(r_all_a)[:5]}  {str(f1_score(p_all_a, r_all_a))[:5]}\t")

    present_all = [[" ".join(pp[0]).replace(" ##", "") for pp in p] for p in prediction_present]
    present_stem = word2stem([[" ".join(pp[0]).replace(" ##", "").split(" ") for pp in p] for p in prediction_present])

    absent_all = [[" ".join(pp[0]).replace(" ##", "") for pp in p] for p in prediction_absent_all]
    absent_stem = word2stem([[" ".join(pp[0]).replace(" ##", "").split(" ") for pp in p] for p in prediction_absent])
    return sum([len(set([tuple(pp[0]) for pp in p])) for p in present_stem]) / len(present_stem), present_all,\
           sum([len(set([tuple(pp[0]) for pp in p])) for p in absent_stem]) / len(prediction_absent_all), absent_all


# def get_score_present(path, test_tagging_p_lines, test_tagging_p_prob_lines):
#     bert_model = "/root_path/PTModel/unilm_pytorch/bert-base-cased"
#     tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
#     items = load_dataset(os.path.join(path, "data.json"))
#     contents, kws = item2token(items, tokenizer, cached_path=os.path.join(path, "cache"))
#     seq_in_lines = read_lines(os.path.join(path, "seq.in"))
#     test_tagging_p_lines = test_tagging_p_lines.copy()
#     test_tagging_p_prob_lines = test_tagging_p_prob_lines.copy()
#     assert len(seq_in_lines) == len(test_tagging_p_lines)
#     assert len(seq_in_lines) == len(test_tagging_p_prob_lines)
#
#     kws_present, kws_absent, kws_absent_ab = three_type_kws(contents, kws)
#
#     for idx, (tagging_line, tagging_prob_line) in enumerate(zip(test_tagging_p_lines, test_tagging_p_prob_lines)):
#         tagging_line = tagging_line.replace("X", "I").split(" ")
#         # tagging_prob_line = tagging_prob_line.split(" ")
#         tagging_prob_line = [max(item) for item in tagging_prob_line]
#         tagging_line = tagging_line[1:-1]
#         tagging_prob_line = tagging_prob_line[1:-1]
#         test_tagging_p_lines[idx] = tagging_line
#         test_tagging_p_prob_lines[idx] = tagging_prob_line
#
#     prediction_present = bio2phrases(contents, test_tagging_p_lines, test_tagging_p_prob_lines)
#
#     # print(prediction_present[0])
#     if not os.path.exists(os.path.join(path, "output")):
#         os.makedirs(os.path.join(path, "output"))
#     write_lines(os.path.join(path, "output", "present.out"),
#                 [str([[" ".join(pp[0]).replace(" ##", "").split(" "), pp[1]] for pp in p]) for p in prediction_present],
#                 "w+")
#     write_lines(os.path.join(path, "output", "present.golden"), [str(ii) for ii in kws_present], "w+")
#
#     write_lines(os.path.join(path, "output", "present.out.readable"),
#                 ["\t".join([" ".join(pp[0]).replace(" ##", "") for pp in p]) for p in prediction_present],
#                 "w+")
#     write_lines(os.path.join(path, "output", "present.golden.readable"),
#                 ["\t".join([" ".join(pp).replace(" ##", "") for pp in p]) for p in kws_present], "w+")
#
#     assert len(prediction_present) == len(seq_in_lines)
#     prediction_present = [[[tuple(pp[0]), pp[1]] for pp in p] for p in prediction_present]
#     kws_present = [[tuple(pp) for pp in p] for p in kws_present]
#
#     p_5_p, r_5_p = get_pr(prediction_present, kws_present, top_n=5, complete=True)
#     p_all_p, r_all_p = get_pr(prediction_present, kws_present, top_n=10000000, complete=False)
#     print(f"{str(p_5_p)[:5]}  {str(r_5_p)[:5]}  {str(f1_score(p_5_p, r_5_p))[:5]} \t"
#           f"{str(p_all_p)[:5]}  {str(r_all_p)[:5]}  {str(f1_score(p_all_p, r_all_p))[:5]}\t")
#
#     present_all = [[" ".join(pp[0]).replace(" ##", "") for pp in p] for p in prediction_present]
#     present_stem = word2stem([[" ".join(pp[0]).replace(" ##", "").split(" ") for pp in p] for p in prediction_present])
#     return sum([len(set([tuple(pp[0]) for pp in p])) for p in present_stem])/len(present_stem), present_all
#

def get_score_center(path, test_tagging_c_lines, test_tagging_c_prob_lines, mask_num=3, total_mask_num=3):
    bert_model = "/root_path/PTModel/unilm_pytorch/bert-base-cased"
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
    items = load_dataset(os.path.join(path, "data.json"))
    contents, kws = item2token(items, tokenizer, cached_path=os.path.join(path, "cache"))
    kws = read_lines(os.path.join(path, "seq.center"))
    kws = [kwl.split("\t") for kwl in kws]
    kws = [[tokenizer.tokenize(kw) for kw in kwl] for kwl in kws]
    for i in range(len(contents) - len(kws)):
        kws.append([])
    seq_in_lines = read_lines(os.path.join(path, "seq.in"))
    test_tagging_c_lines = test_tagging_c_lines.copy()
    test_tagging_c_prob_lines = test_tagging_c_prob_lines.copy()
    assert len(seq_in_lines) == len(test_tagging_c_lines)
    assert len(seq_in_lines) == len(test_tagging_c_prob_lines)

    kws_present, kws_absent, kws_absent_ab = three_type_kws(contents, kws)

    for idx, (tagging_line, tagging_prob_line) in enumerate(zip(test_tagging_c_lines, test_tagging_c_prob_lines)):
        tagging_line = tagging_line.replace("X", "I").split(" ")
        # tagging_prob_line = tagging_prob_line.split(" ")
        tagging_prob_line = [max(item) for item in tagging_prob_line]
        tagging_line = tagging_line[1:-1]
        tagging_prob_line = tagging_prob_line[1:-1]
        test_tagging_c_lines[idx] = tagging_line
        test_tagging_c_prob_lines[idx] = tagging_prob_line

    prediction_present = bio2phrases(contents, test_tagging_c_lines, test_tagging_c_prob_lines)

    # print(prediction_present[0])
    if not os.path.exists(os.path.join(path, "output")):
        os.makedirs(os.path.join(path, "output"))
    write_lines(os.path.join(path, "output", "center.out"),
                [str([[" ".join(pp[0]).replace(" ##", "").split(" "), pp[1]] for pp in p]) for p in prediction_present],
                "w+")
    write_lines(os.path.join(path, "output", "center.out.readable"),
                ["\t".join([" ".join(pp[0]).replace(" ##", "") for pp in p]) for p in prediction_present],
                "w+")
    write_lines(os.path.join(path, "output", "center.golden"), [str(ii) for ii in kws_present], "w+")
    write_lines(os.path.join(path, "output", "center.golden.readable"),
                ["\t".join([" ".join(pp).replace(" ##", "") for pp in p]) for p in kws_present], "w+")

    assert len(prediction_present) == len(seq_in_lines)
    prediction_present = [[[tuple(pp[0]), pp[1]] for pp in p] for p in prediction_present]
    kws_present = [[tuple(pp) for pp in p] for p in kws_present]
    p_5_p, r_5_p = get_pr(prediction_present, kws_present, top_n=5, complete=False)
    p_all_p, r_all_p = get_pr(prediction_present, kws_present, top_n=10000000, complete=False)

    print("********** center **********")
    print(f"{str(p_5_p)[:5]}  {str(r_5_p)[:5]}  {str(f1_score(p_5_p, r_5_p))[:5]} \t"
          f"{str(p_all_p)[:5]}  {str(r_all_p)[:5]}  {str(f1_score(p_all_p, r_all_p))[:5]}\t")

    present_stem = word2stem([[" ".join(pp[0]).replace(" ##", "").split(" ") for pp in p] for p in prediction_present])
    return sum([len(set([tuple(pp[0]) for pp in p]))
                for p in present_stem])/len(present_stem)


# def get_score_absent(path, test_result_lines, test_result_prob_lines, absent_type, mask_num=3, total_mask_num=3):
#     bert_model = "/root_path/PTModel/unilm_pytorch/bert-base-cased"
#     # bert_model = os.path.join("..", "bert-base-cased")
#     tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
#     items = load_dataset(os.path.join(path, "data.json"))
#     contents, kws = item2token(items, tokenizer, cached_path=os.path.join(path, "cache"))
#     seq_in_lines = in2test(read_lines(os.path.join(path, "seq.in")), mask_num) \
#         if absent_type == "GC" else read_lines(os.path.join(path, "mlm_input", "seq.in"))
#     # print(read_lines(os.path.join(path, "seq.in"))[0])
#     # print(seq_in_lines[0])
#     test_result_lines = test_result_lines.copy()
#     test_result_prob_lines = test_result_prob_lines.copy()
#     assert len(seq_in_lines) == len(test_result_lines)
#     assert len(seq_in_lines) == len(test_result_prob_lines)
#
#     kws_present, kws_absent, kws_absent_ab = three_type_kws(contents, kws)
#
#     prediction_absent = []
#     prediction_absent_ab = []
#     for i in tqdm(range(len(seq_in_lines))):
#         prediction_line = []
#         prediction_line_ab = []
#         phrase = []
#         sum_prob = 0
#         test_result_loop = 0
#         test_prob_loop = 0
#         seq_in_line = seq_in_lines[i].split(" ")
#         seq_in_line = remove_mask(seq_in_line, mask_num, total_mask_num)
#         max_id = seq_in_line.index("other")
#         start_center = False
#         has_words = False
#         test_result_line = test_result_lines[i].split(" ")
#         test_result_line = postprocess_test_result(test_result_line, mask_num, max_id)
#         for idx, t in enumerate(seq_in_line):
#             if test_result_loop >= len(test_result_line):
#                 break
#             # test_result_prob_line = test_result_prob_lines[i].split(" ")
#             test_result_prob_line = test_result_prob_lines[i]
#             if t == "[MASK]":
#                 if test_result_line[test_result_loop] != "[unused101]":
#                     phrase.append(test_result_line[test_result_loop])
#                     has_words = True
#                 sum_prob += float(test_result_prob_line[test_prob_loop])
#                 test_result_loop += 1
#                 test_prob_loop += 1
#             if idx >= 1 and seq_in_line[idx - 1] == "[MASK]" and \
#                     seq_in_line[idx] != "[MASK]" and seq_in_line[idx] != "[SEP]" and seq_in_line[idx] != "<S>":
#                 start_center = True
#             if idx >= 1 and start_center and seq_in_line[idx - 1] != "[MASK]" and seq_in_line[idx] == "[MASK]":
#                 start_center = False
#             if t == "<S>":
#                 if ";" in phrase:
#                     phrase.remove(";")
#                 if len(phrase) > 0 and has_words:
#                     prediction_line.append([phrase.copy(), sum_prob / len(phrase)])
#                 phrase.clear()
#                 sum_prob = 0
#                 start_center = False
#                 has_words = False
#             if t == ";" or t == "[SEP]":
#                 if ";" in phrase:
#                     phrase.remove(";")
#                 if len(phrase) > 0 and has_words:
#                     prediction_line_ab.append([phrase.copy(), sum_prob / len(phrase)])
#                 phrase.clear()
#                 sum_prob = 0
#                 start_center = False
#                 has_words = False
#             if t == "[SEP]":
#                 break
#             if start_center:
#                 phrase.append(t)
#         prediction_line.sort(key=lambda item: item[1], reverse=True)
#         prediction_line_ab.sort(key=lambda item: item[1], reverse=True)
#         prediction_absent.append(prediction_line)
#         prediction_absent_ab.append(prediction_line_ab)
#
#     # print(prediction_absent[0])
#
#     assert len(prediction_absent) == len(seq_in_lines)
#     assert len(prediction_absent_ab) == len(seq_in_lines)
#     prediction_absent = [[[tuple(pp[0]), pp[1]] for pp in p] for p in prediction_absent]
#     prediction_absent_ab = [[[tuple(pp[0]), pp[1]] for pp in p] for p in prediction_absent_ab]
#     prediction_absent_all = [p1 + p2 for p1, p2 in zip(prediction_absent, prediction_absent_ab)]
#     kws_absent = [[tuple(pp) for pp in p] for p in kws_absent]
#     kws_absent_ab = [[tuple(pp) for pp in p] for p in kws_absent_ab]
#     kws_absent_all = [p1 + p2 for p1, p2 in zip(kws_absent, kws_absent_ab)]
#
#     if not os.path.exists(os.path.join(path, "output")):
#         os.makedirs(os.path.join(path, "output"))
#     write_lines(os.path.join(path, "output", "absent_nab.out"),
#                 [str([[" ".join(pp[0]).replace(" ##", "").split(" "), pp[1]] for pp in p]) for p in prediction_absent],
#                 "w+")
#     write_lines(os.path.join(path, "output", "absent_ab.out"),
#                 [str([[" ".join(pp[0]).replace(" ##", "").split(" "), pp[1]] for pp in p]) for p in prediction_absent_ab],
#                 "w+")
#     write_lines(os.path.join(path, "output", "absent_all.out"),
#                 [str([[" ".join(pp[0]).replace(" ##", "").split(" "), pp[1]] for pp in p]) for p in prediction_absent_all],
#                 "w+")
#     write_lines(os.path.join(path, "output", "absent_nab.out.readable"),
#                 ["\t".join([" ".join(pp[0]).replace(" ##", "") for pp in p]) for p in prediction_absent],
#                 "w+")
#     write_lines(os.path.join(path, "output", "absent_ab.out.readable"),
#                 ["\t".join([" ".join(pp[0]).replace(" ##", "") for pp in p]) for p in prediction_absent_ab],
#                 "w+")
#     write_lines(os.path.join(path, "output", "absent_all.out.readable"),
#                 ["\t".join([" ".join(pp[0]).replace(" ##", "") for pp in p]) for p in prediction_absent_all],
#                 "w+")
#
#     write_lines(os.path.join(path, "output", "absent_nab.golden"), [str(ii) for ii in kws_absent], "w+")
#     write_lines(os.path.join(path, "output", "absent_ab.golden"), [str(ii) for ii in kws_absent_ab], "w+")
#     write_lines(os.path.join(path, "output", "absent_all.golden"), [str(ii) for ii in kws_absent_all], "w+")
#     write_lines(os.path.join(path, "output", "absent_nab.golden.readable"),
#                 ["\t".join([" ".join(pp).replace(" ##", "") for pp in p]) for p in kws_absent], "w+")
#     write_lines(os.path.join(path, "output", "absent_ab.golden.readable"),
#                 ["\t".join([" ".join(pp).replace(" ##", "") for pp in p]) for p in kws_absent_ab], "w+")
#     write_lines(os.path.join(path, "output", "absent_all.golden.readable"),
#                 ["\t".join([" ".join(pp).replace(" ##", "") for pp in p]) for p in kws_absent_all], "w+")
#
#     p_5_a, r_5_a = get_pr(prediction_absent, kws_absent, top_n=5, complete=True)
#     p_all_a, r_all_a = get_pr(prediction_absent, kws_absent, top_n=10000000, complete=False)
#     p_5_ab, r_5_ab = get_pr(prediction_absent_ab, kws_absent_ab, top_n=5, complete=True)
#     p_all_ab, r_all_ab = get_pr(prediction_absent_ab, kws_absent_ab, top_n=10000000, complete=False)
#     p_5_a_all, r_5_a_all = get_pr(prediction_absent_all, kws_absent_all, top_n=5, complete=True)
#     p_all_a_all, r_all_a_all = get_pr(prediction_absent_all, kws_absent_all, top_n=10000000, complete=False)
#     print(
#         # print("********** absent all **********")
#         f"{str(p_5_a_all)[:5]}  {str(r_5_a_all)[:5]}  {str(f1_score(p_5_a_all, r_5_a_all))[:5]}\t"
#         f"{str(p_all_a_all)[:5]}  {str(r_all_a_all)[:5]}  {str(f1_score(p_all_a_all, r_all_a_all))[:5]}\t"
#         # print("********** absent absolutely **********")
#         f"{str(p_5_ab)[:5]}  {str(r_5_ab)[:5]}  {str(f1_score(p_5_ab, r_5_ab))[:5]}\t"
#         f"{str(p_all_ab)[:5]}  {str(r_all_ab)[:5]}  {str(f1_score(p_all_ab, r_all_ab))[:5]}\t"
#         # print("********** absent not absolutely **********")
#         f"{str(p_5_a)[:5]}  {str(r_5_a)[:5]}  {str(f1_score(p_5_a, r_5_a))[:5]}\t"
#         f"{str(p_all_a)[:5]}  {str(r_all_a)[:5]}  {str(f1_score(p_all_a, r_all_a))[:5]}\t")
#
#     absent_all = [[" ".join(pp[0]).replace(" ##", "") for pp in p] for p in prediction_absent_all]
#     absent_stem = word2stem([[" ".join(pp[0]).replace(" ##", "").split(" ") for pp in p] for p in prediction_absent])
#     return sum([len(set([tuple(pp[0]) for pp in p]))
#                 for p in absent_stem])/len(prediction_absent_all), \
#         absent_all

# if __name__ == "__main__":
#     # get_score(sys.argv[1])
#     get_score(os.path.join("..", "results"))
