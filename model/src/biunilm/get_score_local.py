import json
import os
import sys
import pickle

from nltk.stem.porter import PorterStemmer
from tqdm import tqdm
from bert import tokenization


def read_lines(filename):
    with open(filename, encoding='utf-8') as fp:
        lines = fp.read().strip().split('\n')
        return lines


class Tokenization:
    def __init__(self):
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=os.path.join(".", "vocab.txt"),
            do_lower_case=True)

        self.vocab_bert = set(read_lines(os.path.join(".", "vocab.txt")))

    def tokenize(self, article):
        return self.tokenizer.tokenize(article)

    def convert_unk(self, word):
        if word in self.vocab_bert:
            return word
        else:
            return "[UNK]"


def item2token(items, cached_path=""):
    tokenization = Tokenization()
    all_content = list()
    all_keywords = list()
    if cached_path != "" and not os.path.exists(cached_path):
        os.mkdir(cached_path)

    for idx, item in tqdm(enumerate(items)):
        content = tokenization.tokenize(item["title"] + " . " + item["abstract"])
        keywords = item["keywords"]
        for idx2, keyword in enumerate(keywords):
            keywords[idx2] = tokenization.tokenize(keyword)
        all_content.append(content)
        all_keywords.append(keywords)
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
                if len(one_res) == 0 or t_last == "O" or t_last == "N" or t_last == "[CLS]" or t_last == "[SEP]":
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


def get_pr_kws(prediction, kws, top_n):
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
        kw = set([tuple(item) for item in kws[i]])
        # if len(kw) == 0:
        #     continue
        prediction_line = prediction[i]
        # prediction_line.sort(key=lambda item: item[1], reverse=True)
        # prediction_line = prediction_line[0:min(top_n, len(prediction_line))]
        prediction_line = set([tuple(item) for item in prediction_line])
        for pre in prediction_line:
            if pre in kw:
                p += 1
            total_p += 1
        for k in kw:
            if k in prediction_line:
                r += 1
            total_r += 1
    if total_p == 0:
        assert p == 0
    if total_r == 0:
        assert r == 0
    return p/(total_p + 1e-6), r/(total_r + 1e-6)


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
    return total_p/total_n, total_r/total_n


def get_pr(prediction, kws, top_n):
    return get_pr_kws(prediction, kws, top_n)


def remove_mask(line, mask_num, total_mask_num):
    loop = 0
    line = line.copy()
    while loop < len(line):
        if line[loop] == "[MASK]" and find_sublist(line[:loop + 1], ["other", "phrases", "are"]) == -1:
            line = line[:loop + mask_num] + line[loop + total_mask_num:]
            loop += mask_num - 1
        loop += 1
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
            elif len(longest_common_sublist(c_stem, k_stem)[0]) == 0:
                absent_ab_line.append(k)
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


def get_score(path, mask_num=3, total_mask_num=3):
    items = load_dataset(os.path.join(path, "data.json"))
    contents, kws = item2token(items)
    seq_in_lines = read_lines(os.path.join(path, "seq.in"))
    test_result_lines = read_lines(os.path.join(path, "test_results.txt"))
    test_result_prob_lines = read_lines(os.path.join(path, "test_results_prob.txt"))
    test_tagging_c_lines = read_lines(os.path.join(path, "seq.in.pred_c"))
    test_tagging_c_prob_lines = pickle.load(open(os.path.join(path, "seq.in.pred_c.prob"), 'rb'))
    test_tagging_p_lines = read_lines(os.path.join(path, "seq.in.pred_p"))
    test_tagging_p_prob_lines = pickle.load(open(os.path.join(path, "seq.in.pred_p.prob"), 'rb'))
    assert len(seq_in_lines) == len(test_result_lines)
    assert len(seq_in_lines) == len(test_result_prob_lines)
    assert len(seq_in_lines) == len(test_tagging_c_lines)
    assert len(seq_in_lines) == len(test_tagging_c_prob_lines)
    assert len(seq_in_lines) == len(test_tagging_p_lines)
    assert len(seq_in_lines) == len(test_tagging_p_prob_lines)

    kws_present, kws_absent, kws_absent_ab = three_type_kws(contents, kws)

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
        start_center = False
        for idx, t in enumerate(seq_in_line):
            test_result_line = test_result_lines[i].split(" ")
            if test_result_loop >= len(test_result_line):
                break
            test_result_prob_line = test_result_prob_lines[i].split(" ")
            if t == "[MASK]":
                if test_result_line[test_result_loop] != "<T>":
                    phrase.append(test_result_line[test_result_loop])
                sum_prob += float(test_result_prob_line[test_prob_loop])
                test_result_loop += 1
                test_prob_loop += 1
            if idx >= 1 and seq_in_line[idx - 1] == "[MASK]" and \
                    seq_in_line[idx] != "[MASK]" and seq_in_line[idx] != "[SEP]" and seq_in_line[idx] != "<S>":
                start_center = True
            if idx >= 1 and start_center and seq_in_line[idx - 1] != "[MASK]" and seq_in_line[idx] == "[MASK]":
                start_center = False
            if t == "<S>":
                if len(phrase) > 0:
                    prediction_line.append([phrase.copy(), sum_prob / len(phrase)])
                phrase.clear()
                sum_prob = 0
                start_center = False
            if t == ";" or t == "[SEP]":
                if len(phrase) > 0:
                    prediction_line_ab.append([phrase.copy(), sum_prob / len(phrase)])
                phrase.clear()
                sum_prob = 0
                start_center = False
            if t == "[SEP]":
                break
            if start_center:
                phrase.append(t)
        # print(prediction_line)
        prediction_line.sort(key=lambda item: item[1], reverse=True)
        prediction_line_ab.sort(key=lambda item: item[1], reverse=True)
        prediction_absent.append(prediction_line)
        prediction_absent_ab.append(prediction_line_ab)

    for idx, (tagging_line, tagging_prob_line) in enumerate(zip(test_tagging_p_lines, test_tagging_p_prob_lines)):
        tagging_line = tagging_line.split(" ")
        # tagging_prob_line = tagging_prob_line.split(" ")
        tagging_prob_line = [max(item) for item in tagging_prob_line]
        tagging_line = tagging_line[1:-1]
        tagging_prob_line = tagging_prob_line[1:-1]
        test_tagging_p_lines[idx] = tagging_line
        test_tagging_p_prob_lines[idx] = tagging_prob_line

    prediction_present = bio2phrases(contents, test_tagging_p_lines, test_tagging_p_prob_lines)

    assert len(prediction_absent) == len(seq_in_lines)
    assert len(prediction_present) == len(seq_in_lines)
    assert len(prediction_absent_ab) == len(seq_in_lines)
    prediction_present = [[[tuple(pp[0]), pp[1]] for pp in p]for p in prediction_present]
    prediction_absent = [[[tuple(pp[0]), pp[1]] for pp in p] for p in prediction_absent]
    prediction_absent_ab = [[[tuple(pp[0]), pp[1]] for pp in p] for p in prediction_absent_ab]
    prediction_absent_all = [p1 + p2 for p1, p2 in zip(prediction_absent, prediction_absent_ab)]
    prediction_all = [p1 + p2 + p3 for p1, p2, p3 in zip(prediction_absent, prediction_absent_ab, prediction_present)]
    kws_present = [[tuple(pp) for pp in p] for p in kws_present]
    kws_absent = [[tuple(pp) for pp in p] for p in kws_absent]
    kws_absent_ab = [[tuple(pp) for pp in p] for p in kws_absent_ab]
    kws_absent_all = [p1 + p2 for p1, p2 in zip(kws_absent, kws_absent_ab)]
    kws_all = [p1 + p2 + p3 for p1, p2, p3 in zip(kws_absent, kws_absent_ab, kws_present)]
    # p_3_ps, r_3_ps = get_prf(prediction_present, kws_present, top_n=3)
    p_5_p, r_5_p = get_pr(prediction_present, kws_present, top_n=5)
    p_all_p, r_all_p = get_pr(prediction_present, kws_present, top_n=10000000)
    p_5_a, r_5_a = get_pr(prediction_absent, kws_absent, top_n=5)
    p_all_a, r_all_a = get_pr(prediction_absent, kws_absent, top_n=10000000)
    p_5_ab, r_5_ab = get_pr(prediction_absent_ab, kws_absent_ab, top_n=5)
    p_all_ab, r_all_ab = get_pr(prediction_absent_ab, kws_absent_ab, top_n=10000000)
    p_5_a_all, r_5_a_all = get_pr(prediction_absent_all, kws_absent_all, top_n=5)
    p_all_a_all, r_all_a_all = get_pr(prediction_absent_all, kws_absent_all, top_n=10000000)
    p_5_all, r_5_all = get_pr(prediction_all, kws_all, top_n=5)
    p_all_all, r_all_all = get_pr(prediction_all, kws_all, top_n=10000000)
    # print("********** present **********")
    # print(f"Top-3: P = {p_3_ps}  {r_3_ps}  {f1_score(p_3_ps, r_3_ps)}")
    print(f"{str(p_5_p * 100)[:4]}  {str(r_5_p * 100)[:4]}  {str(f1_score(p_5_p, r_5_p) * 100)[:4]} \t"
          f"{str(p_all_p * 100)[:4]}  {str(r_all_p * 100)[:4]}  {str(f1_score(p_all_p, r_all_p) * 100)[:4]}\t"
    # print("********** absent all **********")
          f"{str(p_5_a_all * 100)[:4]}  {str(r_5_a_all * 100)[:4]}  {str(f1_score(p_5_a_all, r_5_a_all) * 100)[:4]}\t"
          f"{str(p_all_a_all * 100)[:4]}  {str(r_all_a_all * 100)[:4]}  {str(f1_score(p_all_a_all, r_all_a_all) * 100)[:4]}\t"
    # print("********** absent absolutely **********")
          f"{str(p_5_ab * 100)[:4]}  {str(r_5_ab * 100)[:4]}  {str(f1_score(p_5_ab, r_5_ab) * 100)[:4]}\t"
          f"{str(p_all_ab * 100)[:4]}  {str(r_all_ab * 100)[:4]}  {str(f1_score(p_all_ab, r_all_ab) * 100)[:4]}\t"
    # print("********** absent not absolutely **********")
          f"{str(p_5_a * 100)[:4]}  {str(r_5_a * 100)[:4]}  {str(f1_score(p_5_a, r_5_a) * 100)[:4]}\t"
          f"{str(p_all_a * 100)[:4]}  {str(r_all_a * 100)[:4]}  {str(f1_score(p_all_a, r_all_a) * 100)[:4]}\t"
    # print("********** all **********")
          f"{str(p_5_all * 100)[:4]}  {str(r_5_all * 100)[:4]}  {str(f1_score(p_5_all, r_5_all) * 100)[:4]}\t"
          f"{str(p_all_all * 100)[:4]}  {str(r_all_all * 100)[:4]}  {str(f1_score(p_all_all, r_all_all) * 100)[:4]}\t")





get_score(sys.argv[1])
