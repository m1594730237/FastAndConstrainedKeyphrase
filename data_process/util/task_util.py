from util import *
from util.file_util import read_lines
from util.list_util import find_sublist


def item2token(items, cached_path=""):
    tokenization = Tokenization()
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
                content = tokenization.tokenize(item["title"] + temp_split + item["abstract"])
                keywords = item["keywords"]
                for idx2, keyword in enumerate(keywords):
                    keywords[idx2] = tokenization.tokenize(keyword)
                fpc.write("\t\t".join(content) + "\n")
                fpk.write("\t\t".join(["  ".join(k) for k in keywords]) + "\n")
                all_content.append(content)
                all_keywords.append(keywords)
        with open(cached_finished, "w+") as _:
            pass
    # if len(all_keywords) <= 1:
    #     all_keywords = []
    return all_content, all_keywords


def nest_check(presents, centers):
    res = {
        "present_present": False,
        "present_center": False,
        "center_center": False,
    }
    for i in range(len(presents)):
        for j in range(i + 1, len(presents)):
            p1 = presents[i]
            p2 = presents[j]
            if len(p1) > 0 and len(p2) > 0 and (find_sublist(p1, p2) != -1 or find_sublist(p2, p1) != -1):
                res["present_present"] = True
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            c1 = centers[i]
            c2 = centers[j]
            if len(c1) > 0 and len(c2) > 0 and (find_sublist(c1, c2) != -1 or find_sublist(c2, c1) != -1):
                res["center_center"] = True
    for i in range(len(presents)):
        for j in range(len(centers)):
            p = presents[i]
            c = centers[j]
            if len(p) > 0 and len(c) > 0 and c != p and (find_sublist(c, p) != -1 or find_sublist(p, c) != -1):
                res["present_center"] = True
    return res


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
            elif t == "O" or t == "N" or t == "[SEP]":
                has_b = False
                if len(one_res) == 0 or t_last == "O" or t_last == "N" or t == "[SEP]":
                    continue
                prob = total_prob / len(one_res)
                if prob > entity_threshold and res_check(one_res, stop_char):
                    res_line.append([one_res, prob])
            else:
                print(t)
                assert False
        res_line.sort(key=lambda item: item[1], reverse=True)
        # res_line = [item[0] for item in res_line]
        # res_line = sorted(set(res_line), key=res_line.index)  
        res_line = res_line[0:top_n]
        res.append(res_line)
    return res





