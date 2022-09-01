import os
import pickle

from nltk.stem.porter import PorterStemmer


def write_line(fp, line, split_str="\n"):
    fp.write(line + split_str)


def write_lines(filename, lines, mode, split_str="\n"):
    with open(filename, mode=mode, encoding="utf-8") as fp:
        for line in lines:
            write_line(fp, line, split_str)


def read_lines(filename):
    with open(filename, encoding='utf-8') as fp:
        lines = fp.read().strip().split('\n')
        return lines


def res_check(one_res, stop_char):
    # if len(one_res) < 2:
    #     return False
    for c in stop_char:
        if c in one_res:
            return False
    return True


# def get_entities(input_path):
#     in_lines = read_lines(os.path.join(input_path, "seq.in"))
#     entity_lines = []
#     for in_line in in_lines:
#         in_line = in_line.split(" ")
#         entity = []
#         entity_line = []
#         start_flag = False
#         for idx, cur_token in enumerate(in_line):
#             if cur_token == "[SEP]":
#                 break
#             last_token = None if idx == 0 else in_line[idx - 1]
#             next_token = None if idx == len(in_line) - 1 else in_line[idx + 1]
#             if last_token == "[MASK]" and cur_token != "[MASK]" and cur_token != "<S>":
#                 start_flag = True
#             elif cur_token == "[MASK]" and next_token == "[MASK]":
#                 start_flag = False
#                 if len(entity) > 0:
#                     entity_line.append(entity)
#                     entity = []
#             if start_flag:
#                 entity.append(cur_token)
#         entity_lines.append(entity_line)
#     return entity_lines


def tagging2test(input_path, output_path, tagging_c_lines, tagging_c_prob_lines, tagging_p_lines, tagging_p_prob_lines,
                 total_mask_num, top_n, do_stem, has_prompt, auxiliary_sentence="phrase of****is",
                 entity_threshold=-100000000):
    stop_char = ["。", "，", "、", "?", ".", "：", "[UNK]", ",", "！", "；"]
    prefix_auxiliary = auxiliary_sentence.split("****")[0].split(" ")
    suffix_auxiliary = auxiliary_sentence.split("****")[1].split(" ")
    in_lines = read_lines(os.path.join(input_path, "seq.in"))
    center_stem_truncated_num = 0
    all_center_num = 0
    # tagging_c_lines = read_lines(os.path.join(input_path, "seq.in.pred_c"))
    # tagging_c_prob_lines = pickle.load(open(os.path.join(input_path, "seq.in.pred_c.prob"), 'rb'))
    # tagging_p_lines = read_lines(os.path.join(input_path, "seq.in.pred_p"))
    # tagging_p_prob_lines = pickle.load(open(os.path.join(input_path, "seq.in.pred_p.prob"), 'rb'))
    res = list()
    stemmer = PorterStemmer()
    for idx, (in_line, tagging_c_line, tagging_p_line, tagging_c_prob_line, tagging_p_prob_line) in \
            enumerate(zip(in_lines, tagging_c_lines, tagging_p_lines, tagging_c_prob_lines, tagging_p_prob_lines)):       
        in_line = in_line.split(" ")
        tagging_c_line = tagging_c_line.replace("X", "I").split(" ")
        #if idx == 0:
        #    print(tagging_c_line)
        tagging_c_prob_line = [max(item) for item in tagging_c_prob_line]
        in_line = in_line[in_line.index("[SEP]") + 1:]
        tagging_c_line = tagging_c_line[1:-1]
        tagging_c_prob_line = tagging_c_prob_line[1:-1] 
        # assert len(tagging_c_line) == len(tagging_c_prob_line)
        # assert len(in_line) <= len(tagging_c_line) and len(in_line) <= len(tagging_p_line)
        res_line = list()
        one_res = list()
        total_prob = 0
        has_b = False
        for idx2, (i, t, p) in enumerate(zip(in_line, tagging_c_line, tagging_c_prob_line)):
            #print(tagging_c_line)
            t_last = "S" if idx2 == 0 else tagging_c_line[idx2 - 1]
            if t == "B":
                if len(one_res) != 0 and (t_last == "B" or t_last == "I"):
                    prob = total_prob / len(one_res)
                    if prob > entity_threshold and res_check(one_res, stop_char):
                        res_line.append((" ".join(one_res), prob))
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
                    res_line.append((" ".join(one_res), prob))
            else:
                print(t)
                assert False
        res_line.sort(key=lambda item: item[1], reverse=True)
        res_line = [item[0] for item in res_line]
        res_line = sorted(set(res_line), key=res_line.index)  # 保持原顺序去重

        # stem后去重
        res_line_stem = [stemmer.stem(r) for r in res_line]
        new_res_line = []
        for idx2, r in enumerate(res_line):
            if stemmer.stem(r) in res_line_stem[:idx2]:
                continue
            new_res_line.append(r)
        all_center_num += len(res_line)
        center_stem_truncated_num += (len(res_line) - len(new_res_line))
        res_line = new_res_line.copy() if do_stem == "True" else res_line

        res_line = res_line[0:top_n]
        res.append(res_line)
    new_in_lines = list()
    new_label_c_lines = list()
    new_label_p_lines = list()
    assert len(res) == len(in_lines)
    for idx, (in_line, res_line) in enumerate(zip(in_lines, res)):
        in_line = in_line.split(" ")
        # in_line.reverse()
        # in_line = in_line[:in_line.index("<S>")] if "<S>" in in_line else in_line
        # in_line.reverse()
        in_line = in_line[in_line.index("[SEP]"):]
        new_in_line = []
        for r in res_line:
            if has_prompt:
                new_in_line += prefix_auxiliary + r.split(" ") + suffix_auxiliary
            new_in_line += ["[MASK]"] * total_mask_num + r.split(" ") + ["[MASK]"] * total_mask_num + ["<S>"]
        if has_prompt:
            new_in_line += ["other", "phrases", "are"]
        new_in_line += ["[MASK]"] * 10
        new_in_line += in_line
        new_in_lines.append(" ".join(new_in_line))

        new_label_c_line = ["N"] * (new_in_line.index("[SEP]") + 1) + tagging_c_lines[idx].split(" ")[1:]
        new_label_p_line = ["N"] * (new_in_line.index("[SEP]") + 1) + tagging_p_lines[idx].split(" ")[1:]
        c_pad = len(new_in_line) - len(new_label_c_line)
        p_pad = len(new_in_line) - len(new_label_p_line)
        new_label_c_line = new_label_c_line[:len(new_in_line)] if c_pad < 0 else new_label_c_line + ["O"] * c_pad
        new_label_p_line = new_label_p_line[:len(new_in_line)] if p_pad < 0 else new_label_p_line + ["O"] * p_pad
        new_label_c_lines.append(" ".join(new_label_c_line))
        new_label_p_lines.append(" ".join(new_label_p_line))
        assert len(new_in_line) == len(new_label_c_line)
        assert len(new_in_line) == len(new_label_p_line)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    write_lines(os.path.join(output_path, "seq.in"), new_in_lines, "w+")
    write_lines(os.path.join(output_path, "seq.out"), new_in_lines, "w+")
    write_lines(os.path.join(output_path, "seq.labelc"), new_label_c_lines, "w+")
    write_lines(os.path.join(output_path, "seq.labelp"), new_label_p_lines, "w+")
    if do_stem == "True":
        center_stem_truncated_num = 0
    print(f"Center Truncated by Stemming Num = {center_stem_truncated_num}/{all_center_num}"
          f"({center_stem_truncated_num/(all_center_num + 1e-6)*100}%)") 
    return new_in_lines


def run():
    tagging2test(input_path=".", output_path=os.path.join(".", "output"),
                 total_mask_num=6, top_n=3, entity_threshold=-1000000, auxiliary_sentence="phrase of****is")


if __name__ == '__main__':
    run()


