
def read_lines(filename):
    with open(filename, encoding='utf-8') as fp:
        lines = fp.read().rstrip().split('\n')
        return lines


def find_sublist(x, y):
    l1, l2 = len(x), len(y)
    for i in range(l1):
        if x[i:min(i + l2, l1)] == y:
            return i
    return -1


def in2test(in_lines, mask_num):
    out_lines = []
    start_center = False
    for in_line in in_lines:
        center_line = []
        center = []
        in_line = in_line.split(" ")
        for idx, token in enumerate(in_line):
            if token == "[SEP]":
                break
            if idx >= 1 and in_line[idx - 1] == "[MASK]" and \
                    in_line[idx] != "[MASK]" and in_line[idx] != "[SEP]" and in_line[idx] != "<S>":
                start_center = True
            if idx >= 1 and start_center and in_line[idx - 1] != "[MASK]" and in_line[idx] == "[MASK]":
                start_center = False
            if start_center:
                center.append(token)
            elif len(center) > 0:
                center_line.append(center.copy())
                center.clear()

        out_line = []
        for center in center_line:
            out_line += ["phrase", "of"] + center + ["is"]
            out_line += ["[MASK]"] * mask_num + center + ["[MASK]"] * mask_num + ["<S>"]
        out_line += in_line[find_sublist(in_line, ["<S>", "other"]) + 1:]
        out_lines.append(" ".join(out_line))
    return out_lines


if __name__ == '__main__':
    in2test(read_lines("seq.in"), mask_num=1)







