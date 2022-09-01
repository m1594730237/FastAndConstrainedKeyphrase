import json

from tqdm import tqdm

from util.file_util import read_lines


def load_dataset(filename):
    lines = read_lines(filename)
    res = list()
    for line in tqdm(lines):
        j = json.loads(line)
        res.append(j)
    return res


