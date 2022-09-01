
from util import *


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
