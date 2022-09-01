
from util import *
from util.file_util import read_lines


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

