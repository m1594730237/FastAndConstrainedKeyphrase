from random import randint, shuffle, choice
from random import random as rand
import math
import torch

from loader_utils import get_random_word, batch_list_to_batch_tensors, Pipeline

# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
#    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.


def truncate_tokens_pair(src_tk_in, src_tk_out, tokens_labelp, tokens_labelc, max_len, max_len_a=0, \
                         max_len_b=0, trunc_seg=None, always_truncate_tail=False):
    '''
    '''

    # assert max_len_a + max_len_b <= max_len
    max_len_a = max_len - max_len_b
    src_tk_in = src_tk_in[:max_len_a]
    src_tk_out = src_tk_out[:max_len_a]
    tokens_labelp = tokens_labelp[:max_len_a]
    tokens_labelc = tokens_labelc[:max_len_a]
    return src_tk_in, src_tk_out, tokens_labelp, tokens_labelc

class Seq2SeqDataset(torch.utils.data.Dataset):
    """ Load sentence pair (sequential or random order) from corpus """

    def __init__(self, file_src, file_tgt, file_labelp, file_labelc, batch_size, tokenizer, max_len, label_indexer,\
                 file_oracle=None, short_sampling_prob=0.1, sent_reverse_order=False,\
                 bi_uni_pipeline=[]):
        super().__init__()
        self.tokenizer = tokenizer  # tokenize function
        self.max_len = max_len  # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order
        self.label_indexer = label_indexer
        # read the file into memory
        self.ex_list = []
        if file_oracle is None:
            with open(file_src, "r", encoding='utf-8') as f_src, \
                    open(file_tgt, "r", encoding='utf-8') as f_tgt, \
                    open(file_labelp, "r", encoding='utf-8') as f_labelp, \
                    open(file_labelc, "r", encoding='utf-8') as f_labelc:
                for src, tgt, labelp, labelc in zip(f_src, f_tgt, f_labelp, f_labelc):
                    src_list = src.replace("<S>", ";").strip().split(" [SEP] ")
                    tgt_list = tgt.replace("<S>", ";").strip().split(" [SEP] ")
                    src_tk_in = src_list[1].split(" ")
                    tgt_tk_in = src_list[0].split(" ")
                    src_tk_out = tgt_list[1].split(" ")
                    tgt_tk_out = tgt_list[0].split(" ")
                    # src_tk = tokenizer.tokenize(src.strip())
                    # tgt_tk = tokenizer.tokenize(tgt.strip())
                    labelp_list = labelp.strip().split(" ")[len(tgt_tk_in)+1:]
                    labelc_list = labelc.strip().split(" ")[len(tgt_tk_in)+1:]
                    assert len(src_tk_out) > 0
                    assert len(src_tk_in) > 0
                    assert len(src_tk_in) == len(src_tk_out)
                    assert len(tgt_tk_in) == len(tgt_tk_out)
                    self.ex_list.append((src_tk_in, tgt_tk_in, src_tk_out, tgt_tk_out, labelp_list, labelc_list))
        print('Load {0} documents'.format(len(self.ex_list)))

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        proc = choice(self.bi_uni_pipeline)
        instance = proc(instance, self.label_indexer)
        return instance

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, len(self.ex_list)-1)
                batch.append(self.__getitem__(idx))
            # To Tensor
            yield batch_list_to_batch_tensors(batch)



class Preprocess4Seq2seq(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, \
                 new_segment_ids=False, truncate_config={}, mask_source_words=False, \
                 skipgram_prb=0, skipgram_size=0, mask_whole_word=False, \
                 mode="s2s", has_oracle=False, num_qkv=0, s2s_special_token=False, \
                 s2s_add_segment=False, s2s_share_segment=False, pos_shift=False):
        super().__init__()
        self.max_len = max_len # max tokens of a+b, max_len_a + max_len_b <= max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self._tril_matrix = torch.tril(torch.ones((max_len, max_len), dtype=torch.long)) # 下三角掩码矩阵
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word
        self.new_segment_ids = new_segment_ids
        self.always_truncate_tail = truncate_config.get('always_truncate_tail', False)
        self.max_len_a = truncate_config.get('max_len_a', None)
        self.max_len_b = truncate_config.get('max_len_b', None)
        self.trunc_seg = truncate_config.get('trunc_seg', None)
        self.task_idx = 3   # relax projection layer for different tasks
        self.mask_source_words = mask_source_words
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.has_oracle = has_oracle
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift

    def __call__(self, instance, label_indexer):
        # tokens_a, tokens_b, tokens_a_labelp, tokens_a_labelc = instance[:4]
        src_tk_in, tgt_tk_in, src_tk_out, tgt_tk_out, tokens_labelp, tokens_labelc = instance[:6]

        assert len(src_tk_in) == len(src_tk_out)
        assert len(tgt_tk_in) == len(tgt_tk_out)
        assert len(src_tk_in) == len(tokens_labelp)
        tokens_b_label = ['O'] * len(tgt_tk_in)

        if (len(src_tk_in) + len(tgt_tk_in)) > self.max_len - 3:
            src_tk_in, src_tk_out, tokens_labelp, tokens_labelc = truncate_tokens_pair(src_tk_in, src_tk_out, \
                                                  tokens_labelp, tokens_labelc, \
                                                  self.max_len - 3, max_len_a=self.max_len_a, max_len_b=len(tgt_tk_in), \
                                                  trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)
        
        assert (len(src_tk_in) + len(tgt_tk_in)) <= self.max_len - 3

        # Add Special Tokens
        tokens = ['[CLS]'] + src_tk_in + ['[SEP]'] + tgt_tk_in + ['[SEP]']
        tokens_gold = ['[CLS]'] + src_tk_out + ['[SEP]'] + tgt_tk_out + ['[SEP]']
        labelp = ['[CLS]'] + tokens_labelp + ['[SEP]'] + tokens_b_label + ['[SEP]']
        labelc = ['[CLS]'] + tokens_labelc + ['[SEP]'] + tokens_b_label + ['[SEP]']
        #print(tokens)
        #print(tokens_gold)
        #print(labelp)
        #print(labelc)

        labelp_mask = [0] + [1] * len(tokens_labelp) + [0] + [0] * len(tokens_b_label) + [0]
        labelc_mask = [0] + [1] * len(tokens_labelc) + [0] + [0] * len(tokens_b_label) + [0]
        assert len(tokens) == len(labelp)
        assert len(tokens) == len(labelc)
        assert len(labelc) == len(labelc_mask)
        assert len(labelp) == len(labelp_mask)
        assert len(tokens) == len(tokens_gold)
 
        if self.new_segment_ids:
            if self.mode == "s2s":
                if self.s2s_add_segment:
                    if self.s2s_share_segment:
                        segment_ids = [0] + [1] * (len(src_tk_in)+1) + [5]*(len(tgt_tk_in)+1)
                    else:
                        segment_ids = [4] + [6] * \
                            (len(src_tk_in)+1) + [5]*(len(tgt_tk_in)+1)
                else:
                    segment_ids = [4] * (len(src_tk_in)+2) + \
                        [5]*(len(tgt_tk_in)+1)
            else:
                segment_ids = [2] * (len(tokens))
        else:
            segment_ids = [0]*(len(src_tk_in)+2) + [1]*(len(tgt_tk_in)+1)

        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        # masked_ids, masked_pos, masked_weights
        masked_pos = []
        masked_tokens = []
        masked_weights = []
        for pos, token in enumerate(tokens):
            if token == "[MASK]":
                masked_pos.append(pos)
                if tokens_gold[pos] == "<T>":
                    masked_tokens.append("[unused101]")
                else:
                    masked_tokens.append(tokens_gold[pos])
                if tokens_gold[pos] == "<T>":
                    masked_weights.append(1.0)
                else:
                    masked_weights.append(1.0)
        # Token Indexing
        masked_ids = self.indexer(masked_tokens)

        n_pred = len(masked_ids)

        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            if masked_ids is not None:
                masked_ids.extend([0]*n_pad)
            if masked_pos is not None:
                masked_pos.extend([0]*n_pad)
            if masked_weights is not None:
                masked_weights.extend([0]*n_pad)
        else:
            masked_ids = masked_ids[:self.max_pred]
            masked_pos = masked_pos[:self.max_pred]
            masked_weights = masked_weights[:self.max_pred]


        assert len(tokens) == len(labelp)
        input_ids = self.indexer(tokens)
        labelp_ids = []
        for l in labelp:
            labelp_ids.append(label_indexer[l])
        assert len(input_ids) == len(labelp_ids)
        labelc_ids = []
        for l in labelc:
            labelc_ids.append(label_indexer[l])
        assert len(input_ids) == len(labelc_ids)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        labelp_ids.extend([0]*n_pad)
        labelc_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        labelp_mask.extend([0]*n_pad)
        labelc_mask.extend([0]*n_pad)

        if self.num_qkv > 1:
            mask_qkv = [0]*(len(src_tk_in)+2) + [1] * (len(tgt_tk_in)+1)
            mask_qkv.extend([0]*n_pad)
        else:
            mask_qkv = None

        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        input_mask[:, :len(src_tk_in)+2].fill_(1)
        second_st, second_end = len(
                src_tk_in)+2, len(src_tk_in)+len(tgt_tk_in)+3
        input_mask[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end-second_st, :second_end-second_st])
        #input_mask[second_st:second_end, second_st:second_end].fill_(1)

        #print(input_ids)
        #print(segment_ids)
        #print(input_mask)
        #print(labelp_mask)
        #print(labelc_mask)
        #print(masked_weights)
        #print(masked_ids)
        #print(masked_pos)

        assert len(input_ids) == len(segment_ids)
        assert len(labelp_ids) == len(labelc_ids)
        assert len(labelp_mask) == len(labelp_ids)
        assert len(labelc_mask) == len(labelp_mask)
        assert len(input_ids) == len(labelp_ids)
        assert len(masked_ids) == len(masked_pos)
        assert len(masked_weights) == len(masked_pos)

        return (input_ids, segment_ids, input_mask, \
                labelp_ids, labelc_ids, labelp_mask, labelc_mask, mask_qkv, \
                masked_ids, masked_pos, masked_weights, -1, self.task_idx)

class Preprocess4SeqLabel(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, vocab_words, indexer, max_len=512, max_tgt_length=128, new_segment_ids=False, mode="s2s",
                 num_qkv=0, s2s_special_token=False, s2s_add_segment=False, s2s_share_segment=False, pos_shift=False):
        super().__init__()
        self.max_len = max_len
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids
        self.task_idx = 3   # relax projection layer for different tasks
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.max_tgt_length = max_tgt_length
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift

    def __call__(self, instance):
        tokens_a, max_a_len = instance

        # Add Special Tokens
        if self.s2s_special_token:
            padded_tokens_a = ['[S2S_CLS]'] + tokens_a + ['[S2S_SEP]']
        else:
            padded_tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        assert len(padded_tokens_a) <= max_a_len + 2
        if max_a_len + 2 > len(padded_tokens_a):
            padded_tokens_a += ['[PAD]'] * \
                (max_a_len + 2 - len(padded_tokens_a))
        assert len(padded_tokens_a) == max_a_len + 2
        max_len_in_batch = min(self.max_tgt_length +
                               max_a_len + 2, self.max_len)
        tokens = padded_tokens_a
        if self.new_segment_ids:
            if self.mode == "s2s":
                _enc_seg1 = 0 if self.s2s_share_segment else 4
                if self.s2s_add_segment:
                    if self.s2s_share_segment:
                        segment_ids = [
                            0] + [1]*(len(padded_tokens_a)-1) + [5]*(max_len_in_batch - len(padded_tokens_a))
                    else:
                        segment_ids = [
                            4] + [6]*(len(padded_tokens_a)-1) + [5]*(max_len_in_batch - len(padded_tokens_a))
                else:
                    segment_ids = [4]*(len(padded_tokens_a)) + \
                        [5]*(max_len_in_batch - len(padded_tokens_a))
            else:
                segment_ids = [2]*max_len_in_batch
        else:
            segment_ids = [0]*(len(padded_tokens_a)) \
                + [1]*(max_len_in_batch - len(padded_tokens_a))

        if self.num_qkv > 1:
            mask_qkv = [0]*(len(padded_tokens_a)) + [1] * \
                (max_len_in_batch - len(padded_tokens_a))
        else:
            mask_qkv = None

        position_ids = []
        for i in range(len(tokens) + 2):
            position_ids.append(i)
        for i in range(len(tokens) + 2, max_a_len + 2):
            position_ids.append(0)
        for i in range(max_a_len + 2, max_len_in_batch):
            position_ids.append(i - (max_a_len + 2) + len(tokens) + 2)

        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        input_mask = torch.zeros(
            max_len_in_batch, max_len_in_batch, dtype=torch.long)
        if self.mode == "s2s":
            input_mask[:, :len(tokens_a)+2].fill_(1)
        else:
            st, end = 0, len(tokens_a) + 2
            input_mask[st:end, st:end].copy_(
                self._tril_matrix[:end, :end])
            input_mask[end:, :len(tokens_a)+2].fill_(1)
        second_st, second_end = len(padded_tokens_a), max_len_in_batch

        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])


        return (input_ids, segment_ids, position_ids, input_mask, mask_qkv, self.task_idx)

class Preprocess4Seq2seqDecoder(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, vocab_words, indexer, max_len=512, max_tgt_length=128, new_segment_ids=False, mode="s2s",
                 num_qkv=0, s2s_special_token=False, s2s_add_segment=False, s2s_share_segment=False, pos_shift=False):
        super().__init__()
        self.max_len = max_len
        self.max_pred = 64
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids
        self.task_idx = 3   # relax projection layer for different tasks
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.max_tgt_length = max_tgt_length
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift

    def __call__(self, instance):
        tokens, max_a_len = instance

        tokens_list = tokens.replace("<S>", ";").split(" [SEP] ")
        assert len(tokens_list) == 2
        tokens_a = tokens_list[1].split(" ")
        tokens_b = tokens_list[0].split(" ")

        if len(tokens_a) + len(tokens_b) > self.max_len - 3:
            tokens_a = tokens_a[:self.max_len - 3 - len(tokens_b)]

        assert (len(tokens_a) + len(tokens_b)) <= self.max_len - 3
        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

        if self.new_segment_ids:
            if self.mode == "s2s":
                if self.s2s_add_segment:
                    if self.s2s_share_segment:
                        segment_ids = [0] + [1] * (len(tokens_a)+1) + [5]*(len(tokens_b)+1)
                    else:
                        segment_ids = [4] + [6] * \
                            (len(tokens_a)+1) + [5]*(len(tokens_b)+1)
                else:
                    segment_ids = [4] * (len(tokens_a)+2) + \
                        [5]*(len(tokens_b)+1)
            else:
                segment_ids = [2] * (len(tokens))
        else:
            segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)

        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        # masked_ids, masked_pos, masked_weights
        masked_pos = []
        for pos, token in enumerate(tokens):
            if token == "[MASK]":
                masked_pos.append(pos)
        # Token Indexing
        # masked_ids = self.indexer(masked_tokens)

        n_pred = len(masked_pos)

        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            if masked_pos is not None:
                masked_pos.extend([len(tokens) - 1]*n_pad)
        else:
            masked_pos = masked_pos[:self.max_pred]

        input_ids = self.indexer(tokens)
        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)

        if self.num_qkv > 1:
            mask_qkv = [0]*(len(tokens_a)+2) + [1] * (len(tokens_b)+1)
            mask_qkv.extend([0]*n_pad)
        else:
            mask_qkv = None

        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        input_mask[:, :len(tokens_a)+2].fill_(1)
        second_st, second_end = len(
                tokens_a)+2, len(tokens_a)+len(tokens_b)+3
        input_mask[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end-second_st, :second_end-second_st])
        #input_mask[second_st:second_end, second_st:second_end].fill_(1)
        return (input_ids, segment_ids, masked_pos, input_mask, mask_qkv, self.task_idx)
