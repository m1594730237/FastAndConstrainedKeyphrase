"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import logging
import glob
import argparse
import math
import time

from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import random
import pickle

from get_score_merged import get_score_center, get_score
from in2test import in2test
from tagging2test import tagging2test
from tokenization import BertTokenizer, WhitespaceTokenizer
from modeling import BertForSeq2SeqDecoder
from modeling import BertForSeqLabel
from optimization import BertAdam, warmup_linear
# from get_score import get_score, get_score_present, get_score_center
# from get_score import get_score_absent

from data_parallel import DataParallelImbalance
from loader_utils import batch_list_to_batch_tensors
from seq2seq_loader import Preprocess4Seq2seqDecoder
from seq2seq_loader import Preprocess4SeqLabel
import torch.nn.functional as F
from parameters_infer import get_parameters

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)


def main():
    parser = get_parameters()
    args = parser.parse_args()

    s2l_time, seq_in_lines, output_labelc_line, labelc_prob_list, output_labelp_line, labelp_prob_list = seq2label()
    s2s_time, output_mlm_line, mlm_prob_list = seq2seq(seq_in_lines)

    avg_present, present_all, avg_absent, absent_all = get_score(
        path=args.result_dir,
        test_result_lines=output_mlm_line,
        test_result_prob_lines=mlm_prob_list,
        mask_num=args.mask_num,
        total_mask_num=args.mask_num,
        absent_type=args.absent_type,
        test_tagging_p_lines=output_labelp_line,
        test_tagging_p_prob_lines=labelp_prob_list,
    )

    avg_center = get_score_center(
        path=args.result_dir,
        test_tagging_c_lines=output_labelc_line,
        test_tagging_c_prob_lines=labelc_prob_list,
    )

    assert len(present_all) == len(absent_all)

    if not os.path.exists(os.path.join(args.result_dir, "output")):
        os.makedirs(os.path.join(args.result_dir, "output"))
    keyword_all = []
    for i in range(len(present_all)):
        if len(list(set(present_all[i] + absent_all[i]))) > 0:
            keyword_all_line = []
            for item in list(set(present_all[i] + absent_all[i])):
                if "-" not in item:
                    keyword_all_line.append(item)
            keyword_all.append(" // ".join(keyword_all_line))
        else:
            keyword_all.append("None")
    write_lines(os.path.join(args.result_dir, "output", "keywords.out"), keyword_all, "w+")

    print("***** TIME COST *****")
    print(f"TAGGING Time Cost = {s2l_time}s, AVG = {s2l_time / len(seq_in_lines)}")
    print(f"MLM Time Cost = {s2s_time}s, AVG = {s2s_time / len(seq_in_lines)}")
    print(f"TOTAL Time Cost = {s2l_time + s2s_time}s, AVG = {(s2l_time + s2s_time) / (2 * len(seq_in_lines))}")

    print("***** AVG NUMBER *****")
    print(f"AVG Number of PRESENT = {avg_present}")
    print(f"AVG Number of CENTER = {avg_center}")
    print(f"AVG Number of ABSENT = {avg_absent}")


def seq2label():
    parser = get_parameters()

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    tokenizer.max_len = args.max_seq_length

    label_list = ['O', 'B', 'I', 'X', '[S2S_CLS]', '[S2S_SEP]', '[SEP]', '[CLS]', '[S2S_SOS]']

    def convert_label_ids_to_tokens(id_list, label_list):
        id_str_list = []
        for i in id_list:
            id_str_list.append(label_list[i])
        return id_str_list

    bi_uni_pipeline = []
    # max tgt len for seq2label is set to 1
    bi_uni_pipeline.append(Preprocess4SeqLabel(list(tokenizer.vocab.keys()),
                                               tokenizer.convert_tokens_to_ids, args.max_seq_length,
                                               max_tgt_length=1, new_segment_ids=args.new_segment_ids,
                                               mode="s2s", num_qkv=args.num_qkv,
                                               s2s_special_token=args.s2s_special_token,
                                               s2s_add_segment=args.s2s_add_segment,
                                               s2s_share_segment=args.s2s_share_segment,
                                               pos_shift=args.pos_shift))

    amp_handle = None
    if args.fp16 and args.amp:
        from apex import amp
        amp_handle = amp.init(enable_caching=True)
        logger.info("enable fp16 with amp")

    # Prepare model
    cls_num_labels = 2
    type_vocab_size = 6 + \
                      (1 if args.s2s_add_segment else 0) if args.new_segment_ids else 2
    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
        ["[MASK]", "[SEP]", "[S2S_SOS]"])

    def _get_token_id_set(s):
        r = None
        if s:
            w_list = []
            for w in s.split('|'):
                if w.startswith('[') and w.endswith(']'):
                    w_list.append(w.upper())
                else:
                    w_list.append(w)
            r = set(tokenizer.convert_tokens_to_ids(w_list))
        return r

    forbid_ignore_set = _get_token_id_set(args.forbid_ignore_word)
    not_predict_set = _get_token_id_set(args.not_predict_token)
    print(args.model_recover_path)
    for model_recover_path in glob.glob(args.model_recover_path.strip()):
        logger.info("***** Recover model: %s *****", model_recover_path)
        model_recover = torch.load(model_recover_path)
        model = BertForSeqLabel.from_pretrained(args.bert_model, state_dict=model_recover,
                                                num_seq_labels=len(label_list), num_labels=cls_num_labels, num_rel=0,
                                                type_vocab_size=type_vocab_size, task_idx=3, mask_word_id=mask_word_id,
                                                search_beam_size=args.beam_size,
                                                length_penalty=args.length_penalty, eos_id=eos_word_ids,
                                                sos_id=sos_word_id,
                                                forbid_duplicate_ngrams=args.forbid_duplicate_ngrams,
                                                forbid_ignore_set=forbid_ignore_set,
                                                not_predict_set=not_predict_set, ngram_size=args.ngram_size,
                                                min_len=args.min_len,
                                                mode=args.mode, max_position_embeddings=args.max_seq_length,
                                                ffn_type=args.ffn_type,
                                                num_qkv=args.num_qkv, seg_emb=args.seg_emb, pos_shift=args.pos_shift)
        del model_recover

        if args.fp16:
            model.half()
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        torch.cuda.empty_cache()
        model.eval()
        next_i = 0
        # max tgt len for seq2label is set to 1
        max_src_length = args.max_seq_length - 2 - 1

        with open(args.input_file, encoding="utf-8") as fin:
            input_lines = []
            for line in fin.readlines():
                parts = line.strip().split("[SEP] ")
                input_lines.append(parts[-1].split(" "))
        seq_in_lines = copy.deepcopy(input_lines)

        input_lines = [x[:max_src_length] for x in input_lines]
        input_lines = sorted(list(enumerate(input_lines)),
                             key=lambda x: -len(x[1]))

        output_labelp_line = [""] * len(input_lines)
        output_labelc_line = [""] * len(input_lines)
        labelp_prob_list = [None] * len(input_lines)
        labelc_prob_list = [None] * len(input_lines)
        total_batch = math.ceil(len(input_lines) / args.batch_size)

        time_start = time.time()
        with tqdm(total=total_batch) as pbar:
            while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + args.batch_size]
                buf_id = [x[0] for x in _chunk]
                buf = [x[1] for x in _chunk]
                next_i += args.batch_size
                max_a_len = max([len(x) for x in buf])
                instances = []
                for instance in [(x, max_a_len) for x in buf]:
                    for proc in bi_uni_pipeline:
                        instances.append(proc(instance))
                with torch.no_grad():
                    batch = batch_list_to_batch_tensors(
                        instances)
                    batch = [
                        t.to(device) if t is not None else None for t in batch]
                    input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
                    seq_labelp_logits, seq_labelc_logits = model(input_ids, token_type_ids,
                                                                 position_ids, input_mask, task_idx=task_idx,
                                                                 mask_qkv=mask_qkv)

                    seq_labelp_id = torch.argmax(seq_labelp_logits, dim=-1)
                    seq_labelc_id = torch.argmax(seq_labelc_logits, dim=-1)

                    seq_labelp_probs = F.softmax(seq_labelp_logits, dim=2).detach().cpu().numpy().tolist()
                    seq_labelc_probs = F.softmax(seq_labelc_logits, dim=2).detach().cpu().numpy().tolist()
                    for i in range(len(buf)):
                        lp_ids = seq_labelp_id[i]
                        lc_ids = seq_labelc_id[i]
                        output_lp_buf = convert_label_ids_to_tokens(lp_ids, label_list)
                        output_lc_buf = convert_label_ids_to_tokens(lc_ids, label_list)
                        output_tokens_lp = []
                        output_tokens_lc = []
                        for t in output_lp_buf:
                            output_tokens_lp.append(t)
                        for t in output_lc_buf:
                            output_tokens_lc.append(t)
                        output_sequence_labelp = ' '.join(output_tokens_lp)
                        output_sequence_labelc = ' '.join(output_tokens_lc)
                        output_labelp_line[buf_id[i]] = output_sequence_labelp
                        output_labelc_line[buf_id[i]] = output_sequence_labelc
                        labelp_prob_list[buf_id[i]] = seq_labelp_probs[i]
                        labelc_prob_list[buf_id[i]] = seq_labelc_probs[i]

                pbar.update(1)
        time_end = time.time()

        if args.output_labelp_file:
            fn_lpout = args.output_labelp_file
        else:
            fn_lpout = model_recover_path + '.' + args.split + '.labelp'
        with open(fn_lpout, "w", encoding="utf-8") as fout:
            for l in output_labelp_line:
                fout.write(l)
                fout.write("\n")

        if args.output_labelc_file:
            fn_lcout = args.output_labelc_file
        else:
            fn_lcout = model_recover_path + '.' + args.split + '.labelc'
        with open(fn_lcout, "w", encoding="utf-8") as fout:
            for l in output_labelc_line:
                fout.write(l)
                fout.write("\n")

        with open(fn_lpout + ".prob", "wb") as fout:
            pickle.dump(labelp_prob_list, fout)
        with open(fn_lcout + ".prob", "wb") as fout:
            pickle.dump(labelc_prob_list, fout)

        assert args.result_dir
        # print("***** pre-PRESENT *****")
        # avg_present, present_all = get_score_present(
        #     path=args.result_dir,
        #     test_tagging_p_lines=output_labelp_line,
        #     test_tagging_p_prob_lines=labelp_prob_list,
        # )

        # print("***** CENTER *****")
        # avg_center = get_score_center(
        #     path=args.result_dir,
        #     test_tagging_c_lines=output_labelc_line,
        #     test_tagging_c_prob_lines=labelc_prob_list,
        # )

        print("***** TAGGING TO TEST *****")
        new_seq_in_lines = tagging2test(
            input_path=args.result_dir,
            output_path=os.path.join(args.result_dir, "mlm_input"),
            tagging_c_lines=output_labelc_line,
            tagging_c_prob_lines=labelc_prob_list,
            tagging_p_lines=output_labelp_line,
            tagging_p_prob_lines=labelp_prob_list,
            total_mask_num=args.mask_num,
            top_n=args.top_n,
            do_stem=args.center_stem
        )

        # print("***** ABSENT *****")
        if args.absent_type == "GC":
            new_seq_in_lines = in2test(read_lines(os.path.join(args.result_dir, "seq.in")), args.mask_num)
        return time_end - time_start, new_seq_in_lines, output_labelc_line, labelc_prob_list, \
               output_labelp_line, labelp_prob_list


def seq2seq(input_lines):
    parser = get_parameters()

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    tokenizer.max_len = args.max_seq_length

    label_list = ['O', 'B', 'I', 'X', '[S2S_CLS]', '[S2S_SEP]', '[SEP]', '[CLS]', '[S2S_SOS]']

    def convert_label_ids_to_tokens(id_list, label_list):
        id_str_list = []
        for i in id_list:
            id_str_list.append(label_list[i])
        return id_str_list

    bi_uni_pipeline = []
    bi_uni_pipeline.append(Preprocess4Seq2seqDecoder(list(tokenizer.vocab.keys()),
                                                     tokenizer.convert_tokens_to_ids, args.max_seq_length,
                                                     max_tgt_length=args.max_tgt_length,
                                                     new_segment_ids=args.new_segment_ids,
                                                     mode="s2s", num_qkv=args.num_qkv,
                                                     s2s_special_token=args.s2s_special_token,
                                                     s2s_add_segment=args.s2s_add_segment,
                                                     s2s_share_segment=args.s2s_share_segment,
                                                     pos_shift=args.pos_shift))

    amp_handle = None
    if args.fp16 and args.amp:
        from apex import amp
        amp_handle = amp.init(enable_caching=True)
        logger.info("enable fp16 with amp")

    # Prepare model
    cls_num_labels = 2
    type_vocab_size = 6 + \
                      (1 if args.s2s_add_segment else 0) if args.new_segment_ids else 2
    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
        ["[MASK]", "[SEP]", "[S2S_SOS]"])

    def _get_token_id_set(s):
        r = None
        if s:
            w_list = []
            for w in s.split('|'):
                if w.startswith('[') and w.endswith(']'):
                    w_list.append(w.upper())
                else:
                    w_list.append(w)
            r = set(tokenizer.convert_tokens_to_ids(w_list))
        return r

    forbid_ignore_set = _get_token_id_set(args.forbid_ignore_word)
    not_predict_set = _get_token_id_set(args.not_predict_token)
    print(args.model_recover_path)
    for model_recover_path in glob.glob(args.model_recover_path.strip()):
        logger.info("***** Recover model: %s *****", model_recover_path)
        model_recover = torch.load(model_recover_path)
        model = BertForSeq2SeqDecoder.from_pretrained(args.bert_model, state_dict=model_recover,
                                                      num_seq_labels=len(label_list), num_labels=cls_num_labels,
                                                      num_rel=0,
                                                      type_vocab_size=type_vocab_size, task_idx=3,
                                                      mask_word_id=mask_word_id, search_beam_size=args.beam_size,
                                                      length_penalty=args.length_penalty, eos_id=eos_word_ids,
                                                      sos_id=sos_word_id,
                                                      forbid_duplicate_ngrams=args.forbid_duplicate_ngrams,
                                                      forbid_ignore_set=forbid_ignore_set,
                                                      not_predict_set=not_predict_set, ngram_size=args.ngram_size,
                                                      min_len=args.min_len,
                                                      mode=args.mode, max_position_embeddings=args.max_seq_length,
                                                      ffn_type=args.ffn_type,
                                                      num_qkv=args.num_qkv, seg_emb=args.seg_emb,
                                                      pos_shift=args.pos_shift)
        del model_recover

        if args.fp16:
            model.half()
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        torch.cuda.empty_cache()
        model.eval()
        next_i = 0
        max_src_length = args.max_seq_length - 2 - args.max_tgt_length

        input_lines = sorted(list(enumerate(input_lines)),
                             key=lambda x: -len(x[1]))

        output_labelp_line = [""] * len(input_lines)
        output_mlm_line = [""] * len(input_lines)
        mlm_prob_list = [None] * len(input_lines)
        labelp_prob_list = [None] * len(input_lines)
        total_batch = math.ceil(len(input_lines) / args.batch_size)

        time_start = time.time()
        with tqdm(total=total_batch) as pbar:
            while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + args.batch_size]
                buf_id = [x[0] for x in _chunk]
                buf = [x[1] for x in _chunk]
                next_i += args.batch_size
                max_a_len = max([len(x) for x in buf])
                instances = []
                for instance in [(x, max_a_len) for x in buf]:
                    for proc in bi_uni_pipeline:
                        instances.append(proc(instance))
                with torch.no_grad():
                    batch = batch_list_to_batch_tensors(
                        instances)
                    batch = [
                        t.to(device) if t is not None else None for t in batch]
                    input_ids, token_type_ids, masked_pos, input_mask, mask_qkv, task_idx = batch
                    mlm_logits, present_logits = model(input_ids, token_type_ids,
                                                       masked_pos, input_mask, task_idx=task_idx,
                                                       mask_qkv=mask_qkv)

                    mlm_id = torch.argmax(mlm_logits, dim=-1).tolist()
                    mlm_probs = F.softmax(mlm_logits, dim=2).detach().cpu().numpy().tolist()
                    present_id = torch.argmax(present_logits, dim=-1)
                    seq_labelp_probs = F.softmax(present_logits, dim=2).detach().cpu().numpy().tolist()

                    for i in range(len(buf)):
                        lp_ids = present_id[i]
                        output_lp_buf = convert_label_ids_to_tokens(lp_ids, label_list)
                        mlm_ids = mlm_id[i]
                        output_mlm_buf = tokenizer.convert_ids_to_tokens(mlm_ids)
                        output_tokens_mlm = []
                        output_tokens_lp = []
                        for t in output_mlm_buf:
                            output_tokens_mlm.append(t)
                        for t in output_lp_buf:
                            output_tokens_lp.append(t)
                        output_sequence_mlm = ' '.join(output_tokens_mlm)
                        output_sequence_labelp = ' '.join(output_tokens_lp)
                        output_mlm_line[buf_id[i]] = output_sequence_mlm
                        output_labelp_line[buf_id[i]] = output_sequence_labelp
                        labelp_prob_list[buf_id[i]] = seq_labelp_probs[i]
                        mlm_prob_list[buf_id[i]] = max(mlm_probs[i])

                pbar.update(1)
        time_end = time.time()

        if args.output_mlm_file:
            fn_lpout = args.output_mlm_file
        else:
            fn_lpout = model_recover_path + '.' + args.split + '.mlm'
        with open(fn_lpout, "w", encoding="utf-8") as fout:
            for l in output_labelp_line:
                fout.write(l)
                fout.write("\n")

        assert args.result_dir

        # avg_absent, absent_all = get_score_absent(
        #     path=args.result_dir,
        #     test_result_lines=output_mlm_line,
        #     test_result_prob_lines=mlm_prob_list,
        #     mask_num=args.mask_num,
        #     total_mask_num=args.mask_num,
        #     absent_type=args.absent_type
        # )
        return time_end - time_start, output_mlm_line, mlm_prob_list


if __name__ == "__main__":
    main()
