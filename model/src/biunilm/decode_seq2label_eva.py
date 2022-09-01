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
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import random
import pickle

from tagging2test import tagging2test
from tokenization import BertTokenizer, WhitespaceTokenizer
from modeling import BertForSeq2SeqDecoder
from modeling import BertForSeqLabel
from optimization import BertAdam, warmup_linear
from get_score import get_score, get_score_present

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

    label_list = ['O', 'B', 'I', 'N', '[S2S_CLS]', '[S2S_SEP]', '[SEP]', '[CLS]', '[S2S_SOS]']
    def convert_label_ids_to_tokens(id_list, label_list):
        id_str_list = []
        for i in id_list:
            id_str_list.append(label_list[i])
        return id_str_list

    bi_uni_pipeline = []
    bi_uni_pipeline.append(Preprocess4SeqLabel(list(tokenizer.vocab.keys()), \
                                      tokenizer.convert_tokens_to_ids, args.max_seq_length, \
                                      max_tgt_length=args.max_tgt_length, new_segment_ids=args.new_segment_ids,\
                                      mode="s2s", num_qkv=args.num_qkv, s2s_special_token=args.s2s_special_token,\
                                      s2s_add_segment=args.s2s_add_segment,\
                                      s2s_share_segment=args.s2s_share_segment,\
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
                    type_vocab_size=type_vocab_size, task_idx=3, mask_word_id=mask_word_id, search_beam_size=args.beam_size,
                    length_penalty=args.length_penalty, eos_id=eos_word_ids, sos_id=sos_word_id,
                    forbid_duplicate_ngrams=args.forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set,
                    not_predict_set=not_predict_set, ngram_size=args.ngram_size, min_len=args.min_len,
                    mode=args.mode, max_position_embeddings=args.max_seq_length, ffn_type=args.ffn_type,
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
        max_src_length = args.max_seq_length - 2 - args.max_tgt_length

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
                    seq_labelp_logits, seq_labelc_logits  = model(input_ids, token_type_ids,
                                   position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)

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
        if args.output_labelp_file:
            fn_lpout = args.output_labelp_file
        else:
            fn_lpout = model_recover_path+'.'+args.split+'.labelp'
        with open(fn_lpout, "w", encoding="utf-8") as fout:
            for l in output_labelp_line:
                fout.write(l)
                fout.write("\n")
        
        if args.output_labelc_file:
            fn_lcout = args.output_labelc_file
        else:
            fn_lcout = model_recover_path+'.'+args.split+'.labelc'
        with open(fn_lcout, "w", encoding="utf-8") as fout:
            for l in output_labelc_line:
                fout.write(l)
                fout.write("\n")
        
        with open(fn_lpout + ".prob", "wb") as fout:
            pickle.dump(labelp_prob_list, fout)
        with open(fn_lcout + ".prob", "wb") as fout:
            pickle.dump(labelc_prob_list, fout)

        assert args.result_dir
        get_score_present(
            path=args.result_dir,
            test_tagging_p_lines=output_labelp_line,
            test_tagging_p_prob_lines=labelp_prob_list,
        )

        tagging2test(
            input_path=args.result_dir,
            output_path=os.path.join(args.result_dir, "mlm_input"),
            tagging_c_lines=output_labelc_line,
            tagging_c_prob_lines=labelc_prob_list,
            tagging_p_lines=output_labelp_line,
            tagging_p_prob_lines=labelp_prob_list,
            total_mask_num=args.mask_num,
            top_n=args.top_n
        )


if __name__ == "__main__":
    main()
