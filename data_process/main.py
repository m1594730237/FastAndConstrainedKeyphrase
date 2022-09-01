import os

# from postprocess.test2kws import test2kws
from postprocess.get_score import get_score, get_score2
from preprocess.data_preprocess import preprocess_kp20k_test, preprocess_kp20k_train_debug, preprocess_semeval_test, \
    preprocess_meng, preprocess_test, preprocess_liu, preprocess_liu_train, sample_data, preprocess_acm
from statistics.mask_num import *
from load_data import *
from preprocess.bio_tagging import *
from preprocess.torch2tf import *


def run():
    file_list = [
        # (os.path.join(".", "kp20k_small", "kp20k_small_train.json"), False),
        # (os.path.join(".", "kp20k_small", "kp20k_small_train_meng.json"), False),
        # (os.path.join(".", "kp20k", "kp20k_train_debug_kws.json"), False),
        # (os.path.join(".", "kp20k", "kp20k_train_debug_no_kws.json"), False),
        # (os.path.join(".", "kp20k", "kp20k_test_kws.json"), True),
        # (os.path.join(".", "kp20k", "kp20k_test_no_kws.json"), True),
        # (os.path.join(".", "kp20k", "kp20k_train.json"), False),
        # (os.path.join(".", "kp20k", "kp20k_valid_1000_meng_kws.json"), True)
        # (os.path.join(".", "semeval", "semeval_test_kws.json"), True),
        # (os.path.join(".", "semeval", "semeval_test_no_kws.json"), True),
        (os.path.join(".", "kp20k", "kp20k_train_meng.json"), False),
        # (os.path.join(".", "kp20k", "kp20k_test_meng_kws.json"), True),
        # (os.path.join(".", "kp20k", "kp20k_test_meng_no_kws.json"), True),
        # (os.path.join(".", "semeval", "semeval_test_liu.json"), True),
        # (os.path.join(".", "semeval", "semeval_test_meng_kws.json"), True),
        # (os.path.join(".", "semeval", "semeval_test_meng_no_kws.json"), True),
        # (os.path.join(".", "nus", "nus_test_liu.json"), True),
        # (os.path.join(".", "nus", "nus_test_meng_kws.json"), True),
        # (os.path.join(".", "nus", "nus_test_meng_no_kws.json"), True),
        # (os.path.join(".", "krapivin", "krapivin_test_liu.json"), True),
        # (os.path.join(".", "krapivin", "krapivin_test_meng_kws.json"), True),
        # (os.path.join(".", "krapivin", "krapivin_test_meng_no_kws.json"), True),
        # (os.path.join(".", "inspec", "inspec_test_liu.json"), True),
        # (os.path.join(".", "inspec", "inspec_test_meng_kws.json"), True),
        # (os.path.join(".", "inspec", "inspec_test_meng_no_kws.json"), True),
        # (os.path.join(".", "kp20k_liu", "kp20k_test_liu.json"), True),
        # (os.path.join(".", "kp20k_liu", "kp20k_train_liu.json"), False),
        # (os.path.join(".", "kp20k_liu", "kp20k_small_train_liu.json"), False),
        # (os.path.join(".", "acm", "acm_kws.json"), True),
        # (os.path.join(".", "acm", "acm_no_kws.json"), True),
    ]
    num_mask_list = [
        # 2,
        # 3,
        5,
        # 30
    ]
    for filename, is_test in file_list:
        print("load data")
        kp20k = load_dataset(filename)
        print("bio tagging")
        #for num_mask in num_mask_list:
            #print(f"******* {filename}, {num_mask} *******")
            #mask_num(items=kp20k, num_mask=num_mask, split_percent=[0.5, 0.9, 0.95, 0.99],
            #         cached_path=os.path.join("cache", filename.split(os.sep)[-1]),
            #         auxiliary_sentence="phrase of is")
        bio_tagging_ps_cs(items=kp20k, num_center_mask=2, num_absent_mask=8, max_mask_num=64, repeat_num=3,
                          is_bix=True, is_stem=True, is_prompt=True,
                          is_test=is_test, split_percent=[0.5, 0.9, 0.95, 0.99],
                          center_as=f"phrase of {WILDCARD_AS} is", absent_as="other phrases are",
                          cached_path=os.path.join("cache", filename.split(os.sep)[-1]),
                          output_path="/root/data/train_kp20k_meng_uncased_2mask_8ab_64all_3repeat_bix_stem_truncated2_fixed3_are")
        # get_score2(kp20k, cached_path=os.path.join("cache", filename.split(os.sep)[-1]), mask_num=3, total_mask_num=6)


def run_pre():
    file_list = [
        # (os.path.join(".", "semeval", "semeval_test_meng17token.json"), True),
        # (os.path.join(".", "kp20k", "kp20k_test_meng17token.json"), True),
        (os.path.join(".", "kp20k", "kp20k_valid_1000_meng17token.json"), True),
        # (os.path.join(".", "kp20k", "kp20k_train_meng17token.json"), False),
        # (os.path.join(".", "nus", "nus_test_meng17token.json"), True),
        # (os.path.join(".", "krapivin", "krapivin_test_meng17token.json"), True),
        # (os.path.join(".", "inspec", "inspec_test_meng17token.json"), True),
    ]
    for filename, is_test in file_list:
        preprocess_meng(input_file=filename, output_file=filename.replace("meng17token", "meng"))


def run_test():
    files_list = [
        # (
        #     os.path.join("semeval", "semeval_test_meng.json"),
        #     os.path.join("semeval", "semeval_test_meng_kws.json"),
        #     os.path.join("semeval", "semeval_test_meng_no_kws.json")
        # ),
        # (
        #     os.path.join("kp20k", "kp20k_test_meng.json"),
        #     os.path.join("kp20k", "kp20k_test_meng_kws.json"),
        #     os.path.join("kp20k", "kp20k_test_meng_no_kws.json")
        # ),
        # (
        #     os.path.join("kp20k", "kp20k_train_meng.json"),
        #     os.path.join("kp20k", "kp20k_train_meng_kws.json"),
        #     os.path.join("kp20k", "kp20k_train_meng_no_kws.json")
        # ),
        # (
        #     os.path.join("nus", "nus_test_meng.json"),
        #     os.path.join("nus", "nus_test_meng_kws.json"),
        #     os.path.join("nus", "nus_test_meng_no_kws.json")
        # ),
        # (
        #     os.path.join("krapivin", "krapivin_test_meng.json"),
        #     os.path.join("krapivin", "krapivin_test_meng_kws.json"),
        #     os.path.join("krapivin", "krapivin_test_meng_no_kws.json")
        # ),
        # (
        #     os.path.join("inspec", "inspec_test_meng.json"),
        #     os.path.join("inspec", "inspec_test_meng_kws.json"),
        #     os.path.join("inspec", "inspec_test_meng_no_kws.json")
        # ),
        (
            os.path.join("kp20k", "kp20k_valid_1000_meng.json"),
            os.path.join("kp20k", "kp20k_valid_1000_meng_kws.json"),
            os.path.join("kp20k", "kp20k_valid_1000_meng_no_kws.json")
        ),
    ]
    for input_file, output_file_kws, output_file_nkws in files_list:
        preprocess_test(input_file, output_file_kws, output_file_nkws)


if __name__ == '__main__':
    # preprocess_semeval_test()
    # preprocess_kp20k_test()
    # preprocess_kp20k_train_debug()
    run()
    # preprocess_acm(
    #     input_file=os.path.join("acm", "acm-102k.trec.gz"),
    #     output_file_kw=os.path.join("acm", "acm_kws.json"),
    #     output_file_nkw=os.path.join("acm", "acm_no_kws.json")
    # )
    # preprocess_liu_train(input_data_path=os.path.join("output", "kp20k_train_liu.json"),
    #                      input_liu_file=os.path.join("kp20k_liu", "kp20k.train.seq.out"),
    #                      output_path=os.path.join("output", "kp20k_train_liu_bix_raw.json"))
    # preprocess_liu(os.path.join("kp20k_liu", "kp20k.train.seq.in"),
    #                os.path.join(".", "kp20k", "kp20k_train_meng_kws.json"),
    #                os.path.join("kp20k_liu", "kp20k_train_liu.json"))
    # sample_data(input_file=os.path.join(".", "kp20k", "kp20k_valid_meng17token.json"),
    #             output_file=os.path.join(".", "kp20k", "kp20k_valid_1000_meng17token.json"),
    #             num=1000)
    # run_pre()
    # run_test()
    # torch2tf()
