EVAL_SPLIT=kp20k.test
MODEL_FILE=model.28.bin
PROJECT_PATH=/root_path
#RESULT_DIR=${PROJECT_PATH}/data/test_kp20k_meng_kw_word_uncased
RESULT_DIR=${PROJECT_PATH}/data/test_acm_kw_word_uncased
MODEL_RECOVER_PATH=${PROJECT_PATH}/output_model/UNILM_model/unilm_2_8_64mask_3R_T1_dataMengFBI_L384_stem_CW/bert_save
export CUDA_VISIBLE_DEVICES=0
python3 -u biunilm/decode_all_eva.py \
  --bert_model ${PROJECT_PATH}/PTModel/unilm_pytorch/bert-base-cased \
  --new_segment_ids --mode s2s \
  --input_file ${RESULT_DIR}/seq.in \
  --result_dir ${RESULT_DIR} \
  --output_labelp_file ${RESULT_DIR}/seq.in.pred_p \
  --output_labelc_file ${RESULT_DIR}/seq.in.pred_c \
  --output_mlm_file ${RESULT_DIR}/seq.in.mlm \
  --model_recover_path ${MODEL_RECOVER_PATH}/${MODEL_FILE} \
  --max_seq_length 384 --max_tgt_length 64 \
  --batch_size 32 --top_n 6 \
  --mask_num 2
