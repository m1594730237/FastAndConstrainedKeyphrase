PROJECT_PATH=/root_path
DATA_DIR=${PROJECT_PATH}/data/train_kp20k_meng_uncased_2mask_8ab_64all_3repeat_bix_stem_truncated2_fixed3_are
OUTPUT_DIR=${PROJECT_PATH}/output_model/UNILM_model/unilm_2_8_64mask_3R_T1_dataMengFBI_L384_stem_CW_seed42_are

MODEL_RECOVER_PATH=${PROJECT_PATH}/PTModel/unilm_pytorch/unilm1-base-cased.bin
export CUDA_VISIBLE_DEVICES=0,1,2,3
if [ ! -d ${OUTPUT_DIR} ]; then
  mkdir -p ${OUTPUT_DIR}/bert_save
  mkdir -p ${OUTPUT_DIR}/bert_log
fi
python3 biunilm/run_seq2seq_prefetcher.py --do_train \
  --fp16 --amp --num_workers 12 \
  --bert_model ${PROJECT_PATH}/PTModel/unilm_pytorch/bert-base-cased \
  --new_segment_ids --tokenized_input \
  --data_dir ${DATA_DIR} \
  --src_file seq.in --tgt_file seq.out \
  --labelp_file seq.labelp --labelc_file seq.labelc \
  --output_dir ${OUTPUT_DIR}/bert_save \
  --log_dir ${OUTPUT_DIR}/bert_log \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 384 --max_position_embeddings 384 \
  --train_batch_size 200 --gradient_accumulation_steps 1 \
  --learning_rate 0.00001 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 35 --max_pred 64 \
  --seed 42
