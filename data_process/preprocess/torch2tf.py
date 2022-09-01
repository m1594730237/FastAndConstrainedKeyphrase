from transformers.convert_bert_pytorch_checkpoint_to_original_tf import convert_pytorch_checkpoint_to_tf
from transformers import BertModel, BertConfig
# from transformers.modeling_bert import BertModel
import os
import torch


def torch2tf():
    model_name = "bert-base-cased"
    cache_dir = os.path.join(".", "bert-base-cased")
    tf_cache_dir = os.path.join(".", "bert-base-cased-tf")
    pytorch_model_path = os.path.join(cache_dir, "unilm1-base-cased.bin")
    config_path = os.path.join(cache_dir, "bert_config.json")
    model_config = BertConfig.from_pretrained(config_path)
    model = BertModel.from_pretrained(
        pretrained_model_name_or_path=pytorch_model_path,
        config=model_config,
        state_dict=torch.load(pytorch_model_path, map_location=torch.device('cpu')),
        cache_dir=cache_dir
    )
    convert_pytorch_checkpoint_to_tf(
        model=model,
        ckpt_dir=tf_cache_dir,
        model_name=model_name
    )
