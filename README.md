# Fast and Constrained Absent Keyphrase Generation by Prompt-Based Learning

This repository contains the data and code for paper:

> **Fast and Constrained Absent Keyphrase Generation by Prompt-Based Learning**,<br/>
> Huanqin Wu, Baijiaxin Ma, Wei Liu, Tao Chen, Dan Nie, <br/>
> To appear: Thirty-Seventh AAAI Conference on Artificial Intelligence.<br/>
> [arXiv](https://www.aaai.org/AAAI22Papers/AAAI-4989.WuH.pdf)

### Environment

- prepare for APEX

```shell
 	apt-get update
    apt-get install -y vim wget ssh

    PWD_DIR=$(pwd)
    cd $(mktemp -d)
    git clone -q https://github.com/NVIDIA/apex.git
    cd apex
    git reset --hard 1603407bf49c7fc3da74fceb6a6c7b47fece2ef8
    python setup.py install --user --cuda_ext --cpp_ext
    cd $PWD_DIR
```

- other packages

```shell
    pip install --user tensorboardX six numpy tqdm path.py pandas scikit-learn lmdb pyarrow py-lz4framed methodtools py-rouge pyrouge nltk
    python -c "import nltk; nltk.download('punkt')"
    pip install -e git://github.com/Maluuba/nlg-eval.git#egg=nlg-eval
```

get pretrained models from:  https://unilm.blob.core.windows.net/ckpt/unilm1-base-cased.bin

### Data Processing

Run main.py directly to process the KP20K training data:

```shell
    # run data processing
    python3 data_process/main.py
```

The main.py call the bio_tagging_ps_cs function in data_process/prepocess/bio_tagging.py to process the KP20K training data into the model input format.

### Run Model

The following three commands can be used for model training, validation and prediction:

```shell
    sh model/src/start_train.sh
    sh model/src/start_eval_infer.sh
    sh model/src/start_infer_acm.sh
```