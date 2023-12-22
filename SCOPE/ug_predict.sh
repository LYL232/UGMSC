#!/usr/bin/env bash
# -*- coding: utf-8 -*-

REPO_PATH=/path/to/SCOPE
BERT_PATH=$REPO_PATH/UGFPT
DATA_DIR=$REPO_PATH/data
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

ckpt_path=ug_outputs/bs32epoch30/checkpoint/epoch=14-df=80.1105-cf=78.8214.ckpt

OUTPUT_DIR=ug_outputs/predict
mkdir -p $OUTPUT_DIR

python -u finetune/ug_predict.py \
  --bert_path $BERT_PATH \
  --ckpt_path $ckpt_path \
  --data_dir $DATA_DIR \
  --save_path $OUTPUT_DIR \
  --label_file $DATA_DIR/test.sighan15.lbl.tsv \
  --gpus=0,


