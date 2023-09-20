PRETRAINED_DIR="pretrained"
DATE_DIR="data"
OUTPUT_DIR="output"
YEAR="15"

python -m torch.distributed.launch --master_port=453$(($RANDOM % 90 + 10)) --nproc_per_node=1 src/run_ugmodel.py \
  --model_type ug \
  --model_name_or_path $PRETRAINED_DIR \
  --image_model_type 0 \
  --output_dir $OUTPUT_DIR \
  --do_predict \
  --data_dir $DATE_DIR \
  --train_file trainall.times2.pkl \
  --dev_file test.sighan$YEAR.pkl \
  --dev_label_file test.sighan$YEAR.lbl.tsv \
  --predict_file test.sighan$YEAR.pkl \
  --predict_label_file test.sighan$YEAR.lbl.tsv \
  --per_gpu_eval_batch_size 50 \
  --eval_all_checkpoints \
  --resfonts font3_fanti
