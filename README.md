# DeepTrace
Here we present the codes of DeepTrace.

This repository includes: source codes of pre-trainng, finetune and data samples. This package is still under development, as more features will be included gradually.

## Requirements and Setup
Python version >= 3.6

PyTorch version >= 1.10.0
```
# clone the repository
git clone https://github.com/Bamrock/DeepTrace.git
cd DeepTrace
pip install -r requirements.txt
```

(Optional, install apex for fp16 training)

```
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


## Pre-training

```
python -m torch.distributed.launch --nproc_per_node= \
    --nnodes= --node_rank= --master_addr= \
    --master_port= \
    pretrain.py \
    --output_dir $OUTPUT_PATH \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --n_gpu= \
    --gradient_accumulation_steps 10 \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 16 \
    --save_steps 500 \
    --save_total_limit 20 \
    --max_steps -1 \
    --evaluate_during_training \
    --logging_steps 500 \
    --line_by_line \
    --learning_rate 3e-4 \
    --block_size 200 \
    --adam_epsilon 1e-8 \
    --weight_decay 0.01 \
    --beta1 0.9 \
    --beta2 0.98 \
    --mlm_probability 0.05 \
    --warmup_steps 1000 \
    --overwrite_output_dir \
    --n_process 24
```

Add --fp16 tag if you want to perfrom mixed precision. (You have to install the 'apex' from source first).


## Finetune 

```

python -m torch.distributed.launch --nproc_per_node= \
    --nnodes= --node_rank= --master_addr= \
    --master_port= \
    fine_tuning.py \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_train \
    --do_eval \
    --data_dir $DATA_PATH \
    --max_seq_length 200 \
    --per_gpu_eval_batch_size=16   \
    --per_gpu_train_batch_size=16   \
    --learning_rate 5e-5 \
    --num_train_epochs 5.0 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 1000 \
    --save_steps -1 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --n_process 24
```

Add --fp16 tag if you want to perfrom mixed precision. (You have to install the 'apex' from source first).


## Prediction

```

python fine_tuning.py \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_predict \
    --data_dir $DATA_PATH  \
    --max_seq_length 200 \
    --per_gpu_pred_batch_size=64   \
    --output_dir $MODEL_PATH \
    --predict_dir $PREDICTION_PATH \
    --n_process 24
```
Add --fp16 tag if you want to perfrom mixed precision. (You have to install the 'apex' from source first).

Our repository references the code of DNABERT: https://github.com/jerryji1993/DNABERT

