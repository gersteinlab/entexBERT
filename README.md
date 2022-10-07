## Requirements
DNABERT (https://github.com/jerryji1993/DNABERT) \
pytorch                   1.7.1 \
cudatoolkit               11.0.221

## Example
KMER: 3, 4, 5 or 6\
MODEL_PATH: where the pre-trained DNABERT model is located
python3 1.ft_bert.py \
    --model_type ${model} \
    --tokenizer_name=dna$KMER \
    --model_name_or_path \$MODEL_PATH \
    --task_name dnaprom \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir \$DATA_PATH \
    --predict_dir \$DATA_PATH \
    --max_seq_length ${seq_len} \
    --per_gpu_eval_batch_size=${batch}   \
    --per_gpu_train_batch_size=${batch}   \
    --learning_rate ${lr} \
    --num_train_epochs ${ep} \
    --output_dir \$OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 5000 \
    --save_steps 20000 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --n_process 8 \
    --pred_layer ${layer} \
    --seed ${seed}
